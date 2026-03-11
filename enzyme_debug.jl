using Revise
using Molly
using Enzyme
using Unitful
using InteractiveUtils
using BenchmarkTools
using Base.Threads

# Set loose type analysis to help Enzyme with Julia 1.11's complex type inference
Enzyme.API.looseTypeAnalysis!(true)

# ==============================================================================
# 1. PURELY FUNCTIONAL HELPERS (ENZYME-STABLE)
# ==============================================================================

@inline function _zero_specific_recursive!(lists::Tuple)
    list = first(lists)
    if length(list.inters) > 0
        z = zero(list.inters[1])
        fill!(list.inters, z)
    end
    _zero_specific_recursive!(Base.tail(lists))
end
_zero_specific_recursive!(::Tuple{}) = nothing

@inline _update_pairwise(inters::Tuple, params, idxs::Tuple) = 
    (_update_pairwise_recursive(inters, params, idxs))

@inline _update_pairwise_recursive(::Tuple{}, params, ::Tuple{}) = ()
@inline function _update_pairwise_recursive(inters::Tuple, params, idxs::Tuple)
    new_inter = Molly.inject_interaction(first(inters), params, first(idxs)...)
    return (new_inter, _update_pairwise_recursive(Base.tail(inters), params, Base.tail(idxs))...)
end

@inline _update_general(inters::Tuple, params, idxs::Tuple) = 
    (_update_general_recursive(inters, params, idxs))

@inline _update_general_recursive(::Tuple{}, params, ::Tuple{}) = ()
@inline function _update_general_recursive(inters::Tuple, params, idxs::Tuple)
    new_inter = Molly.inject_interaction(first(inters), params, first(idxs)...)
    return (new_inter, _update_general_recursive(Base.tail(inters), params, Base.tail(idxs))...)
end

@inline _update_specific(lists::Tuple, params, idxs::Tuple) = 
    (_update_specific_recursive(lists, params, idxs))

@inline _update_specific_recursive(::Tuple{}, params, ::Tuple{}) = ()
@inline function _update_specific_recursive(lists::Tuple, params, idxs::Tuple)
    list = first(lists)
    idx = first(idxs)
    
    if isempty(idx)
        return (list, _update_specific_recursive(Base.tail(lists), params, Base.tail(idxs))...)
    else
        new_inters = Molly.inject_interaction.(list.inters, Ref(params), idx...)
        new_list = _reconstruct_list(list, new_inters)
        return (new_list, _update_specific_recursive(Base.tail(lists), params, Base.tail(idxs))...)
    end
end

_reconstruct_list(l::InteractionList1Atoms, i) = InteractionList1Atoms(l.is, i, l.types)
_reconstruct_list(l::InteractionList2Atoms, i) = InteractionList2Atoms(l.is, l.js, i, l.types)
_reconstruct_list(l::InteractionList3Atoms, i) = InteractionList3Atoms(l.is, l.js, l.ks, i, l.types)
_reconstruct_list(l::InteractionList4Atoms, i) = InteractionList4Atoms(l.is, l.js, l.ks, l.ls, i, l.types)

# ==============================================================================
# 2. CORE EVALUATION FUNCTIONS
# ==============================================================================

function evaluate_frame_energy(params::Vector{FT}, sys_ref::System{D, AT, FT}, 
                               coords_nounits, box_nounits, neighbors, 
                               atom_idxs, pairwise_idxs, specific_idxs, general_idxs) where {D, AT, FT}
                               
    idx_mass, idx_σ, idx_ϵ = atom_idxs
    
    new_atoms = Molly.inject_atom.(sys_ref.atoms, Ref(params), idx_mass, idx_σ, idx_ϵ)
    
    new_pairwise = _update_pairwise(sys_ref.pairwise_inters, params, pairwise_idxs)
    new_specific = _update_specific(sys_ref.specific_inter_lists, params, specific_idxs)
    new_general  = _update_general(sys_ref.general_inters, params, general_idxs)
    
    sys_final = typeof(sys_ref)(
        new_atoms,             
        coords_nounits, 
        box_nounits,
        sys_ref.velocities,
        sys_ref.atoms_data,
        sys_ref.topology,
        new_pairwise,                 
        new_specific, 
        new_general,                  
        sys_ref.constraints,
        sys_ref.virtual_sites,
        sys_ref.virtual_site_flags,
        sys_ref.neighbor_finder,
        sys_ref.loggers,
        sys_ref.df,
        sys_ref.force_units,
        sys_ref.energy_units,
        sys_ref.k,
        sys_ref.masses,
        sys_ref.total_mass,
        sys_ref.data
    )

    return FT(potential_energy(sys_final, neighbors; n_threads=1))
end

function evaluate_frame_gradients(sys_ref::System{D, AT, FT}, 
                                  coords_nounits, box_nounits, neighbors, 
                                  params::Vector{FT}, grads_enzyme::Vector{FT}, 
                                  atom_idxs, pairwise_idxs, specific_idxs, general_idxs) where {D, AT, FT}
    
    fill!(grads_enzyme, zero(FT))
    
    result = autodiff(
        set_runtime_activity(ReverseWithPrimal), 
        evaluate_frame_energy, 
        Active, 
        Duplicated(params, grads_enzyme), 
        Const(sys_ref),
        Const(coords_nounits), 
        Const(box_nounits),
        Const(neighbors),
        Const(atom_idxs),
        Const(pairwise_idxs),
        Const(specific_idxs),
        Const(general_idxs)
    )
    
    return result[2]
end

# ==============================================================================
# 3. BENCHMARK SUITE (GC-ISOLATED)
# ==============================================================================

function benchmark_serial(n_frames, sys_ref, coords, box, nbrs, params, a_idx, p_idx, s_idx, g_idx)
    grads = zeros(Float32, length(params))
    
    for i in 1:n_frames
        # ISOLATION: Disable GC while Enzyme dynamically traces the allocations
        GC.enable(false)
        
        evaluate_frame_gradients(sys_ref, coords, box, nbrs, params, grads, a_idx, p_idx, s_idx, g_idx)
        
        # Safely sweep memory after Enzyme finishes the reverse pass
        GC.enable(true)
        GC.gc(false)
    end
end

function benchmark_parallel_safe(n_frames, sys_ref, coords, box, nbrs, params, a_idx, p_idx, s_idx, g_idx)
    n_threads = Threads.nthreads()
    thread_grads = [zeros(Float32, length(params)) for _ in 1:n_threads]
    
    # BATCHING: We process frames in groups equal to the number of threads.
    # This ensures threads execute fully before we sweep the enormous allocation spikes.
    batch_size = n_threads 
    
    for b in 1:batch_size:n_frames
        end_idx = min(b + batch_size - 1, n_frames)
        
        # Isolate the entire threaded batch from the GC
        GC.enable(false)
        
        Threads.@threads :static for i in b:end_idx
            tid = Threads.threadid()
            evaluate_frame_gradients(sys_ref, coords, box, nbrs, params, thread_grads[tid], a_idx, p_idx, s_idx, g_idx)
        end
        
        # Re-enable and explicitly clean up the batch allocations synchronously
        GC.enable(true)
        GC.gc(false)
    end
end

# ==============================================================================
# 4. SETUP AND EXECUTION 
# ==============================================================================

FT = Float32
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir   = joinpath(data_dir, "force_fields")
ff = MolecularForceField(FT, joinpath.(ff_dir, ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"])...; units=true)

sys_base = System("ethanol_vac.pdb", ff; nonbonded_method=:none, loggers=NamedTuple())
sys_nounits = System(ustrip(sys_base); coords=ustrip.(sys_base.coords), boundary=ustrip(sys_base.boundary), loggers=NamedTuple())
neighbors = find_neighbors(sys_nounits)

# Find first HarmonicBond list
bond_list_idx = findfirst(l -> eltype(l.inters) <: HarmonicBond, sys_nounits.specific_inter_lists)
bond_list = sys_nounits.specific_inter_lists[bond_list_idx]
theta_active = Float32[bond_list.inters[1].k, bond_list.inters[1].r0]

# Setup indices
atom_idxs = (zeros(Int, length(sys_nounits.atoms)), zeros(Int, length(sys_nounits.atoms)), zeros(Int, length(sys_nounits.atoms)))
pairwise_idxs = Tuple(() for _ in 1:length(sys_nounits.pairwise_inters))
general_idxs  = Tuple(() for _ in 1:length(sys_nounits.general_inters))

specific_idxs_array = []
for i in 1:length(sys_nounits.specific_inter_lists)
    if i == bond_list_idx
        idx_k, idx_r0 = zeros(Int, length(bond_list.inters)), zeros(Int, length(bond_list.inters))
        idx_k[1], idx_r0[1] = 1, 2
        push!(specific_idxs_array, (idx_k, idx_r0))
    else
        push!(specific_idxs_array, ())
    end
end
specific_idxs = Tuple(specific_idxs_array)

grads_enzyme = zeros(FT, length(theta_active))
coords = ustrip.(sys_base.coords)
box    = ustrip(sys_base.boundary)

println("\n--- PRIMAL TYPE STABILITY CHECK ---")
@code_warntype evaluate_frame_energy(theta_active, sys_nounits, coords, box, neighbors, 
                                     atom_idxs, pairwise_idxs, specific_idxs, general_idxs)

println("\n--- RUNNING ENZYME AUTODIFF ---")
energy = evaluate_frame_gradients(sys_nounits, coords, box, neighbors, theta_active, grads_enzyme, 
                                   atom_idxs, pairwise_idxs, specific_idxs, general_idxs)

println("Energy:    ", energy)
println("Gradients: ", grads_enzyme)

println("\n=======================================================")
println("                 BENCHMARKING SUITE")
println("=======================================================\n")
println("Threads active: $(Threads.nthreads())")

n_test_frames = 1100

println("\n--- SERIAL BENCHMARK (100 Frames) ---")
@btime benchmark_serial($n_test_frames, $sys_nounits, $coords, $box, $neighbors, $theta_active, 
                        $atom_idxs, $pairwise_idxs, $specific_idxs, $general_idxs)

println("\n--- PARALLEL SAFE BENCHMARK (100 Frames) ---")
@btime benchmark_parallel_safe($n_test_frames, $sys_nounits, $coords, $box, $neighbors, $theta_active, 
                               $atom_idxs, $pairwise_idxs, $specific_idxs, $general_idxs)