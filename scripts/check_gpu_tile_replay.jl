#!/usr/bin/env julia

using Printf
using CUDA
using Molly
using Unitful

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))

const DEFAULT_PDB = joinpath(@__DIR__, "..", "ethanol_solv.pdb")
const DEFAULT_FLOAT_TYPES = (Float32, Float64)
const DEFAULT_LAMBDAS = (1.0, 0.5, 0.15, 0.0)
const DEFAULT_SOLUTE = 1:9
const DEFAULT_SCHEDULER = :ele_scaled
const DEFAULT_DEVICE_ID = 1
const DEFAULT_LJ_CUTOFF_MODEL = :distance
const DEFAULT_LJ_ACTIVATION_FRACTION = 0.8
const DEFAULT_NONBONDED_ENERGY_TYPE = nothing

function usage()
    println(
        """
        Usage:
          julia +1.11 scripts/check_gpu_tile_replay.jl [options]

        Options:
          --pdb PATH                Input structure. Default: ethanol_solv.pdb
          --float-types CSV         Float32,Float64 or a subset. Default: Float32,Float64
          --lambda-values CSV       Soft-core lambda values. Default: 1.0,0.5,0.15,0.0
          --solute-indices SPEC     Solute atoms, e.g. 1:9 or 1,2,3. Default: 1:9
          --scheduler NAME          Lambda scheduler. Default: ele_scaled
          --device-id N             CUDA device id. Default: 1
          --lj-cutoff-model NAME    distance|shifted_potential|shifted_force|cubic_spline
          --lj-activation-frac X    Cubic-spline activation fraction of rc. Default: 0.8
          --nonbonded-energy-type T nothing|Float32|Float64 override for Molly pairwise energy evaluation
          --help                    Show this message
        """
    )
end

function parse_csv_strings(value::AbstractString)
    parts = String[]
    for item in split(value, ",")
        stripped = strip(item)
        isempty(stripped) && continue
        push!(parts, stripped)
    end
    return unique(parts)
end

function parse_float_types(value::AbstractString)
    mapping = Dict("float32" => Float32, "float64" => Float64)
    types = DataType[]
    for item in parse_csv_strings(lowercase(value))
        haskey(mapping, item) || throw(ArgumentError("Unsupported float type `$item`."))
        push!(types, mapping[item])
    end
    isempty(types) && throw(ArgumentError("--float-types cannot be empty."))
    return Tuple(types)
end

function parse_float_values(value::AbstractString)
    vals = Float64[]
    for item in parse_csv_strings(value)
        push!(vals, parse(Float64, item))
    end
    isempty(vals) && throw(ArgumentError("--lambda-values cannot be empty."))
    return Tuple(vals)
end

function parse_range_or_csv(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && return Int[]
    if occursin(":", stripped)
        parts = split(stripped, ":")
        length(parts) == 2 || throw(ArgumentError("Invalid range `$value`; expected `start:stop`."))
        return collect(parse(Int, parts[1]):parse(Int, parts[2]))
    end
    return parse.(Int, parse_csv_strings(stripped))
end

function parse_nonbonded_energy_type(value::AbstractString)
    lowered = lowercase(strip(value))
    if lowered in ("nothing", "native", "system", "default")
        return nothing
    elseif lowered == "float32"
        return Float32
    elseif lowered == "float64"
        return Float64
    end
    throw(ArgumentError("Unsupported --nonbonded-energy-type=$value"))
end

function parse_cli(args::Vector{String})
    opts = Dict{Symbol, Any}(
        :pdb => abspath(DEFAULT_PDB),
        :float_types => DEFAULT_FLOAT_TYPES,
        :lambda_values => DEFAULT_LAMBDAS,
        :solute_indices => collect(DEFAULT_SOLUTE),
        :scheduler => DEFAULT_SCHEDULER,
        :device_id => DEFAULT_DEVICE_ID,
        :lj_cutoff_model => DEFAULT_LJ_CUTOFF_MODEL,
        :lj_activation_fraction => DEFAULT_LJ_ACTIVATION_FRACTION,
        :nonbonded_energy_type => DEFAULT_NONBONDED_ENERGY_TYPE,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            usage()
            exit(0)
        elseif arg == "--pdb"
            i += 1
            i <= length(args) || throw(ArgumentError("--pdb requires a value"))
            opts[:pdb] = abspath(args[i])
            i += 1
        elseif arg == "--float-types"
            i += 1
            i <= length(args) || throw(ArgumentError("--float-types requires a value"))
            opts[:float_types] = parse_float_types(args[i])
            i += 1
        elseif arg == "--lambda-values"
            i += 1
            i <= length(args) || throw(ArgumentError("--lambda-values requires a value"))
            opts[:lambda_values] = parse_float_values(args[i])
            i += 1
        elseif arg == "--solute-indices"
            i += 1
            i <= length(args) || throw(ArgumentError("--solute-indices requires a value"))
            opts[:solute_indices] = parse_range_or_csv(args[i])
            i += 1
        elseif arg == "--scheduler"
            i += 1
            i <= length(args) || throw(ArgumentError("--scheduler requires a value"))
            opts[:scheduler] = Symbol(lowercase(strip(args[i])))
            i += 1
        elseif arg == "--device-id"
            i += 1
            i <= length(args) || throw(ArgumentError("--device-id requires a value"))
            opts[:device_id] = parse(Int, args[i])
            i += 1
        elseif arg == "--lj-cutoff-model"
            i += 1
            i <= length(args) || throw(ArgumentError("--lj-cutoff-model requires a value"))
            opts[:lj_cutoff_model] = Symbol(lowercase(strip(args[i])))
            i += 1
        elseif arg == "--lj-activation-frac"
            i += 1
            i <= length(args) || throw(ArgumentError("--lj-activation-frac requires a value"))
            opts[:lj_activation_fraction] = parse(Float64, args[i])
            i += 1
        elseif arg == "--nonbonded-energy-type"
            i += 1
            i <= length(args) || throw(ArgumentError("--nonbonded-energy-type requires a value"))
            opts[:nonbonded_energy_type] = parse_nonbonded_energy_type(args[i])
            i += 1
        else
            throw(ArgumentError("Unknown option: $arg"))
        end
    end

    isfile(opts[:pdb]) || throw(ArgumentError("PDB file not found: $(opts[:pdb])"))
    CUDA.functional() || throw(ArgumentError("CUDA is not available"))
    ext = Base.get_extension(Molly, :MollyCUDAExt)
    ext === nothing && throw(ArgumentError("MollyCUDAExt is not loaded"))
    opts[:lj_cutoff_model] in (:distance, :shifted_potential, :shifted_force, :cubic_spline) ||
        throw(ArgumentError("Unsupported --lj-cutoff-model=$(opts[:lj_cutoff_model])"))
    0.0 < opts[:lj_activation_fraction] < 1.0 ||
        throw(ArgumentError("--lj-activation-frac must be in (0, 1)"))
    return opts
end

format_real(value::Real) = @sprintf("%.8f", Float64(value))
format_bool(value::Bool) = value ? "true" : "false"

function upper_tile_index(i::Int, j::Int, n_blocks::Int)
    r = Int64(i - 1)
    n = Int64(n_blocks)
    return Int((r * (2 * n - r + 1)) ÷ 2 + Int64(j - i + 1))
end

function energy_units_val(sys)
    return Val(sys.energy_units)
end

function clone_solute_atoms(base_atoms_cpu, solute_index_set::Set{Int}, lambda_value, ::Type{FT}) where {FT <: AbstractFloat}
    AtomT = typeof(first(base_atoms_cpu))
    atoms_new = AtomT[]
    λ = FT(lambda_value)
    for atom in base_atoms_cpu
        if atom.index in solute_index_set
            push!(atoms_new, Molly.Atom(atom.index, atom.atom_type, atom.mass, atom.charge, atom.σ, atom.ϵ, λ, Molly.InsertRole))
        else
            push!(atoms_new, Molly.Atom(atom.index, atom.atom_type, atom.mass, atom.charge, atom.σ, atom.ϵ, atom.λ, atom.alch_role))
        end
    end
    return atoms_new
end

function only_interaction(sys, T)
    matches = [inter for inter in Tuple(sys.pairwise_inters) if inter isa T]
    length(matches) == 1 || throw(ArgumentError("Expected exactly one $(T), found $(length(matches))"))
    return only(matches)
end

function replace_lj_cutoff(inter, model::Symbol, activation_fraction::Float64)
    base_cutoff = inter.cutoff
    base_cutoff isa Molly.DistanceCutoff || throw(ArgumentError("Expected DistanceCutoff, got $(typeof(base_cutoff))"))
    rc = base_cutoff.dist_cutoff
    new_cutoff = if model == :distance
        Molly.DistanceCutoff(rc)
    elseif model == :shifted_potential
        Molly.ShiftedPotentialCutoff(rc)
    elseif model == :shifted_force
        Molly.ShiftedForceCutoff(rc)
    elseif model == :cubic_spline
        Molly.CubicSplineCutoff(rc * typeof(ustrip(rc))(activation_fraction), rc)
    else
        throw(ArgumentError("Unsupported LJ cutoff model `$model`"))
    end
    return Molly.LennardJonesSoftCoreGapsys(
        cutoff=new_cutoff,
        α=inter.α,
        use_neighbors=inter.use_neighbors,
        shortcut=inter.shortcut,
        σ_mixing=inter.σ_mixing,
        ϵ_mixing=inter.ϵ_mixing,
        λ_mixing=inter.λ_mixing,
        scheduler=inter.scheduler,
        weight_special=inter.weight_special,
    )
end

function build_gpu_system(
    pdb_path::AbstractString,
    ::Type{FT},
    lambda_value,
    solute_indices,
    scheduler_name::Symbol,
    device_id::Int;
    lj_cutoff_model::Symbol=DEFAULT_LJ_CUTOFF_MODEL,
    lj_activation_fraction::Float64=DEFAULT_LJ_ACTIVATION_FRACTION,
    nonbonded_energy_type=DEFAULT_NONBONDED_ENERGY_TYPE,
) where {FT <: AbstractFloat}
    cfg = AWHGrads.default_simulation_config(; FT=FT, AT=CuArray, device_id=device_id)
    AWHGrads.apply_simulation_config!(cfg)
    base_sys = Molly.System(
        pdb_path,
        AWHGrads.ff;
        array_type=cfg.AT,
        units=true,
        nonbonded_method=:cutoff,
        nonbonded_energy_type=nonbonded_energy_type,
    )

    scheduler = AWHGrads.resolve_lambda_scheduler(scheduler_name)
    awh_control = AWHGrads.AWHControlConfig()
    lj_0 = only_interaction(base_sys, Molly.LennardJones)
    coul_0 = only_interaction(base_sys, Molly.CoulombReactionField)
    lj_sc = AWHGrads.build_lj_softcore_interaction(lj_0, awh_control, scheduler, :gapsys, false)
    lj_sc = replace_lj_cutoff(lj_sc, lj_cutoff_model, lj_activation_fraction)
    coul_sc = AWHGrads.build_coulomb_softcore_interaction(coul_0, awh_control, scheduler, :gapsys_rf, false)

    atoms_cpu = collect(Molly.from_device(base_sys.atoms))
    atoms_sc = clone_solute_atoms(atoms_cpu, Set(solute_indices), lambda_value, FT)
    return Molly.System(
        base_sys;
        atoms=Molly.to_device(atoms_sc, Molly.array_type(base_sys.coords)),
        pairwise_inters=(coul_sc, lj_sc),
    )
end

function pairwise_interactions(sys)
    return Tuple(filter(Molly.use_neighbors, Tuple(sys.pairwise_inters)))
end

function pairwise_component_rows(sys)
    rows = NamedTuple[]
    for (idx, inter) in enumerate(Tuple(sys.pairwise_inters))
        push!(rows, (label=string(nameof(typeof(inter))), inters=(inter,)))
    end
    return rows
end

function gpu_pairwise_energy(sys, buffers, inters_tuple, step_n::Int)
    T = Molly.float_type(sys)
    fill!(buffers.pe_vec_nounits, zero(T))
    Molly.pairwise_pe_loop_gpu!(buffers.pe_vec_nounits, buffers, sys, inters_tuple, nothing, step_n)
    CUDA.synchronize()
    return only(Molly.from_device(buffers.pe_vec_nounits)) * sys.energy_units
end

function snapshot_tile_state(buffers)
    num_tiles = Int(only(Molly.from_device(buffers.num_interacting_tiles)))
    overflow = Int(only(Molly.from_device(buffers.interacting_tiles_overflow)))
    return (
        coords=collect(Molly.from_device(buffers.coords_reordered)),
        velocities=collect(Molly.from_device(buffers.velocities_reordered)),
        atoms=collect(Molly.from_device(buffers.atoms_reordered)),
        compressed_masks=Array(Molly.from_device(buffers.compressed_masks)),
        tiles_i=Array(Molly.from_device(buffers.interacting_tiles_i))[1:num_tiles],
        tiles_j=Array(Molly.from_device(buffers.interacting_tiles_j))[1:num_tiles],
        tiles_type=Array(Molly.from_device(buffers.interacting_tiles_type))[1:num_tiles],
        num_tiles=num_tiles,
        overflow=overflow,
    )
end

function bit_is_set(mask::UInt32, lane_j::Int)
    target_bit = UInt32(1) << (32 - lane_j)
    return (mask & target_bit) != 0
end

function replay_pairwise_from_gpu_tiles(snapshot, boundary, inters_tuple, energy_units, dist_cutoff_sq, step_n::Int, ::Type{FT}) where {FT <: AbstractFloat}
    coords = snapshot.coords
    velocities = snapshot.velocities
    atoms = snapshot.atoms
    masks = snapshot.compressed_masks
    n_atoms = length(coords)
    n_blocks = cld(n_atoms, 32)
    total = zero(FT) * energy_units

    for tile_idx in 1:snapshot.num_tiles
        tile_i = Int(snapshot.tiles_i[tile_idx])
        tile_j = Int(snapshot.tiles_j[tile_idx])
        mask_idx = upper_tile_index(tile_i, tile_j, n_blocks)

        row_start = (tile_i - 1) * 32 + 1
        col_start = (tile_j - 1) * 32 + 1
        row_stop = min(row_start + 31, n_atoms)
        col_stop = min(col_start + 31, n_atoms)

        for lane_i in 1:(row_stop - row_start + 1)
            idx_i = row_start + lane_i - 1
            eligible_mask = masks[lane_i, 1, mask_idx]
            special_mask = masks[lane_i, 2, mask_idx]
            lane_j_start = tile_i == tile_j ? lane_i + 1 : 1

            for lane_j in lane_j_start:(col_stop - col_start + 1)
                bit_is_set(eligible_mask, lane_j) || continue

                idx_j = col_start + lane_j - 1
                coord_i = coords[idx_i]
                coord_j = coords[idx_j]
                dr = Molly.vector(coord_i, coord_j, boundary)
                r2 = sum(abs2, dr)
                r2 <= dist_cutoff_sq || continue

                special = bit_is_set(special_mask, lane_j)
                pe = Molly.sum_pairwise_potentials(
                    inters_tuple,
                    atoms[idx_i],
                    atoms[idx_j],
                    Val(energy_units),
                    special,
                    coord_i,
                    coord_j,
                    boundary,
                    velocities[idx_i],
                    velocities[idx_j],
                    step_n,
                )
                total += pe[1]
            end
        end
    end

    return total
end

function build_pairwise_only_system(sys, inters_tuple)
    return Molly.System(sys; pairwise_inters=inters_tuple, specific_inter_lists=(), general_inters=())
end

function host_pairwise_energy(sys_host, neighbors_host, inters_tuple)
    pairwise_sys = build_pairwise_only_system(sys_host, inters_tuple)
    return Molly.potential_energy(pairwise_sys, neighbors_host; n_threads=1)
end

function host_nonpairwise_energy(sys_host, neighbors_host)
    nonpairwise_sys = Molly.System(sys_host; pairwise_inters=())
    return Molly.potential_energy(nonpairwise_sys, neighbors_host; n_threads=1)
end

function tile_stats(snapshot)
    clean = count(==(UInt8(0)), snapshot.tiles_type)
    masked = snapshot.num_tiles - clean
    return (clean=clean, masked=masked, overflow=snapshot.overflow)
end

function abs_kjmol(value)
    return abs(Float64(ustrip(u"kJ/mol", value)))
end

function signed_kjmol(value)
    return Float64(ustrip(u"kJ/mol", value))
end

function print_delta(io::IO, label::AbstractString, lhs, rhs)
    delta = lhs - rhs
    println(io, "  ", label, " = ", format_real(signed_kjmol(delta)), " kJ/mol")
end

function print_component_block(io::IO, label::AbstractString, gpu_value, replay_value, host_value)
    println(io, "  ", label)
    println(io, "    gpu_native = ", format_real(ustrip(u"kJ/mol", gpu_value)))
    println(io, "    cpu_replay_gpu_tiles = ", format_real(ustrip(u"kJ/mol", replay_value)))
    println(io, "    host_from_device = ", format_real(ustrip(u"kJ/mol", host_value)))
    println(io, "    gpu_minus_replay = ", format_real(signed_kjmol(gpu_value - replay_value)), " kJ/mol")
    println(io, "    replay_minus_host = ", format_real(signed_kjmol(replay_value - host_value)), " kJ/mol")
end

function conclusion_line(gpu_pairwise, replay_pairwise, host_pairwise, tol_kj_mol::Float64)
    gpu_vs_replay = abs_kjmol(gpu_pairwise - replay_pairwise)
    replay_vs_host = abs_kjmol(replay_pairwise - host_pairwise)
    if gpu_vs_replay > tol_kj_mol
        return "GPU tiled pairwise path diverges from CPU replay of exact GPU tiles"
    elseif replay_vs_host > tol_kj_mol
        return "Host from_device reconstruction diverges from exact GPU tile semantics"
    end
    return "No detectable mismatch above tolerance"
end

function analyze_case(
    pdb_path::AbstractString,
    ::Type{FT},
    lambda_value,
    solute_indices,
    scheduler_name::Symbol,
    device_id::Int;
    lj_cutoff_model::Symbol=DEFAULT_LJ_CUTOFF_MODEL,
    lj_activation_fraction::Float64=DEFAULT_LJ_ACTIVATION_FRACTION,
) where {FT <: AbstractFloat}
    sys_gpu = build_gpu_system(
        pdb_path,
        FT,
        lambda_value,
        solute_indices,
        scheduler_name,
        device_id;
        lj_cutoff_model=lj_cutoff_model,
        lj_activation_fraction=lj_activation_fraction,
    )
    buffers = Molly.init_buffers!(sys_gpu, 1, true)

    native_total = Molly.potential_energy(sys_gpu, nothing, buffers, 0; n_threads=1)
    CUDA.synchronize()
    snapshot = snapshot_tile_state(buffers)
    stats = tile_stats(snapshot)

    pairwise_full = pairwise_interactions(sys_gpu)
    gpu_pairwise_full = gpu_pairwise_energy(sys_gpu, buffers, pairwise_full, 0)
    dist_cutoff_sq = sys_gpu.neighbor_finder.dist_cutoff_2
    replay_pairwise_full = replay_pairwise_from_gpu_tiles(
        snapshot,
        sys_gpu.boundary,
        pairwise_full,
        sys_gpu.energy_units,
        dist_cutoff_sq,
        0,
        FT,
    )

    sys_host = Molly.from_device(sys_gpu)
    neighbors_host = Molly.find_neighbors(sys_host)
    host_pairwise_full = host_pairwise_energy(sys_host, neighbors_host, pairwise_full)
    host_nonpairwise = host_nonpairwise_energy(sys_host, neighbors_host)
    host_total = host_pairwise_full + host_nonpairwise
    replay_total = replay_pairwise_full + host_nonpairwise
    gpu_nonpairwise = native_total - gpu_pairwise_full

    component_results = NamedTuple[]
    for row in pairwise_component_rows(sys_gpu)
        gpu_component = gpu_pairwise_energy(sys_gpu, buffers, row.inters, 0)
        replay_component = replay_pairwise_from_gpu_tiles(
            snapshot,
            sys_gpu.boundary,
            row.inters,
            sys_gpu.energy_units,
            dist_cutoff_sq,
            0,
            FT,
        )
        host_component = host_pairwise_energy(sys_host, neighbors_host, row.inters)
        push!(
            component_results,
            (
                label=row.label,
                gpu=gpu_component,
                replay=replay_component,
                host=host_component,
            ),
        )
    end

    tol_kj_mol = FT == Float32 ? 1e-3 : 1e-8

    return (
        FT=FT,
        lambda_value=lambda_value,
        lj_cutoff_model=lj_cutoff_model,
        lj_activation_fraction=lj_activation_fraction,
        native_total=native_total,
        replay_total=replay_total,
        host_total=host_total,
        gpu_pairwise_full=gpu_pairwise_full,
        replay_pairwise_full=replay_pairwise_full,
        host_pairwise_full=host_pairwise_full,
        gpu_nonpairwise=gpu_nonpairwise,
        host_nonpairwise=host_nonpairwise,
        tile_stats=stats,
        component_results=component_results,
        conclusion=conclusion_line(gpu_pairwise_full, replay_pairwise_full, host_pairwise_full, tol_kj_mol),
        tol_kj_mol=tol_kj_mol,
    )
end

function print_case(io::IO, result)
    title = "FT=$(result.FT) lambda=$(result.lambda_value) lj_cutoff=$(result.lj_cutoff_model)"
    println(io)
    println(io, title)
    println(io, repeat("=", length(title)))
    if result.lj_cutoff_model == :cubic_spline
        println(io, "LJ activation fraction = ", format_real(result.lj_activation_fraction))
    end
    println(io, "Tiles")
    println(io, "  clean = ", result.tile_stats.clean)
    println(io, "  masked = ", result.tile_stats.masked)
    println(io, "  overflow = ", result.tile_stats.overflow)
    println(io)
    println(io, "Pairwise")
    println(io, "  gpu_native = ", format_real(ustrip(u"kJ/mol", result.gpu_pairwise_full)))
    println(io, "  cpu_replay_gpu_tiles = ", format_real(ustrip(u"kJ/mol", result.replay_pairwise_full)))
    println(io, "  host_from_device = ", format_real(ustrip(u"kJ/mol", result.host_pairwise_full)))
    print_delta(io, "gpu_minus_replay", result.gpu_pairwise_full, result.replay_pairwise_full)
    print_delta(io, "replay_minus_host", result.replay_pairwise_full, result.host_pairwise_full)
    println(io)
    println(io, "Nonpairwise")
    println(io, "  gpu_nonpairwise = ", format_real(ustrip(u"kJ/mol", result.gpu_nonpairwise)))
    println(io, "  host_nonpairwise = ", format_real(ustrip(u"kJ/mol", result.host_nonpairwise)))
    print_delta(io, "gpu_nonpairwise_minus_host", result.gpu_nonpairwise, result.host_nonpairwise)
    println(io)
    println(io, "Total")
    println(io, "  gpu_native = ", format_real(ustrip(u"kJ/mol", result.native_total)))
    println(io, "  cpu_replay_gpu_tiles_plus_host_nonpairwise = ", format_real(ustrip(u"kJ/mol", result.replay_total)))
    println(io, "  host_from_device = ", format_real(ustrip(u"kJ/mol", result.host_total)))
    print_delta(io, "gpu_minus_replay", result.native_total, result.replay_total)
    print_delta(io, "replay_minus_host", result.replay_total, result.host_total)
    println(io)
    println(io, "Components")
    for row in result.component_results
        print_component_block(io, row.label, row.gpu, row.replay, row.host)
    end
    println(io)
    println(io, "Tolerance = ", format_real(result.tol_kj_mol), " kJ/mol")
    println(io, "Conclusion = ", result.conclusion)
end

function main()
    opts = parse_cli(ARGS)
    for FT in opts[:float_types]
        for lambda_value in opts[:lambda_values]
            result = analyze_case(
                opts[:pdb],
                FT,
                lambda_value,
                opts[:solute_indices],
                opts[:scheduler],
                opts[:device_id],
                lj_cutoff_model=opts[:lj_cutoff_model],
                lj_activation_fraction=opts[:lj_activation_fraction],
            )
            print_case(stdout, result)
        end
    end
end

main()
