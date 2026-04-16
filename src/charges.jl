const LJ_PARAMETER_FAMILIES = (:sigma, :epsilon)
const CHARGE_PARAMETER_FAMILIES = (:charge_chi, :charge_eta)

function normalize_charge_training_config(cfg::ChargeTrainingConfig)
    cfg.hardness_floor > 0 || throw(ArgumentError("ChargeTrainingConfig.hardness_floor must be positive, got $(cfg.hardness_floor)."))
    cfg.reference_hardness > cfg.hardness_floor || throw(ArgumentError("ChargeTrainingConfig.reference_hardness must exceed hardness_floor."))
    cfg.typing_basis in (:atom_class, :atom_type) || throw(ArgumentError("ChargeTrainingConfig.typing_basis must be :atom_class or :atom_type, got `$(cfg.typing_basis)`."))
    cfg.net_charge_constraint == :molecule || throw(ArgumentError("ChargeTrainingConfig.net_charge_constraint must be :molecule, got `$(cfg.net_charge_constraint)`."))
    return cfg
end

pool_trains_charge(pool::ParameterPoolConfig) = any(family -> family in CHARGE_PARAMETER_FAMILIES, pool.trainable_families)
pool_trains_lj(pool::ParameterPoolConfig) = any(family -> family in LJ_PARAMETER_FAMILIES, pool.trainable_families)

function resolve_charge_typing_key(atom_data, charge_cfg::ChargeTrainingConfig)
    atom_type = String(atom_data.atom_type)
    if charge_cfg.typing_basis == :atom_class
        if hasproperty(ff, :class_of)
            return get(getproperty(ff, :class_of), atom_type, atom_type)
        elseif hasproperty(ff, :type_to_class)
            return get(getproperty(ff, :type_to_class), atom_type, atom_type)
        end
    end
    return atom_type
end

function system_molecule_ids(sys)
    n_atoms = length(sys.atoms)
    if isnothing(sys.topology)
        return ones(Int, n_atoms)
    end
    return Int.(sys.topology.atom_molecule_inds)
end

function system_reference_molecule_net_charges(sys, ::Type{FT}) where {FT <: AbstractFloat}
    molecule_ids = system_molecule_ids(sys)
    n_molecules = isempty(molecule_ids) ? 0 : maximum(molecule_ids)
    totals = zeros(FT, n_molecules)
    atoms_cpu = Molly.from_device(sys.atoms)
    for (atom_idx, molecule_id) in enumerate(molecule_ids)
        totals[molecule_id] += FT(atoms_cpu[atom_idx].charge)
    end
    return totals
end

function build_charge_reference_stats(
    sys,
    atom_pool_names,
    pool_cfgs_by_name::Dict{Symbol, ParameterPoolConfig},
    charge_cfg::ChargeTrainingConfig,
    ::Type{FT},
) where {FT <: AbstractFloat}
    atoms_cpu = Molly.from_device(sys.atoms)
    sums = Dict{Tuple{Symbol, String}, FT}()
    counts = Dict{Tuple{Symbol, String}, Int}()

    for atom_idx in eachindex(atoms_cpu)
        pool_name = atom_pool_names[atom_idx]
        isnothing(pool_name) && continue
        pool_cfg = pool_cfgs_by_name[pool_name]
        pool_trains_charge(pool_cfg) || continue

        charge_key = resolve_charge_typing_key(sys.atoms_data[atom_idx], charge_cfg)
        stat_key = (pool_name, charge_key)
        sums[stat_key] = get(sums, stat_key, zero(FT)) + FT(atoms_cpu[atom_idx].charge)
        counts[stat_key] = get(counts, stat_key, 0) + 1
    end

    means = Dict{Tuple{Symbol, String}, FT}()
    for (key, total) in sums
        means[key] = total / FT(counts[key])
    end
    return means
end

@inline _atom_has_field(::Type{T}, field::Symbol) where {T} = field in fieldnames(T)

@inline function rebuild_atom_like(
    at;
    mass=at.mass,
    charge=at.charge,
    sigma=at.σ,
    epsilon=at.ϵ,
    lambda_value=nothing,
    alch_role=nothing,
)
    if _atom_has_field(typeof(at), :λ) && _atom_has_field(typeof(at), :alch_role)
        λ_local = isnothing(lambda_value) ? getfield(at, :λ) : lambda_value
        role_local = isnothing(alch_role) ? getfield(at, :alch_role) : alch_role
        return Molly.Atom(at.index, at.atom_type, mass, charge, sigma, epsilon, λ_local, role_local)
    elseif _atom_has_field(typeof(at), :λ)
        λ_local = isnothing(lambda_value) ? getfield(at, :λ) : lambda_value
        return Molly.Atom(at.index, at.atom_type, mass, charge, sigma, epsilon, λ_local)
    end
    return Molly.Atom(at.index, at.atom_type, mass, charge, sigma, epsilon)
end

function solve_constrained_charges(
    params::AbstractVector{FT},
    atom_charge_chi::AbstractVector{Int},
    atom_charge_eta::AbstractVector{Int},
    molecule_ids::AbstractVector{Int},
    molecule_charge_targets::AbstractVector{FT},
    reference_charges::AbstractVector{FT},
) where {FT <: AbstractFloat}
    charges = copy(reference_charges)
    n_atoms = length(reference_charges)
    n_molecules = length(molecule_charge_targets)

    trainable_charge_counts = zeros(Int, n_molecules)
    fixed_charge_totals = zeros(FT, n_molecules)
    inv_eta_sums = zeros(FT, n_molecules)
    chi_over_eta_sums = zeros(FT, n_molecules)

    for atom_idx in 1:n_atoms
        molecule_id = molecule_ids[atom_idx]
        chi_idx = atom_charge_chi[atom_idx]
        eta_idx = atom_charge_eta[atom_idx]

        if chi_idx > 0 && eta_idx > 0
            eta_i = params[eta_idx]
            inv_eta = inv(eta_i)
            inv_eta_sums[molecule_id] += inv_eta
            chi_over_eta_sums[molecule_id] += params[chi_idx] * inv_eta
            trainable_charge_counts[molecule_id] += 1
        else
            fixed_charge_totals[molecule_id] += reference_charges[atom_idx]
        end
    end

    lagrange = zeros(FT, n_molecules)
    for molecule_id in 1:n_molecules
        if trainable_charge_counts[molecule_id] == 0
            continue
        end
        rhs = molecule_charge_targets[molecule_id] - fixed_charge_totals[molecule_id]
        lagrange[molecule_id] = -(rhs + chi_over_eta_sums[molecule_id]) / inv_eta_sums[molecule_id]
    end

    for atom_idx in 1:n_atoms
        chi_idx = atom_charge_chi[atom_idx]
        eta_idx = atom_charge_eta[atom_idx]
        if chi_idx > 0 && eta_idx > 0
            molecule_id = molecule_ids[atom_idx]
            charges[atom_idx] = -(params[chi_idx] + lagrange[molecule_id]) / params[eta_idx]
        end
    end

    return charges
end

function inject_atom_parameters(
    atoms,
    params::AbstractVector{FT},
    atom_idxs,
) where {FT <: AbstractFloat}
    idx_mass = atom_idxs.mass
    idx_charge_chi = atom_idxs.charge_chi
    idx_charge_eta = atom_idxs.charge_eta
    idx_sigma = atom_idxs.sigma
    idx_epsilon = atom_idxs.epsilon
    reference_charges = FT.(atom_idxs.reference_charges)
    molecule_charge_targets = FT.(atom_idxs.molecule_charge_targets)

    charges = if any(>(0), idx_charge_chi) || any(>(0), idx_charge_eta)
        solve_constrained_charges(
            params,
            idx_charge_chi,
            idx_charge_eta,
            atom_idxs.molecule_ids,
            molecule_charge_targets,
            reference_charges,
        )
    else
        reference_charges
    end

    new_atoms = Vector{eltype(atoms)}(undef, length(atoms))
    for atom_idx in eachindex(atoms)
        at = atoms[atom_idx]
        new_mass = idx_mass[atom_idx] > 0 ? params[idx_mass[atom_idx]] : at.mass
        new_sigma = idx_sigma[atom_idx] > 0 ? params[idx_sigma[atom_idx]] : at.σ
        new_epsilon = idx_epsilon[atom_idx] > 0 ? params[idx_epsilon[atom_idx]] : at.ϵ
        new_atoms[atom_idx] = rebuild_atom_like(
            at;
            mass=new_mass,
            charge=charges[atom_idx],
            sigma=new_sigma,
            epsilon=new_epsilon,
        )
    end

    return new_atoms
end
