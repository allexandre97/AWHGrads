function _glob_to_regex(pattern::AbstractString)
    escaped = replace(pattern, r"([.^$+(){}\[\]|\\])" => s"\\\1")
    escaped = replace(escaped, "*" => ".*", "?" => ".")
    return Regex("^" * escaped * "\$")
end

function _pool_selector_configured(pool::ParameterPoolConfig)
    return !isnothing(pool.atom_indices) ||
        !isempty(pool.atom_types) ||
        !isempty(pool.atom_type_patterns) ||
        !isempty(pool.atom_names) ||
        !isempty(pool.residue_names) ||
        !isempty(pool.residue_numbers) ||
        !isempty(pool.molecule_ids)
end

function normalize_parameter_pool_config(pool::ParameterPoolConfig)
    families = unique(Symbol.(pool.trainable_families))
    isempty(families) && throw(ArgumentError("Parameter pool `$(pool.name)` must train at least one parameter family."))
    invalid_families = setdiff(families, [:sigma, :epsilon, :charge_chi, :charge_eta])
    isempty(invalid_families) || throw(ArgumentError("Parameter pool `$(pool.name)` has unsupported trainable families: $(join(string.(invalid_families), ", "))."))
    any(f -> f in (:charge_chi, :charge_eta), families) &&
        !all(f -> f in families, (:charge_chi, :charge_eta)) &&
        throw(ArgumentError("Parameter pool `$(pool.name)` must include both :charge_chi and :charge_eta when enabling charge training."))
    _pool_selector_configured(pool) || throw(ArgumentError("Parameter pool `$(pool.name)` must define at least one selector."))
    pool.reference_penalty_strength >= 0 || throw(ArgumentError("Parameter pool `$(pool.name)` must have non-negative reference_penalty_strength."))
    !isnothing(pool.max_phi_step) && pool.max_phi_step <= 0 && throw(ArgumentError("Parameter pool `$(pool.name)` must have positive max_phi_step."))
    !isnothing(pool.max_sigma_drift) && pool.max_sigma_drift < 0 && throw(ArgumentError("Parameter pool `$(pool.name)` must have non-negative max_sigma_drift."))
    !isnothing(pool.max_epsilon_drift) && pool.max_epsilon_drift < 0 && throw(ArgumentError("Parameter pool `$(pool.name)` must have non-negative max_epsilon_drift."))

    return ParameterPoolConfig(
        name=pool.name,
        atom_indices=isnothing(pool.atom_indices) ? nothing : collect(Int, pool.atom_indices),
        atom_types=String.(pool.atom_types),
        atom_type_patterns=String.(pool.atom_type_patterns),
        atom_names=String.(pool.atom_names),
        residue_names=String.(pool.residue_names),
        residue_numbers=Int.(pool.residue_numbers),
        molecule_ids=Int.(pool.molecule_ids),
        trainable_families=families,
        exclude_atom_types=String.(pool.exclude_atom_types),
        max_phi_step=isnothing(pool.max_phi_step) ? nothing : Float64(pool.max_phi_step),
        max_sigma_drift=isnothing(pool.max_sigma_drift) ? nothing : Float64(pool.max_sigma_drift),
        max_epsilon_drift=isnothing(pool.max_epsilon_drift) ? nothing : Float64(pool.max_epsilon_drift),
        reference_penalty_strength=Float64(pool.reference_penalty_strength),
    )
end

function default_parameter_pool_max_phi_step(opt_cfg::OptimizationConfig, n_pools::Int)
    if n_pools <= 1
        return Float64(opt_cfg.max_phi_step_solute)
    end
    return Float64(opt_cfg.max_phi_step_solvent)
end

function legacy_parameter_pool_configs(
    sim_cfg::SimulationConfig,
    opt_cfg::OptimizationConfig,
    n_atoms::Int,
)
    primary_indices = unique(sort(collect(Int, sim_cfg.solute_idx)))
    any(idx -> idx < 1 || idx > n_atoms, primary_indices) && throw(ArgumentError("SimulationConfig.solute_idx contains atom indices outside valid range 1:$n_atoms."))

    pools = ParameterPoolConfig[
        ParameterPoolConfig(
            name=:primary,
            atom_indices=primary_indices,
            trainable_families=Symbol[:sigma, :epsilon],
            max_phi_step=Float64(opt_cfg.max_phi_step_solute),
        ),
    ]

    if opt_cfg.optimize_solvent
        environment_indices = setdiff(collect(1:n_atoms), primary_indices)
        if !isempty(environment_indices)
            push!(
                pools,
                ParameterPoolConfig(
                    name=:environment,
                    atom_indices=environment_indices,
                    trainable_families=Symbol[:sigma, :epsilon],
                    max_phi_step=Float64(opt_cfg.max_phi_step_solvent),
                ),
            )
        end
    end

    return pools
end

function _pool_matches_atom(
    sys::System,
    atom_idx::Int,
    pool::ParameterPoolConfig,
)
    atom_data = sys.atoms_data[atom_idx]
    atom_type = String(atom_data.atom_type)
    atom_type in pool.exclude_atom_types && return false

    checks = Bool[]
    if !isnothing(pool.atom_indices)
        push!(checks, atom_idx in pool.atom_indices)
    end
    !isempty(pool.atom_types) && push!(checks, atom_type in pool.atom_types)
    !isempty(pool.atom_type_patterns) && push!(checks, any(occursin(_glob_to_regex(pattern), atom_type) for pattern in pool.atom_type_patterns))
    !isempty(pool.atom_names) && push!(checks, String(atom_data.atom_name) in pool.atom_names)
    !isempty(pool.residue_names) && push!(checks, String(atom_data.res_name) in pool.residue_names)
    !isempty(pool.residue_numbers) && push!(checks, Int(atom_data.res_number) in pool.residue_numbers)
    if !isempty(pool.molecule_ids)
        isnothing(sys.topology) && throw(ArgumentError("Parameter pool `$(pool.name)` uses molecule_ids selectors but the system topology has no molecule assignments."))
        push!(checks, Int(sys.topology.atom_molecule_inds[atom_idx]) in pool.molecule_ids)
    end

    return !isempty(checks) && all(checks)
end

function resolve_system_parameter_pool_names(
    sys::System,
    pools::Vector{ParameterPoolConfig},
)
    memberships = Vector{Union{Nothing, Symbol}}(undef, length(sys.atoms))
    fill!(memberships, nothing)

    for atom_idx in eachindex(sys.atoms)
        matched_names = Symbol[]
        for pool in pools
            _pool_matches_atom(sys, atom_idx, pool) && push!(matched_names, pool.name)
        end
        if length(matched_names) > 1
            throw(ArgumentError("Atom $atom_idx matched multiple parameter pools: $(join(string.(matched_names), ", ")). Parameter pools must be disjoint."))
        end
        memberships[atom_idx] = isempty(matched_names) ? nothing : only(matched_names)
    end

    return memberships
end

function resolved_parameter_pool_configs(
    sim_cfg::SimulationConfig,
    opt_cfg::OptimizationConfig,
    sys_ref::System,
)
    raw_pools = isempty(sim_cfg.parameter_pools) ?
        legacy_parameter_pool_configs(sim_cfg, opt_cfg, length(sys_ref.atoms)) :
        sim_cfg.parameter_pools

    pools = [normalize_parameter_pool_config(pool) for pool in raw_pools]
    pool_names = [pool.name for pool in pools]
    length(unique(pool_names)) == length(pool_names) || throw(ArgumentError("Parameter pool names must be unique; got $(join(string.(pool_names), ", "))."))

    memberships = resolve_system_parameter_pool_names(sys_ref, pools)
    ref_label = try
        String(sys_ref.data[:source_file])
    catch
        "parameter reference leg"
    end
    for pool in pools
        any(isequal(pool.name), memberships) || throw(ArgumentError("Parameter pool `$(pool.name)` did not match any atoms in `$ref_label`."))
    end

    return pools
end
