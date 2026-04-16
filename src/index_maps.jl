"""
    build_index_maps(sys, lj_param_indices_by_key, charge_param_indices_by_key,
                     parameter_pools, charge_cfg)

Build the index tuples consumed by Molly's `inject_atom`/`inject_interaction`
machinery. Optimized LJ parameters are keyed by `(pool_name, atom_type)` so
different parameter pools can carry independent copies of the same atom type.
Charge-equilibration parameters are keyed by `(pool_name, charge_typing_key)`.
"""
function build_index_maps(
    sys,
    lj_param_indices_by_key::Dict{Tuple{Symbol, String}, <:NamedTuple},
    charge_param_indices_by_key::Dict{Tuple{Symbol, String}, <:NamedTuple},
    parameter_pools::Vector{ParameterPoolConfig},
    charge_cfg::ChargeTrainingConfig,
)
    atom_pool_names = resolve_system_parameter_pool_names(sys, parameter_pools)
    pool_cfgs_by_name = Dict(pool.name => pool for pool in parameter_pools)
    n_atoms = length(sys.atoms)
    atoms_cpu = Molly.from_device(sys.atoms)
    idx_mass   = zeros(Int, n_atoms)
    idx_charge_chi = zeros(Int, n_atoms)
    idx_charge_eta = zeros(Int, n_atoms)
    idx_σ      = zeros(Int, n_atoms)
    idx_ϵ      = zeros(Int, n_atoms)
    molecule_ids = system_molecule_ids(sys)
    molecule_charge_targets = system_reference_molecule_net_charges(sys, Float64)
    reference_charges = Float64.(getproperty.(atoms_cpu, :charge))

    for i in 1:n_atoms
        pool_name = atom_pool_names[i]
        isnothing(pool_name) && continue

        pool_cfg = pool_cfgs_by_name[pool_name]
        atype = String(sys.atoms_data[i].atom_type)
        lj_key = (pool_name, atype)
        if pool_trains_lj(pool_cfg)
            haskey(lj_param_indices_by_key, lj_key) || throw(ArgumentError("Parameter pool `$(pool_name)` matched atom type `$(atype)` in the current leg, but that pool/type pair was not present in the parameter reference leg."))
            idx_σ[i] = lj_param_indices_by_key[lj_key].sigma
            idx_ϵ[i] = lj_param_indices_by_key[lj_key].epsilon
        end

        charge_key = (pool_name, resolve_charge_typing_key(sys.atoms_data[i], charge_cfg))
        if pool_trains_charge(pool_cfg)
            haskey(charge_param_indices_by_key, charge_key) || throw(ArgumentError("Parameter pool `$(pool_name)` matched charge type `$(charge_key[2])` in the current leg, but that pool/type pair was not present in the parameter reference leg."))
            idx_charge_chi[i] = charge_param_indices_by_key[charge_key].chi
            idx_charge_eta[i] = charge_param_indices_by_key[charge_key].eta
        end
    end

    return (
        (
            mass=idx_mass,
            charge_chi=idx_charge_chi,
            charge_eta=idx_charge_eta,
            sigma=idx_σ,
            epsilon=idx_ϵ,
            molecule_ids=molecule_ids,
            molecule_charge_targets=molecule_charge_targets,
            reference_charges=reference_charges,
        ),
        Tuple(() for _ in 1:length(sys.pairwise_inters)),
        Tuple(() for _ in 1:length(sys.specific_inter_lists)),
        Tuple(() for _ in 1:length(sys.general_inters))
    )
end
