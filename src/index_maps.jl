"""
    build_index_maps(sys, param_indices_by_key, parameter_pools)

Build the index tuples consumed by Molly's `inject_atom`/`inject_interaction`
machinery. Optimized LJ parameters are keyed by `(pool_name, atom_type)` so
different parameter pools can carry independent copies of the same atom type.
"""
function build_index_maps(
    sys,
    param_indices_by_key::Dict{Tuple{Symbol, String}, Tuple{Int, Int}},
    parameter_pools::Vector{ParameterPoolConfig},
)
    atom_pool_names = resolve_system_parameter_pool_names(sys, parameter_pools)
    n_atoms = length(sys.atoms)
    idx_mass   = zeros(Int, n_atoms)
    idx_σ      = zeros(Int, n_atoms)
    idx_ϵ      = zeros(Int, n_atoms)

    for i in 1:n_atoms
        pool_name = atom_pool_names[i]
        isnothing(pool_name) && continue

        atype = String(sys.atoms_data[i].atom_type)
        key = (pool_name, atype)
        haskey(param_indices_by_key, key) || throw(ArgumentError("Parameter pool `$(pool_name)` matched atom type `$(atype)` in the current leg, but that pool/type pair was not present in the parameter reference leg."))
        sigma_idx, epsilon_idx = param_indices_by_key[key]
        idx_σ[i] = sigma_idx
        idx_ϵ[i] = epsilon_idx
    end

    return (
        (idx_mass, idx_σ, idx_ϵ),
        Tuple(() for _ in 1:length(sys.pairwise_inters)),
        Tuple(() for _ in 1:length(sys.specific_inter_lists)),
        Tuple(() for _ in 1:length(sys.general_inters))
    )
end
