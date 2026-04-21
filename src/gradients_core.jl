using Enzyme
using Molly

Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.strictAliasing!(false)

# ==============================================================================
# 1. PURELY FUNCTIONAL HELPERS (ENZYME-STABLE)
# ==============================================================================

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
        # Use broadcasting to create a new array for this interaction list
        new_inters = Molly.inject_interaction.(list.inters, Ref(params), idx...)
        new_list = _reconstruct_list(list, new_inters)
        return (new_list, _update_specific_recursive(Base.tail(lists), params, Base.tail(idxs))...)
    end
end

# Interaction List reconstructors
_reconstruct_list(l::InteractionList1Atoms, i) = InteractionList1Atoms(l.is, i, l.types)
_reconstruct_list(l::InteractionList2Atoms, i) = InteractionList2Atoms(l.is, l.js, i, l.types)
_reconstruct_list(l::InteractionList3Atoms, i) = InteractionList3Atoms(l.is, l.js, l.ks, i, l.types)
_reconstruct_list(l::InteractionList4Atoms, i) = InteractionList4Atoms(l.is, l.js, l.ks, l.ls, i, l.types)

@inline function _enzyme_gradient_system(sys_ref::System{D, AT, FT}) where {D, AT, FT}
    if Molly.nonbonded_energy_type(sys_ref) === FT
        return sys_ref
    end
    return System(
        sys_ref;
        nonbonded_energy_type=FT,
        launch_config=Molly.CUDALaunchConfig(),
    )
end

@inline function _rebuild_system_like(
    sys_ref::System{D, AT, FT},
    new_atoms,
    coords_nounits,
    box_nounits,
    new_pairwise,
    new_specific,
    new_general,
) where {D, AT, FT}
    new_masses = convert(typeof(sys_ref.masses), mass.(new_atoms))
    new_total_mass = convert(typeof(sys_ref.total_mass), sum(new_masses))

    return typeof(sys_ref)(
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
        new_masses,
        new_total_mass,
        sys_ref.data,
        sys_ref.nonbonded_energy_type,
        Molly.CUDALaunchConfig(),
    )
end

# ==============================================================================
# 2. CORE EVALUATION FUNCTIONS
# ==============================================================================

"""
    evaluate_frame_energy(params, sys_ref, coords_nounits, box_nounits, neighbors,
                          atom_idxs, pairwise_idxs, specific_idxs, general_idxs)

Rebuild a single λ-state system with a candidate parameter vector and return its
potential energy for one stored frame.
"""
function evaluate_frame_energy(params::Vector{FT}, sys_ref::System{D, AT, FT}, 
                               coords_nounits, box_nounits, neighbors, 
                               atom_idxs, pairwise_idxs, specific_idxs, general_idxs) where {D, AT, FT}

    # 1. Functional atom construction with optional constrained charge updates.
    new_atoms = inject_atom_parameters(sys_ref.atoms, params, atom_idxs)
    
    # 2. Functional interaction updates
    new_pairwise = _update_pairwise(sys_ref.pairwise_inters, params, pairwise_idxs)
    new_specific = _update_specific(sys_ref.specific_inter_lists, params, specific_idxs)
    new_general  = _update_general(sys_ref.general_inters, params, general_idxs)
    
    # 3. Build Final System while mirroring Molly's stored field layout in one place.
    sys_final = _rebuild_system_like(
        sys_ref,
        new_atoms,
        coords_nounits,
        box_nounits,
        new_pairwise,
        new_specific,
        new_general,
    )

    return potential_energy(sys_final, neighbors; n_threads=1)
end

"""
    evaluate_frame_gradients(sys_ref, coords_nounits, box_nounits, neighbors,
                             params, grads_enzyme, atom_idxs, pairwise_idxs,
                             specific_idxs, general_idxs)

Differentiate `evaluate_frame_energy` with respect to the parameter vector using
Enzyme and return the primal energy.
"""
function evaluate_frame_gradients(sys_ref::System{D, AT, FT}, 
                                  coords_nounits, box_nounits, neighbors, 
                                  params::Vector{FT}, grads_enzyme::Vector{FT}, 
                                  atom_idxs, pairwise_idxs, specific_idxs, general_idxs) where {D, AT, FT}
    sys_grad = _enzyme_gradient_system(sys_ref)
    return with_compiler_safe_logger() do
        fill!(grads_enzyme, zero(FT))

        result = autodiff(
            set_runtime_activity(ReverseWithPrimal), 
            evaluate_frame_energy, 
            Active, 
            Duplicated(params, grads_enzyme), 
            Const(sys_grad),
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
end

# ==============================================================================
# 3. FREE ENERGY ENDPOINT EVALUATION
# ==============================================================================

"""
    awh_log_gibbs_weights(bias_data)

Return Molly's fixed AWH λ-state Gibbs log-weights

`g_ref(λ) = f_ref(λ) + logρ_ref(λ)`

used in `process_sample` through `scratch_z = log_rho + f - potentials`. This
is the correct quantity for MBAR/AWH mixture denominators. It is not a
coordinate-space bias potential and is never differentiated as a force term.
"""
function awh_log_gibbs_weights(bias_data)
    return bias_data.f .+ bias_data.log_rho
end

function awh_log_gibbs_weights(bias_data, ::Type{FT}) where {FT <: AbstractFloat}
    return FT.(awh_log_gibbs_weights(bias_data))
end

@inline _analysis_float_type(x::AbstractArray{T}) where {T <: AbstractFloat} = T
@inline _analysis_float_type(::Type{T}) where {T <: AbstractFloat} = T
@inline _analysis_float_type(x::T) where {T <: AbstractFloat} = T

function promote_energy_analysis_type(args...)
    Ts = map(_analysis_float_type, args)
    return foldl(promote_type, Ts)
end

"""
    reference_log_mixture_denominator(energies_ref, log_gibbs_weights, beta;
                                      volumes=FT[], P0_energy_per_vol=zero(FT))

For each stored frame `x_k`, compute

`log ∑_j exp(g_ref(j) - u_ref(x_k, j))`

where `g_ref = f_ref + logρ_ref` are the fixed AWH Gibbs log-weights and
`u_ref` is the reduced potential under the reference Hamiltonians that generated
the trajectory.
"""
function reference_log_mixture_denominator(
    energies_ref::AbstractMatrix{ET},
    log_gibbs_weights::AbstractVector{GT},
    beta::BT;
    volumes::AbstractVector{VT}=BT[],
    P0_energy_per_vol::PT=zero(BT),
) where {ET <: AbstractFloat, GT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}
    M, num_lambda = size(energies_ref)
    if length(log_gibbs_weights) != num_lambda
        throw(ArgumentError("reference_log_mixture_denominator expected log_gibbs_weights length $num_lambda, got $(length(log_gibbs_weights))."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("reference_log_mixture_denominator expected `volumes` length $M, got $(length(volumes))."))
    end
    if M == 0 || num_lambda == 0
        throw(ArgumentError("reference_log_mixture_denominator received empty energies matrix."))
    end

    AT = promote_energy_analysis_type(energies_ref, log_gibbs_weights, beta, volumes, P0_energy_per_vol)
    beta_AT = AT(beta)
    P0_AT = AT(P0_energy_per_vol)

    pv_terms = zeros(AT, M)
    if !isempty(volumes)
        @inbounds for k in 1:M
            pv_terms[k] = beta_AT * P0_AT * AT(volumes[k])
        end
    end

    log_mixture_denom = zeros(AT, M)
    scratch = zeros(AT, num_lambda)
    @inbounds for k in 1:M
        pv_k = pv_terms[k]
        for j in 1:num_lambda
            scratch[j] = AT(log_gibbs_weights[j]) - (beta_AT * AT(energies_ref[k, j]) + pv_k)
        end
        max_log = maximum(scratch)
        log_mixture_denom[k] = max_log + log(sum(exp.(scratch .- max_log)))
    end
    return log_mixture_denom
end

"""
    compute_weights_and_ess(energies_current, energies_ref, active_lambda_idx, beta,
                            volumes=FT[], P0=zero(FT))

Compute per-frame reweighting factors from the reference ensemble to the current
parameterization together with their effective sample size.

This reweights the full extended state `(x_k, λ_k)` under a fixed AWH Gibbs
scheme. Because the λ-state log-weight `g_ref(λ_k)` is the same in the numerator
and denominator, it cancels exactly; only the reduced-potential change at the
physically sampled `λ_k` remains.
"""
function compute_weights_and_ess(
    energies_current::AbstractMatrix{ETC}, 
    energies_ref::AbstractMatrix{ETR}, 
    active_lambda_idx::Vector{Int}, 
    beta::BT,
    volumes::AbstractVector{VT} = BT[],
    P0::PT = zero(BT)
) where {ETC <: AbstractFloat, ETR <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}

    M = length(active_lambda_idx)
    if size(energies_current, 1) != M
        throw(ArgumentError("compute_weights_and_ess expected energies_current to have $M rows, got $(size(energies_current, 1))."))
    end
    if size(energies_ref) != size(energies_current)
        throw(ArgumentError("compute_weights_and_ess expected energies_ref size $(size(energies_current)), got $(size(energies_ref))."))
    end
    if M == 0
        throw(ArgumentError("compute_weights_and_ess received zero frames (`active_lambda_idx` is empty)."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("compute_weights_and_ess expected `volumes` length $M, got $(length(volumes))."))
    end
    num_lambda = size(energies_current, 2)
    if any(idx -> idx < 1 || idx > num_lambda, active_lambda_idx)
        throw(ArgumentError("compute_weights_and_ess got active λ indices outside valid range 1:$num_lambda."))
    end
    AT = promote_energy_analysis_type(energies_current, energies_ref, beta, volumes, P0)
    beta_AT = AT(beta)
    P0_AT = AT(P0)
    log_W = zeros(AT, M)
    
    for k in 1:M
        l_k = active_lambda_idx[k]
        
        # Include PV term if volumes and pressure are provided
        pv_term = isempty(volumes) ? zero(AT) : beta_AT * P0_AT * AT(volumes[k])
        
        u_k_n = beta_AT * AT(energies_current[k, l_k]) + pv_term
        u_k_0 = beta_AT * AT(energies_ref[k, l_k]) + pv_term
        
        log_W[k] = -(u_k_n - u_k_0)
    end
    
    max_log_W = maximum(log_W)
    W_unnorm = exp.(log_W .- max_log_W)
    
    Z_W = sum(W_unnorm)
    w_norm = W_unnorm ./ Z_W
    
    ess = one(AT) / sum(w_norm .^ 2)
    return w_norm, ess
end

"""
    compute_empirical_gradients_and_fim(param_names, gradients_dict, w_norm,
                                        active_lambda_idx, beta)

Estimate the weighted score mean and Fisher information matrix from the
frame-by-frame energy gradients.
"""
function compute_empirical_gradients_and_fim(
    param_names::Vector{String},
    gradients_dict,
    w_norm::AbstractVector{WT}, 
    active_lambda_idx::Vector{Int}, 
    beta::BT
) where {WT <: AbstractFloat, BT <: AbstractFloat}

    P = length(param_names)
    M = length(w_norm)
    if length(active_lambda_idx) != M
        throw(ArgumentError("compute_empirical_gradients_and_fim expected active_lambda_idx length $M, got $(length(active_lambda_idx))."))
    end
    AT = promote_energy_analysis_type(w_norm, beta)
    beta_AT = AT(beta)
    
    S = zeros(AT, P, M)
    for (i, p_key) in enumerate(param_names)
        grad_matrix = gradients_dict[p_key]
        if size(grad_matrix, 1) != M
            throw(ArgumentError("compute_empirical_gradients_and_fim expected gradient matrix for $p_key to have $M rows, got $(size(grad_matrix, 1))."))
        end
        num_lambda = size(grad_matrix, 2)
        if any(idx -> idx < 1 || idx > num_lambda, active_lambda_idx)
            throw(ArgumentError("compute_empirical_gradients_and_fim got active λ indices outside valid range 1:$num_lambda for parameter $p_key."))
        end
        for k in 1:M
            l_k = active_lambda_idx[k]
            S[i, k] = beta_AT * AT(grad_matrix[k, l_k])
        end
    end
    
    s_mean = zeros(AT, P)
    for i in 1:P
        for k in 1:M
            s_mean[i] += w_norm[k] * S[i, k]
        end
    end
    
    fim = zeros(AT, P, P)
    for i in 1:P
        for j in 1:P
            cov_ij = zero(AT)
            for k in 1:M
                cov_ij += w_norm[k] * S[i, k] * S[j, k]
            end
            fim[i, j] = cov_ij - (s_mean[i] * s_mean[j])
        end
    end
    
    return s_mean, fim
end

"""
    compute_global_endpoint_gradients(param_names, gradients_dict, energies_current,
                                      energies_ref, active_lambda_idx,
                                      lambda_target_idx, beta, log_gibbs_weights,
                                      volumes=FT[], P0_energy_per_vol=zero(FT);
                                      compute_gradients=true)

Use global MBAR-style reweighting to estimate the free energy of one endpoint λ
state, optionally alongside its thermodynamic gradient. `log_gibbs_weights`
must be Molly's fixed AWH Gibbs log-weights `f_ref + logρ_ref`, not a
force-derived potential.
"""
function compute_global_endpoint_gradients(
    param_names::Vector{String},
    gradients_dict,
    energies_current::AbstractMatrix{ETC},
    energies_ref::AbstractMatrix{ETR},
    active_lambda_idx::Vector{Int},
    lambda_target_idx::Int,
    beta::BT,
    log_gibbs_weights::AbstractVector{GT},
    volumes::AbstractVector{VT} = BT[],
    P0_energy_per_vol::PT = zero(BT);
    compute_gradients::Bool=true
) where {ETC <: AbstractFloat, ETR <: AbstractFloat, BT <: AbstractFloat, GT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}


    P = length(param_names)
    M = length(active_lambda_idx)
    if size(energies_current, 1) != M
        throw(ArgumentError("compute_global_endpoint_gradients expected energies_current to have $M rows, got $(size(energies_current, 1))."))
    end
    if size(energies_ref) != size(energies_current)
        throw(ArgumentError("compute_global_endpoint_gradients expected energies_ref size $(size(energies_current)), got $(size(energies_ref))."))
    end
    _, num_lambda = size(energies_current)
    if lambda_target_idx < 1 || lambda_target_idx > num_lambda
        throw(ArgumentError("compute_global_endpoint_gradients got lambda_target_idx=$lambda_target_idx outside valid range 1:$num_lambda."))
    end
    if any(idx -> idx < 1 || idx > num_lambda, active_lambda_idx)
        throw(ArgumentError("compute_global_endpoint_gradients got active λ indices outside valid range 1:$num_lambda."))
    end
    AT = promote_energy_analysis_type(
        energies_current,
        energies_ref,
        log_gibbs_weights,
        beta,
        volumes,
        P0_energy_per_vol,
    )
    beta_AT = AT(beta)
    P0_AT = AT(P0_energy_per_vol)
    log_W_target = zeros(AT, M)
    log_mixture_denom = reference_log_mixture_denominator(
        energies_ref,
        log_gibbs_weights,
        beta;
        volumes=volumes,
        P0_energy_per_vol=P0_energy_per_vol,
    )
    
    # 1. Global MBAR weights to the requested target λ. The denominator uses
    # Molly's fixed AWH Gibbs log-weights g_ref = f_ref + logρ_ref.
    for k in 1:M
        pv_term = isempty(volumes) ? zero(AT) : beta_AT * P0_AT * AT(volumes[k])
        u_k_target = beta_AT * AT(energies_current[k, lambda_target_idx]) + pv_term
        log_W_target[k] = -u_k_target - log_mixture_denom[k]
    end

    max_log_W = maximum(log_W_target)
    W_unnorm = exp.(log_W_target .- max_log_W)
    w_lambda_global = W_unnorm ./ sum(W_unnorm)

    # 2. Globally weighted gradient expectation (thermodynamic gradient).
    grad_F_lambda = zeros(AT, length(param_names))
    if compute_gradients
        for (i, p_key) in enumerate(param_names)
            grad_matrix = gradients_dict[p_key]
            for k in 1:M
                grad_F_lambda[i] += w_lambda_global[k] * (beta_AT * AT(grad_matrix[k, lambda_target_idx]))
            end
        end
    end

    # 3. Dimensionless free energy of the target endpoint.
    F_mbar = -(max_log_W + log(sum(W_unnorm)))

    return grad_F_lambda, F_mbar
end

"""
    compute_full_mbar_profile(energies_current, energies_ref, log_gibbs_weights, beta;
                              volumes=FT[], P0_energy_per_vol=zero(FT))

Reconstruct the complete λ free-energy profile implied by the stored trajectory
under a candidate parameter vector. `log_gibbs_weights` must be the fixed AWH
Gibbs log-weights `f_ref + logρ_ref` used to sample λ.
"""
function compute_full_mbar_profile(
    energies_current::AbstractMatrix{ETC},
    energies_ref::AbstractMatrix{ETR},
    log_gibbs_weights::AbstractVector{GT},
    beta::BT;
    volumes::AbstractVector{VT}=BT[],
    P0_energy_per_vol::PT=zero(BT)
) where {ETC <: AbstractFloat, ETR <: AbstractFloat, GT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}
    M, num_lambda = size(energies_current)
    if size(energies_ref) != (M, num_lambda)
        throw(ArgumentError("compute_full_mbar_profile expected energies_ref size $(M), $(num_lambda), got $(size(energies_ref))."))
    end
    if length(log_gibbs_weights) != num_lambda
        throw(ArgumentError("compute_full_mbar_profile expected log_gibbs_weights length $num_lambda, got $(length(log_gibbs_weights))."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("compute_full_mbar_profile expected `volumes` length $M, got $(length(volumes))."))
    end
    if M == 0 || num_lambda == 0
        throw(ArgumentError("compute_full_mbar_profile received empty energies matrix."))
    end

    AT = promote_energy_analysis_type(
        energies_current,
        energies_ref,
        log_gibbs_weights,
        beta,
        volumes,
        P0_energy_per_vol,
    )
    beta_AT = AT(beta)
    P0_AT = AT(P0_energy_per_vol)

    pv_terms = zeros(AT, M)
    if !isempty(volumes)
        @inbounds for k in 1:M
            pv_terms[k] = beta_AT * P0_AT * AT(volumes[k])
        end
    end
    log_mixture_denom = reference_log_mixture_denominator(
        energies_ref,
        log_gibbs_weights,
        beta;
        volumes=volumes,
        P0_energy_per_vol=P0_energy_per_vol,
    )

    F_profile = zeros(AT, num_lambda)
    log_W_target = zeros(AT, M)
    @inbounds for λ in 1:num_lambda
        for k in 1:M
            pv_k = pv_terms[k]
            log_W_target[k] = -(beta_AT * AT(energies_current[k, λ]) + pv_k) - log_mixture_denom[k]
        end
        max_log_W = maximum(log_W_target)
        F_profile[λ] = -(max_log_W + log(sum(exp.(log_W_target .- max_log_W))))
    end

    return F_profile
end

"""
    compute_full_mbar_profile_from_log_mixture_denom(energies_current, log_mixture_denom, beta;
                                                     volumes=FT[], P0_energy_per_vol=zero(FT))

Reconstruct the complete λ free-energy profile when the per-frame reference
mixture denominator `log_mixture_denom` has already been precomputed. This is
used by Stage B accumulation where each probe segment can have a different
frozen AWH reference bias.
"""
function compute_full_mbar_profile_from_log_mixture_denom(
    energies_current::AbstractMatrix{ET},
    log_mixture_denom::AbstractVector{DT},
    beta::BT;
    volumes::AbstractVector{VT}=BT[],
    P0_energy_per_vol::PT=zero(BT)
) where {ET <: AbstractFloat, DT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}
    M, num_lambda = size(energies_current)
    if length(log_mixture_denom) != M
        throw(ArgumentError("compute_full_mbar_profile_from_log_mixture_denom expected log_mixture_denom length $M, got $(length(log_mixture_denom))."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("compute_full_mbar_profile_from_log_mixture_denom expected `volumes` length $M, got $(length(volumes))."))
    end
    if M == 0 || num_lambda == 0
        throw(ArgumentError("compute_full_mbar_profile_from_log_mixture_denom received empty energies matrix."))
    end

    AT = promote_energy_analysis_type(energies_current, log_mixture_denom, beta, volumes, P0_energy_per_vol)
    beta_AT = AT(beta)
    P0_AT = AT(P0_energy_per_vol)

    pv_terms = zeros(AT, M)
    if !isempty(volumes)
        @inbounds for k in 1:M
            pv_terms[k] = beta_AT * P0_AT * AT(volumes[k])
        end
    end

    F_profile = zeros(AT, num_lambda)
    log_W_target = zeros(AT, M)
    @inbounds for λ in 1:num_lambda
        for k in 1:M
            pv_k = pv_terms[k]
            log_W_target[k] = -(beta_AT * AT(energies_current[k, λ]) + pv_k) - AT(log_mixture_denom[k])
        end
        max_log_W = maximum(log_W_target)
        F_profile[λ] = -(max_log_W + log(sum(exp.(log_W_target .- max_log_W))))
    end

    return F_profile
end

"""
    compute_state_reweighting_ess_from_log_mixture_denom(energies_current, log_mixture_denom, beta;
                                                         volumes=FT[], P0_energy_per_vol=zero(FT))

Compute a per-state effective sample size (ESS) for the MBAR target weights
using precomputed reference mixture denominators.
"""
function compute_state_reweighting_ess_from_log_mixture_denom(
    energies_current::AbstractMatrix{ET},
    log_mixture_denom::AbstractVector{DT},
    beta::BT;
    volumes::AbstractVector{VT}=BT[],
    P0_energy_per_vol::PT=zero(BT)
) where {ET <: AbstractFloat, DT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}
    M, num_lambda = size(energies_current)
    if length(log_mixture_denom) != M
        throw(ArgumentError("compute_state_reweighting_ess_from_log_mixture_denom expected log_mixture_denom length $M, got $(length(log_mixture_denom))."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("compute_state_reweighting_ess_from_log_mixture_denom expected `volumes` length $M, got $(length(volumes))."))
    end
    if M == 0 || num_lambda == 0
        throw(ArgumentError("compute_state_reweighting_ess_from_log_mixture_denom received empty energies matrix."))
    end

    AT = promote_energy_analysis_type(energies_current, log_mixture_denom, beta, volumes, P0_energy_per_vol)
    beta_AT = AT(beta)
    P0_AT = AT(P0_energy_per_vol)

    pv_terms = zeros(AT, M)
    if !isempty(volumes)
        @inbounds for k in 1:M
            pv_terms[k] = beta_AT * P0_AT * AT(volumes[k])
        end
    end

    ess_by_state = zeros(AT, num_lambda)
    log_W_target = zeros(AT, M)
    @inbounds for λ in 1:num_lambda
        for k in 1:M
            pv_k = pv_terms[k]
            log_W_target[k] = -(beta_AT * AT(energies_current[k, λ]) + pv_k) - AT(log_mixture_denom[k])
        end
        max_log_W = maximum(log_W_target)
        w = exp.(log_W_target .- max_log_W)
        w ./= sum(w)
        ess_by_state[λ] = one(AT) / sum(w .^ 2)
    end

    return ess_by_state
end

"""
    compute_parity_gap(F_mbar, F_awh; ref_idx=1)

Measure the maximum deviation between an MBAR-reconstructed free-energy profile
and the AWH free-energy estimate after aligning them at `ref_idx`.
"""
function compute_parity_gap(
    F_mbar::AbstractVector{MT},
    F_awh::AbstractVector{ATW};
    ref_idx::Int=1,
) where {MT <: AbstractFloat, ATW <: AbstractFloat}
    if length(F_mbar) != length(F_awh)
        throw(ArgumentError("compute_parity_gap expected vectors with equal length, got $(length(F_mbar)) and $(length(F_awh))."))
    end
    if isempty(F_mbar)
        throw(ArgumentError("compute_parity_gap received empty free-energy profiles."))
    end
    if ref_idx < 1 || ref_idx > length(F_mbar)
        throw(ArgumentError("compute_parity_gap got ref_idx=$ref_idx outside valid range 1:$(length(F_mbar))."))
    end

    PT = promote_energy_analysis_type(F_mbar, F_awh)
    F_mbar_aligned = PT.(F_mbar) .- PT(F_mbar[ref_idx])
    F_awh_aligned = PT.(F_awh) .- PT(F_awh[ref_idx])
    return maximum(abs.(F_mbar_aligned .- F_awh_aligned))
end
