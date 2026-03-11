using Enzyme
using Molly

Enzyme.API.looseTypeAnalysis!(true)

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

# ==============================================================================
# 2. CORE EVALUATION FUNCTIONS
# ==============================================================================

function evaluate_frame_energy(params::Vector{FT}, sys_ref::System{D, AT, FT}, 
                               coords_nounits, box_nounits, neighbors, 
                               atom_idxs, pairwise_idxs, specific_idxs, general_idxs) where {D, AT, FT}
                               
    idx_mass, idx_σ, idx_ϵ = atom_idxs
    
    # 1. Functional atom construction via broadcasting
    new_atoms = Molly.inject_atom.(sys_ref.atoms, Ref(params), idx_mass, idx_σ, idx_ϵ)
    
    # 2. Functional interaction updates
    new_pairwise = _update_pairwise(sys_ref.pairwise_inters, params, pairwise_idxs)
    new_specific = _update_specific(sys_ref.specific_inter_lists, params, specific_idxs)
    new_general  = _update_general(sys_ref.general_inters, params, general_idxs)
    
    # 3. Build Final System (Stack-allocated, 0 primary heap allocations inside System)
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
# 3. FREE ENERGY ENDPOINT EVALUATION
# ==============================================================================

function compute_weights_and_ess(
    energies_current::Matrix{FT}, 
    energies_ref::Matrix{FT}, 
    active_lambda_idx::Vector{Int}, 
    beta::FT,
    volumes::Vector{FT} = Float32[],
    P0::FT = zero(FT)
) where {FT <: AbstractFloat}
    
    M = length(active_lambda_idx)
    if M == 0
        throw(ArgumentError("compute_weights_and_ess received zero frames (`active_lambda_idx` is empty)."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("compute_weights_and_ess expected `volumes` length $M, got $(length(volumes))."))
    end
    log_W = zeros(FT, M)
    
    for k in 1:M
        l_k = active_lambda_idx[k]
        
        # Include PV term if volumes and pressure are provided
        pv_term = isempty(volumes) ? zero(FT) : beta * P0 * volumes[k]
        
        u_k_n = beta * energies_current[k, l_k] + pv_term
        u_k_0 = beta * energies_ref[k, l_k] + pv_term
        
        log_W[k] = -(u_k_n - u_k_0)
    end
    
    max_log_W = maximum(log_W)
    W_unnorm = exp.(log_W .- max_log_W)
    
    Z_W = sum(W_unnorm)
    w_norm = W_unnorm ./ Z_W
    
    ess = FT(1.0) / sum(w_norm .^ 2)
    return w_norm, ess
end

function compute_empirical_gradients_and_fim(
    param_names::Vector{String},
    gradients_dict::Dict{String, Matrix{FT}}, 
    w_norm::Vector{FT}, 
    active_lambda_idx::Vector{Int}, 
    beta::FT
) where {FT <: AbstractFloat}

    P = length(param_names)
    M = length(w_norm)
    
    S = zeros(FT, P, M)
    for (i, p_key) in enumerate(param_names)
        grad_matrix = gradients_dict[p_key]
        for k in 1:M
            l_k = active_lambda_idx[k]
            S[i, k] = beta * grad_matrix[k, l_k]
        end
    end
    
    s_mean = zeros(FT, P)
    for i in 1:P
        for k in 1:M
            s_mean[i] += w_norm[k] * S[i, k]
        end
    end
    
    fim = zeros(FT, P, P)
    for i in 1:P
        for j in 1:P
            cov_ij = zero(FT)
            for k in 1:M
                cov_ij += w_norm[k] * S[i, k] * S[j, k]
            end
            fim[i, j] = cov_ij - (s_mean[i] * s_mean[j])
        end
    end
    
    return s_mean, fim
end

function compute_global_endpoint_gradients(
    param_names::Vector{String},
    gradients_dict::Dict{String, Matrix{FT}},
    energies_current::Matrix{FT},
    energies_ref::Matrix{FT},
    active_lambda_idx::Vector{Int},
    lambda_target_idx::Int,
    beta::FT,
    awh_bias::Vector{FT},
    volumes::Vector{FT} = FT[],
    P0_energy_per_vol::FT = zero(FT);
    compute_gradients::Bool=true
) where {FT <: AbstractFloat}


    P = length(param_names)
    M = length(active_lambda_idx)
    num_lambda = length(awh_bias)

    # Preallocate to avoid GC thrashing in the inner loop
    log_denoms = zeros(FT, num_lambda)
    log_W_target = zeros(FT, M)
    
    # 1. Global MBAR Weights to Target Lambda
    for k in 1:M
        pv_term = isempty(volumes) ? zero(FT) : beta * P0_energy_per_vol * volumes[k]
        u_k_target = beta * energies_current[k, lambda_target_idx] + pv_term
        
        # Mixture denominator (log-sum-exp)
        for j in 1:num_lambda
            u_k_j_ref = beta * energies_ref[k, j] + pv_term
            log_denoms[j] = awh_bias[j] - u_k_j_ref
        end
        max_log_denom = maximum(log_denoms)
        denom_term = max_log_denom + log(sum(exp.(log_denoms .- max_log_denom)))
        
        log_W_target[k] = -u_k_target - denom_term
    end

    max_log_W = maximum(log_W_target)
    W_unnorm = exp.(log_W_target .- max_log_W)
    w_lambda_global = W_unnorm ./ sum(W_unnorm)

    # 2. Globally Weighted Gradient Expectation (Thermodynamic Gradient)
    grad_F_lambda = zeros(FT, length(param_names))
    if compute_gradients
        for (i, p_key) in enumerate(param_names)
            grad_matrix = gradients_dict[p_key]
            for k in 1:M
                grad_F_lambda[i] += w_lambda_global[k] * (beta * grad_matrix[k, lambda_target_idx])
            end
        end
    end

    # 3. True Thermodynamic Free Energy (Dimensionless MBAR)
    F_mbar = -(max_log_W + log(sum(W_unnorm)))

    return grad_F_lambda, F_mbar
end

function compute_full_mbar_profile(
    energies_current::Matrix{FT},
    energies_ref::Matrix{FT},
    awh_bias::Vector{FT},
    beta::FT;
    volumes::Vector{FT}=FT[],
    P0_energy_per_vol::FT=zero(FT)
) where {FT <: AbstractFloat}
    M, num_lambda = size(energies_current)
    if size(energies_ref) != (M, num_lambda)
        throw(ArgumentError("compute_full_mbar_profile expected energies_ref size $(M), $(num_lambda), got $(size(energies_ref))."))
    end
    if length(awh_bias) != num_lambda
        throw(ArgumentError("compute_full_mbar_profile expected awh_bias length $num_lambda, got $(length(awh_bias))."))
    end
    if !isempty(volumes) && length(volumes) != M
        throw(ArgumentError("compute_full_mbar_profile expected `volumes` length $M, got $(length(volumes))."))
    end
    if M == 0 || num_lambda == 0
        throw(ArgumentError("compute_full_mbar_profile received empty energies matrix."))
    end

    pv_terms = zeros(FT, M)
    if !isempty(volumes)
        @inbounds for k in 1:M
            pv_terms[k] = beta * P0_energy_per_vol * volumes[k]
        end
    end

    # Shared denominator term for each frame, reused across all λ targets.
    log_mixture_denom = zeros(FT, M)
    scratch = zeros(FT, num_lambda)
    @inbounds for k in 1:M
        pv_k = pv_terms[k]
        for j in 1:num_lambda
            scratch[j] = awh_bias[j] - (beta * energies_ref[k, j] + pv_k)
        end
        max_log = maximum(scratch)
        log_mixture_denom[k] = max_log + log(sum(exp.(scratch .- max_log)))
    end

    F_profile = zeros(FT, num_lambda)
    log_W_target = zeros(FT, M)
    @inbounds for λ in 1:num_lambda
        for k in 1:M
            pv_k = pv_terms[k]
            log_W_target[k] = -(beta * energies_current[k, λ] + pv_k) - log_mixture_denom[k]
        end
        max_log_W = maximum(log_W_target)
        F_profile[λ] = -(max_log_W + log(sum(exp.(log_W_target .- max_log_W))))
    end

    return F_profile
end

function compute_parity_gap(F_mbar::Vector{FT}, F_awh::Vector{FT}; ref_idx::Int=1) where {FT <: AbstractFloat}
    if length(F_mbar) != length(F_awh)
        throw(ArgumentError("compute_parity_gap expected vectors with equal length, got $(length(F_mbar)) and $(length(F_awh))."))
    end
    if isempty(F_mbar)
        throw(ArgumentError("compute_parity_gap received empty free-energy profiles."))
    end
    if ref_idx < 1 || ref_idx > length(F_mbar)
        throw(ArgumentError("compute_parity_gap got ref_idx=$ref_idx outside valid range 1:$(length(F_mbar))."))
    end

    F_mbar_aligned = F_mbar .- F_mbar[ref_idx]
    F_awh_aligned = F_awh .- F_awh[ref_idx]
    return maximum(abs.(F_mbar_aligned .- F_awh_aligned))
end
