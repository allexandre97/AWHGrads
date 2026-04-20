"""
    leg_volumes(leg, FT)

Return the production-frame volumes for a leg when a `pV` correction is part of
that leg's free-energy definition.
"""
function leg_volumes(leg::LegArtifacts, ::Type{FT}) where {FT <: AbstractFloat}
    if !leg.include_pv
        return FT[]
    end
    if !isnothing(leg.eval_cache) && !isnothing(leg.eval_cache.frame_cache)
        return FT.(leg.eval_cache.frame_cache.volumes)
    end
    return FT.(ustrip.(leg.logger_prod.volume_history))
end

function evaluate_leg_ensemble(
    leg::LegArtifacts,
    params::Vector{FT},
    param_names::Vector{String};
    compute_gradients::Bool=true,
) where {FT <: AbstractFloat}
    if !isnothing(leg.eval_cache)
        return evaluate_ensemble(
            leg.eval_cache,
            params,
            param_names,
            leg.idxs...;
            compute_gradients=compute_gradients,
        )
    end

    return evaluate_ensemble(
        leg.logger_prod,
        leg.neighbors,
        leg.awh_prod,
        leg.sys_base,
        params,
        param_names,
        leg.idxs...;
        compute_gradients=compute_gradients,
    )
end


"""
    compute_leg_weights_and_ess(leg, energies_current, beta_val, volumes)

Wrapper around `compute_weights_and_ess` that injects the leg-specific
reference energies and optional `pV` term.
"""
function compute_leg_weights_and_ess(
    leg::LegArtifacts,
    energies_current::AbstractMatrix{ET},
    beta_val::BT,
    volumes::AbstractVector{VT};
    frame_indices::Union{Nothing, AbstractVector{Int}}=nothing,
) where {ET <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat}
    idx_history = isnothing(frame_indices) ? leg.logger_prod.active_idx_history : leg.logger_prod.active_idx_history[frame_indices]
    energies_local = isnothing(frame_indices) ? energies_current : @view energies_current[frame_indices, :]
    u_ref_local = isnothing(frame_indices) ? leg.u_ref : @view leg.u_ref[frame_indices, :]
    volumes_local = leg.include_pv && !isnothing(frame_indices) ? volumes[frame_indices] : volumes
    if leg.include_pv
        return compute_weights_and_ess(
            energies_local,
            u_ref_local,
            idx_history,
            beta_val,
            volumes_local,
            leg.p0_energy_per_vol,
        )
    end
    return compute_weights_and_ess(
        energies_local,
        u_ref_local,
        idx_history,
        beta_val,
    )
end


"""
    compute_leg_endpoint_state(leg, trainable_param_names, gradients_phi,
                               energies_current, beta_val, volumes;
                               compute_gradients=true)

Estimate `ΔG_leg = F(λ=1) - F(λ=0)` and optionally its gradient with respect to
the trainable `ϕ` parameters.
"""
function compute_leg_endpoint_state(
    leg::LegArtifacts,
    trainable_param_names::Vector{String},
    gradients_phi,
    energies_current::AbstractMatrix{ET},
    beta_val::BT,
    volumes::AbstractVector{VT};
    compute_gradients::Bool=true,
    frame_indices::Union{Nothing, AbstractVector{Int}}=nothing,
) where {ET <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat}
    idx_history = isnothing(frame_indices) ? leg.logger_prod.active_idx_history : leg.logger_prod.active_idx_history[frame_indices]
    energies_local = isnothing(frame_indices) ? energies_current : @view energies_current[frame_indices, :]
    u_ref_local = isnothing(frame_indices) ? leg.u_ref : @view leg.u_ref[frame_indices, :]
    gradients_phi_local = isnothing(frame_indices) ? gradients_phi : Dict(
        p_key => @view(gradients_phi[p_key][frame_indices, :]) for p_key in trainable_param_names
    )
    volumes_local = leg.include_pv && !isnothing(frame_indices) ? volumes[frame_indices] : volumes
    # Endpoint MBAR conditions on arbitrary λ targets, so the denominator must
    # use Molly's fixed AWH Gibbs log-weights g_ref = f_ref + logρ_ref.
    log_gibbs_weights = awh_log_gibbs_weights(leg.active_bias)

    if leg.include_pv
        grad_F_1, F_1 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi_local,
            energies_local,
            u_ref_local,
            idx_history,
            leg.coupled_state_idx,
            beta_val,
            log_gibbs_weights,
            volumes_local,
            leg.p0_energy_per_vol;
            compute_gradients=compute_gradients,
        )
        grad_F_0, F_0 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi_local,
            energies_local,
            u_ref_local,
            idx_history,
            leg.decoupled_state_idx,
            beta_val,
            log_gibbs_weights,
            volumes_local,
            leg.p0_energy_per_vol;
            compute_gradients=compute_gradients,
        )
    else
        grad_F_1, F_1 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi_local,
            energies_local,
            u_ref_local,
            idx_history,
            leg.coupled_state_idx,
            beta_val,
            log_gibbs_weights;
            compute_gradients=compute_gradients,
        )
        grad_F_0, F_0 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi_local,
            energies_local,
            u_ref_local,
            idx_history,
            leg.decoupled_state_idx,
            beta_val,
            log_gibbs_weights;
            compute_gradients=compute_gradients,
        )
    end

    return grad_F_1 .- grad_F_0, F_1 - F_0
end


Base.@kwdef struct OptimizationPool
    name::Symbol
    global_indices::Vector{Int}
    trainable_indices::Vector{Int}
    sigma_global_indices::Vector{Int}
    sigma_trainable_indices::Vector{Int}
    epsilon_global_indices::Vector{Int}
    epsilon_trainable_indices::Vector{Int}
    max_phi_step::Float64
    max_sigma_drift::Union{Nothing, Float64}
    max_epsilon_drift::Union{Nothing, Float64}
    reference_penalty_strength::Float64
end


"""
    optimization_active_pools(trainable_position_map, parameter_pools)

Build the runtime pool metadata used by the joint optimizer. Pool-specific step
caps and drift limits are applied after solving one joint Fisher-preconditioned
update over the full trainable parameter vector.
"""
function optimization_active_pools(
    trainable_position_map::Dict{Int, Int},
    parameter_pools::Vector{ResolvedParameterPool},
)
    pools = OptimizationPool[]

    for pool in parameter_pools
        global_indices = [idx for idx in pool.global_indices if haskey(trainable_position_map, idx)]
        isempty(global_indices) && continue
        push!(
            pools,
            OptimizationPool(
                name=pool.name,
                global_indices=global_indices,
                trainable_indices=[trainable_position_map[idx] for idx in global_indices],
                sigma_global_indices=[idx for idx in pool.sigma_global_indices if haskey(trainable_position_map, idx)],
                sigma_trainable_indices=[trainable_position_map[idx] for idx in pool.sigma_global_indices if haskey(trainable_position_map, idx)],
                epsilon_global_indices=[idx for idx in pool.epsilon_global_indices if haskey(trainable_position_map, idx)],
                epsilon_trainable_indices=[trainable_position_map[idx] for idx in pool.epsilon_global_indices if haskey(trainable_position_map, idx)],
                max_phi_step=pool.max_phi_step,
                max_sigma_drift=pool.max_sigma_drift,
                max_epsilon_drift=pool.max_epsilon_drift,
                reference_penalty_strength=pool.reference_penalty_strength,
            ),
        )
    end

    isempty(pools) && throw(ArgumentError("No optimization pools were constructed."))
    return pools
end

function apply_pool_reference_penalty!(
    grad_loss::AbstractVector{AT},
    fim_joint::AbstractMatrix{AT},
    theta_active::AbstractVector{FT},
    theta_ref::AbstractVector{FT},
    chain_rule_multiplier::AbstractVector{FT},
    optimization_pools::Vector{OptimizationPool},
) where {AT <: AbstractFloat, FT <: AbstractFloat}
    for pool in optimization_pools
        strength = AT(pool.reference_penalty_strength)
        strength <= zero(AT) && continue
        for (idx_global, idx_local) in zip(pool.global_indices, pool.trainable_indices)
            delta_theta = AT(theta_active[idx_global] - theta_ref[idx_global])
            chain = AT(chain_rule_multiplier[idx_global])
            grad_loss[idx_local] += strength * delta_theta * chain
            fim_joint[idx_local, idx_local] += strength * chain * chain
        end
    end
    return nothing
end

function clip_update_by_pool!(
    update_direction_train::AbstractVector{AT},
    optimization_pools::Vector{OptimizationPool},
) where {AT <: AbstractFloat}
    clip_stats = Dict{Symbol, NamedTuple}()
    for pool in optimization_pools
        if isempty(pool.trainable_indices)
            clip_stats[pool.name] = (max_phi_update=zero(AT), clip_scaling=one(AT))
            continue
        end
        max_phi_update = maximum(abs.(update_direction_train[pool.trainable_indices]))
        clip_scaling = one(AT)
        pool_cap = AT(pool.max_phi_step)
        if max_phi_update > pool_cap
            clip_scaling = pool_cap / max_phi_update
            update_direction_train[pool.trainable_indices] .*= clip_scaling
            max_phi_update = pool_cap
        end
        clip_stats[pool.name] = (max_phi_update=max_phi_update, clip_scaling=clip_scaling)
    end
    return clip_stats
end

function pool_drift_metrics(
    theta_values::AbstractVector{FT},
    theta_ref::AbstractVector{FT},
    optimization_pools::Vector{OptimizationPool},
) where {FT <: AbstractFloat}
    drift_metrics = Dict{Symbol, NamedTuple}()
    drift_ok = true
    for pool in optimization_pools
        sigma_drift = isempty(pool.sigma_global_indices) ? zero(FT) : maximum(abs.(theta_values[pool.sigma_global_indices] .- theta_ref[pool.sigma_global_indices]))
        epsilon_drift = isempty(pool.epsilon_global_indices) ? zero(FT) : maximum(abs.(theta_values[pool.epsilon_global_indices] .- theta_ref[pool.epsilon_global_indices]))
        sigma_ready = isnothing(pool.max_sigma_drift) || sigma_drift <= FT(pool.max_sigma_drift)
        epsilon_ready = isnothing(pool.max_epsilon_drift) || epsilon_drift <= FT(pool.max_epsilon_drift)
        drift_ok &= sigma_ready && epsilon_ready
        drift_metrics[pool.name] = (
            sigma_drift=sigma_drift,
            epsilon_drift=epsilon_drift,
            sigma_ready=sigma_ready,
            epsilon_ready=epsilon_ready,
        )
    end
    return drift_ok, drift_metrics
end


function line_search_noise_tolerance(
    error_residual::RT,
    tolerance_fraction::TT,
) where {RT <: AbstractFloat, TT <: AbstractFloat}
    AT = promote_energy_analysis_type(error_residual, tolerance_fraction)
    return AT(tolerance_fraction) * abs(AT(error_residual))
end


function line_search_residual_acceptable(
    error_residual::RT,
    error_residual_prop::PT,
    tolerance_fraction::TT,
) where {RT <: AbstractFloat, PT <: AbstractFloat, TT <: AbstractFloat}
    return line_search_residual_acceptable(
        error_residual,
        error_residual_prop,
        tolerance_fraction,
        zero(promote_energy_analysis_type(error_residual, error_residual_prop, tolerance_fraction)),
    )
end


function line_search_acceptance_threshold(
    error_residual::RT,
    tolerance_fraction::TT,
    additional_improvement_requirement::ATX,
) where {RT <: AbstractFloat, TT <: AbstractFloat, ATX <: AbstractFloat}
    noise_tolerance = line_search_noise_tolerance(error_residual, tolerance_fraction)
    AT = promote_energy_analysis_type(error_residual, tolerance_fraction, additional_improvement_requirement)
    threshold_floor = max(AT(noise_tolerance), sqrt(eps(AT)))
    raw_threshold = abs(AT(error_residual)) + AT(noise_tolerance) - max(zero(AT), AT(additional_improvement_requirement))
    return max(threshold_floor, raw_threshold)
end


function line_search_acceptance_threshold(
    error_residual::RT,
    tolerance_fraction::TT,
) where {RT <: AbstractFloat, TT <: AbstractFloat}
    return line_search_acceptance_threshold(
        error_residual,
        tolerance_fraction,
        zero(promote_energy_analysis_type(error_residual, tolerance_fraction)),
    )
end


function line_search_residual_acceptable(
    error_residual::RT,
    error_residual_prop::PT,
    tolerance_fraction::TT,
    additional_improvement_requirement::ATX,
) where {RT <: AbstractFloat, PT <: AbstractFloat, TT <: AbstractFloat, ATX <: AbstractFloat}
    AT = promote_energy_analysis_type(error_residual, error_residual_prop, tolerance_fraction, additional_improvement_requirement)
    acceptance_threshold = line_search_acceptance_threshold(
        error_residual,
        tolerance_fraction,
        additional_improvement_requirement,
    )
    return abs(AT(error_residual_prop)) <= acceptance_threshold
end

function optimization_analysis_type(
    leg_artifacts::Vector{LegArtifacts},
    targets::Vector{AbstractTrainingTarget},
    ::Type{FT},
    beta_val::BT,
    dG_std_corr::DT,
) where {FT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat}
    AT = promote_energy_analysis_type(FT, beta_val, dG_std_corr)
    for leg in leg_artifacts
        if leg.u_ref !== nothing
            AT = promote_type(AT, eltype(leg.u_ref))
        end
        if leg.active_bias !== nothing
            AT = promote_type(AT, eltype(awh_log_gibbs_weights(leg.active_bias)))
        end
        if leg.include_pv
            AT = promote_type(AT, typeof(leg.p0_energy_per_vol))
        end
    end
    for target in targets
        if target isa ResolvedCycleFreeEnergyTarget
            AT = promote_type(AT, typeof(target.target_dG_kcal_mol))
            if !isnothing(target.tolerance_kcal_mol)
                AT = promote_type(AT, typeof(target.tolerance_kcal_mol))
            end
        elseif target isa ResolvedStateObservableTarget
            AT = promote_type(AT, typeof(target.target_value))
            if !isnothing(target.tolerance)
                AT = promote_type(AT, typeof(target.tolerance))
            end
        end
    end
    return AT
end

Base.@kwdef struct TargetEvaluation
    name::Symbol = :unknown
    kind::Symbol = :unknown
    prediction::Any = 0.0
    target_value::Any = 0.0
    residual::Any = 0.0
    tolerance::Any = 1.0
    normalized_residual::Any = 0.0
    in_band::Bool = false
    weight::Any = 1.0
    loss::Any = 0.0
    gradient::Any = nothing
    ess::Any = 0.0
    unit_label::String = ""
    leg::Any = nothing
    state_label::Any = nothing
end

Base.@kwdef struct OptimizationConfidenceSummary
    enabled::Bool = false
    scale::Any = 1.0
    prediction_disagreement::Any = 0.0
    objective_disagreement::Any = 0.0
    endpoint_disagreement::Any = 0.0
    cycle_disagreement::Any = 0.0
    gradient_disagreement::Any = 0.0
    additional_residual_requirement::Any = 0.0
    eligible_targets::Int = 0
    skipped_targets::Vector{Symbol} = Symbol[]
    eligible_legs::Int = 0
    skipped_legs::Vector{Symbol} = Symbol[]
end

@inline function huber_loss_value(
    residual::RT,
    huber_delta::DT,
) where {RT <: AbstractFloat, DT <: AbstractFloat}
    AT = promote_energy_analysis_type(residual, huber_delta)
    abs_residual = abs(AT(residual))
    delta = AT(huber_delta)
    if abs_residual <= delta
        return AT(0.5) * abs_residual * abs_residual
    end
    return delta * (abs_residual - AT(0.5) * delta)
end

@inline function huber_loss_derivative(
    residual::RT,
    huber_delta::DT,
) where {RT <: AbstractFloat, DT <: AbstractFloat}
    AT = promote_energy_analysis_type(residual, huber_delta)
    residual_AT = AT(residual)
    delta = AT(huber_delta)
    return abs(residual_AT) <= delta ? residual_AT : delta * sign(residual_AT)
end

@inline function default_target_relative_tolerance(
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {AT <: AbstractFloat}
    rel_tol = AT(opt_cfg.default_target_relative_tolerance)
    rel_tol > zero(AT) ||
        throw(ArgumentError("OptimizationConfig.default_target_relative_tolerance must be positive."))
    return rel_tol
end

function default_target_tolerance_native(
    target::ResolvedCycleFreeEnergyTarget,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {AT <: AbstractFloat}
    rel_tol = default_target_relative_tolerance(opt_cfg, AT)
    abs_floor = AT(opt_cfg.cycle_target_absolute_tolerance_kcal_mol)
    abs_floor > zero(AT) ||
        throw(ArgumentError("OptimizationConfig.cycle_target_absolute_tolerance_kcal_mol must be positive."))
    target_mag = abs(AT(target.target_dG_kcal_mol))
    return max(rel_tol * target_mag, abs_floor)
end

function default_target_tolerance_native(
    target::ResolvedStateObservableTarget,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {AT <: AbstractFloat}
    rel_tol = default_target_relative_tolerance(opt_cfg, AT)
    abs_floor = AT(opt_cfg.observable_target_absolute_tolerance)
    abs_floor > zero(AT) ||
        throw(ArgumentError("OptimizationConfig.observable_target_absolute_tolerance must be positive."))
    target_mag = abs(AT(target.target_value))
    return max(rel_tol * target_mag, abs_floor)
end

function target_tolerance_value(
    target::ResolvedCycleFreeEnergyTarget,
    beta_val::BT,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {BT <: AbstractFloat, AT <: AbstractFloat}
    native_tolerance = isnothing(target_configured_tolerance(target)) ?
        default_target_tolerance_native(target, opt_cfg, AT) :
        AT(target_configured_tolerance(target))
    native_tolerance > zero(AT) ||
        throw(ArgumentError("Resolved cycle target `$(target.name)` has a non-positive tolerance."))
    return native_tolerance * AT(4.184) * AT(beta_val)
end

function target_tolerance_value(
    target::ResolvedStateObservableTarget,
    beta_val,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {AT <: AbstractFloat}
    tolerance = isnothing(target_configured_tolerance(target)) ?
        default_target_tolerance_native(target, opt_cfg, AT) :
        AT(target_configured_tolerance(target))
    tolerance > zero(AT) ||
        throw(ArgumentError("Resolved observable target `$(target.name)` has a non-positive tolerance."))
    return tolerance
end

@inline function normalized_target_residual(
    prediction::PT,
    target_value::TT,
    tolerance::AT,
) where {PT <: AbstractFloat, TT <: AbstractFloat, AT <: AbstractFloat}
    tolerance > zero(AT) || throw(ArgumentError("Target tolerance must be positive."))
    return (AT(prediction) - AT(target_value)) / tolerance
end

@inline target_residual_within_tolerance(
    residual::RT,
    tolerance::TT,
) where {RT <: AbstractFloat, TT <: AbstractFloat} = abs(residual) <= tolerance

function evaluate_target_loss(
    target::AbstractTrainingTarget,
    prediction::PT,
    target_value::TT,
    beta_val::BT,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {PT <: AbstractFloat, TT <: AbstractFloat, BT <: AbstractFloat, AT <: AbstractFloat}
    tolerance = target_tolerance_value(target, beta_val, opt_cfg, AT)
    residual = AT(prediction) - AT(target_value)
    normalized_residual = normalized_target_residual(prediction, target_value, tolerance)
    weight = target_weight(target, AT)
    loss = weight * huber_loss_value(normalized_residual, AT(opt_cfg.huber_delta))
    derivative = weight * huber_loss_derivative(normalized_residual, AT(opt_cfg.huber_delta)) / tolerance
    return (
        residual=residual,
        tolerance=tolerance,
        normalized_residual=normalized_residual,
        loss=loss,
        derivative=derivative,
        in_band=target_residual_within_tolerance(residual, tolerance),
    )
end

function compute_leg_log_mixture_denom(
    leg::LegArtifacts,
    beta_val::BT,
    volumes::AbstractVector{VT};
    frame_indices::Union{Nothing, AbstractVector{Int}}=nothing,
) where {BT <: AbstractFloat, VT <: AbstractFloat}
    u_ref_local = isnothing(frame_indices) ? leg.u_ref : @view leg.u_ref[frame_indices, :]
    volumes_local = leg.include_pv && !isnothing(frame_indices) ? volumes[frame_indices] : volumes
    log_gibbs_weights = awh_log_gibbs_weights(leg.active_bias)
    if leg.include_pv
        return reference_log_mixture_denominator(
            u_ref_local,
            log_gibbs_weights,
            beta_val;
            volumes=volumes_local,
            P0_energy_per_vol=leg.p0_energy_per_vol,
        )
    end
    return reference_log_mixture_denominator(
        u_ref_local,
        log_gibbs_weights,
        beta_val,
    )
end

function compute_leg_state_weights_from_log_mixture_denom(
    leg::LegArtifacts,
    energies_current::AbstractMatrix{ET},
    log_mixture_denom::AbstractVector{DT},
    state_idx::Int,
    beta_val::BT,
    volumes::AbstractVector{VT};
    frame_indices::Union{Nothing, AbstractVector{Int}}=nothing,
) where {ET <: AbstractFloat, DT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat}
    energies_local = isnothing(frame_indices) ? energies_current : @view energies_current[frame_indices, :]
    log_mixture_local = isnothing(frame_indices) ? log_mixture_denom : @view log_mixture_denom[frame_indices]
    volumes_local = leg.include_pv && !isnothing(frame_indices) ? volumes[frame_indices] : volumes

    M, num_lambda = size(energies_local)
    1 <= state_idx <= num_lambda ||
        throw(ArgumentError("Requested target state index $state_idx outside valid range 1:$num_lambda for leg $(leg.name)."))

    AT = promote_energy_analysis_type(
        energies_local,
        log_mixture_local,
        beta_val,
        volumes_local,
        leg.p0_energy_per_vol,
    )
    beta_AT = AT(beta_val)
    p0_AT = AT(leg.p0_energy_per_vol)
    log_weights = zeros(AT, M)

    @inbounds for frame_idx in 1:M
        pv_term = isempty(volumes_local) ? zero(AT) : beta_AT * p0_AT * AT(volumes_local[frame_idx])
        log_weights[frame_idx] =
            -(beta_AT * AT(energies_local[frame_idx, state_idx]) + pv_term) -
            AT(log_mixture_local[frame_idx])
    end

    max_log_weight = maximum(log_weights)
    weights = exp.(log_weights .- max_log_weight)
    weights ./= sum(weights)
    ess = one(AT) / sum(weights .^ 2)
    return weights, ess
end

function compute_leg_state_observable_estimate(
    leg::LegArtifacts,
    trainable_param_names::Vector{String},
    energy_gradients_phi,
    observable_values::AbstractVector{OV},
    observable_gradients_phi,
    energies_current::AbstractMatrix{ET},
    log_mixture_denom::AbstractVector{DT},
    state_idx::Int,
    beta_val::BT,
    volumes::AbstractVector{VT};
    compute_gradients::Bool=true,
    frame_indices::Union{Nothing, AbstractVector{Int}}=nothing,
) where {OV <: AbstractFloat, ET <: AbstractFloat, DT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat}
    observable_values_local = isnothing(frame_indices) ? observable_values : @view observable_values[frame_indices]
    weights, ess = compute_leg_state_weights_from_log_mixture_denom(
        leg,
        energies_current,
        log_mixture_denom,
        state_idx,
        beta_val,
        volumes;
        frame_indices=frame_indices,
    )

    AT = promote_energy_analysis_type(
        observable_values_local,
        energies_current,
        log_mixture_denom,
        beta_val,
        volumes,
        leg.p0_energy_per_vol,
    )
    observable_prediction = zero(AT)
    @inbounds for frame_idx in eachindex(weights)
        observable_prediction += weights[frame_idx] * AT(observable_values_local[frame_idx])
    end

    observable_gradient = zeros(AT, length(trainable_param_names))
    if !compute_gradients
        return observable_gradient, observable_prediction, ess
    end

    beta_AT = AT(beta_val)
    for (param_idx, p_key) in enumerate(trainable_param_names)
        if haskey(observable_gradients_phi, p_key)
            obs_grad_vec = observable_gradients_phi[p_key]
            obs_grad_local = isnothing(frame_indices) ? obs_grad_vec : @view obs_grad_vec[frame_indices]
            @inbounds for frame_idx in eachindex(weights)
                observable_gradient[param_idx] += weights[frame_idx] * AT(obs_grad_local[frame_idx])
            end
        end

        energy_grad_matrix = energy_gradients_phi[p_key]
        energy_grad_local = if isnothing(frame_indices)
            @view energy_grad_matrix[:, state_idx]
        else
            @view energy_grad_matrix[frame_indices, state_idx]
        end
        @inbounds for frame_idx in eachindex(weights)
            centered_observable = AT(observable_values_local[frame_idx]) - observable_prediction
            observable_gradient[param_idx] -=
                weights[frame_idx] * centered_observable * beta_AT * AT(energy_grad_local[frame_idx])
        end
    end

    return observable_gradient, observable_prediction, ess
end

function cycle_prediction_and_gradient_from_leg_cache(
    leg_artifacts::Vector{LegArtifacts},
    leg_endpoint_value_cache,
    leg_endpoint_gradient_cache,
    dG_std_corr::DT,
    ::Type{AT};
    compute_gradients::Bool=true,
) where {DT <: AbstractFloat, AT <: AbstractFloat}
    prediction = AT(dG_std_corr)
    gradient = zeros(AT, isempty(leg_endpoint_gradient_cache) ? 0 : length(first(values(leg_endpoint_gradient_cache))))
    for leg in leg_artifacts
        coeff = AT(leg.coefficient)
        prediction += coeff * AT(leg_endpoint_value_cache[leg.name])
        if compute_gradients
            gradient .+= coeff .* AT.(leg_endpoint_gradient_cache[leg.name])
        end
    end
    return gradient, prediction
end

function evaluate_training_targets(
    targets::Vector{AbstractTrainingTarget},
    leg_artifacts::Vector{LegArtifacts},
    leg_energies_cache,
    leg_grads_phi,
    leg_volumes_cache,
    leg_endpoint_value_cache,
    leg_endpoint_gradient_cache,
    leg_log_mixture_denom_cache,
    leg_ess_cache,
    params::Vector{FT},
    param_names::Vector{String},
    trainable_param_names::Vector{String},
    chain_rule_multiplier,
    beta_val::BT,
    dG_std_corr::DT,
    opt_cfg::OptimizationConfig,
    ::Type{AT};
    compute_gradients::Bool=true,
) where {FT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat, AT <: AbstractFloat}
    leg_artifacts_by_name = Dict(leg.name => leg for leg in leg_artifacts)
    target_evaluations = TargetEvaluation[]
    objective_value = zero(AT)
    objective_gradient = compute_gradients ? zeros(AT, length(trainable_param_names)) : zeros(AT, 0)
    observable_target_cache = Dict{Symbol, Any}()

    for target in targets
        if target isa ResolvedCycleFreeEnergyTarget
            grad_target, prediction = cycle_prediction_and_gradient_from_leg_cache(
                leg_artifacts,
                leg_endpoint_value_cache,
                leg_endpoint_gradient_cache,
                dG_std_corr,
                AT;
                compute_gradients=compute_gradients,
            )
            target_value = target_reference_value(target, beta_val, AT)
            target_loss_eval = evaluate_target_loss(
                target,
                prediction,
                target_value,
                beta_val,
                opt_cfg,
                AT,
            )
            objective_value += target_loss_eval.loss
            if compute_gradients
                objective_gradient .+= target_loss_eval.derivative .* grad_target
            end

            cycle_ess = isempty(leg_ess_cache) ? zero(AT) : minimum(values(leg_ess_cache))
            push!(
                target_evaluations,
                TargetEvaluation(
                    name=target.name,
                    kind=:cycle_free_energy,
                    prediction=prediction,
                    target_value=target_value,
                    residual=target_loss_eval.residual,
                    tolerance=target_loss_eval.tolerance,
                    normalized_residual=target_loss_eval.normalized_residual,
                    in_band=target_loss_eval.in_band,
                    weight=target_weight(target, AT),
                    loss=target_loss_eval.loss,
                    gradient=compute_gradients ? grad_target : nothing,
                    ess=cycle_ess,
                    unit_label=target_unit_label(target),
                ),
            )
        elseif target isa ResolvedStateObservableTarget
            leg = leg_artifacts_by_name[target.leg]
            obs_values, obs_grads_theta = evaluate_leg_state_observable(
                leg,
                params,
                param_names,
                target.observable,
                target.state_idx;
                compute_gradients=compute_gradients,
            )

            obs_grads_phi = Dict{String, Vector{FT}}()
            if compute_gradients
                for (idx, p_key) in enumerate(param_names)
                    obs_grads_phi[p_key] = obs_grads_theta[p_key] .* chain_rule_multiplier[idx]
                end
            end

            grad_target, prediction, state_ess = compute_leg_state_observable_estimate(
                leg,
                trainable_param_names,
                leg_grads_phi[leg.name],
                obs_values,
                obs_grads_phi,
                leg_energies_cache[leg.name],
                leg_log_mixture_denom_cache[leg.name],
                target.state_idx,
                beta_val,
                leg_volumes_cache[leg.name];
                compute_gradients=compute_gradients,
            )
            target_value = target_reference_value(target, beta_val, AT)
            target_loss_eval = evaluate_target_loss(
                target,
                prediction,
                target_value,
                beta_val,
                opt_cfg,
                AT,
            )
            objective_value += target_loss_eval.loss
            if compute_gradients
                objective_gradient .+= target_loss_eval.derivative .* grad_target
            end

            observable_target_cache[target.name] = (
                leg_name=target.leg,
                state_idx=target.state_idx,
                state_label=target.state_label,
                values=obs_values,
                gradients_phi=obs_grads_phi,
            )

            push!(
                target_evaluations,
                TargetEvaluation(
                    name=target.name,
                    kind=:state_observable,
                    prediction=prediction,
                    target_value=target_value,
                    residual=target_loss_eval.residual,
                    tolerance=target_loss_eval.tolerance,
                    normalized_residual=target_loss_eval.normalized_residual,
                    in_band=target_loss_eval.in_band,
                    weight=target_weight(target, AT),
                    loss=target_loss_eval.loss,
                    gradient=compute_gradients ? grad_target : nothing,
                    ess=state_ess,
                    unit_label=target_unit_label(target),
                    leg=target.leg,
                    state_label=target.state_label,
                ),
            )
        else
            throw(ArgumentError("Unsupported resolved target type $(typeof(target))."))
        end
    end

    return target_evaluations, objective_value, objective_gradient, observable_target_cache
end

function split_half_frame_indices(n_frames::Int, min_frames::Int)
    min_required = max(2, min_frames)
    if n_frames < min_required
        return nothing
    end
    n_first = fld(n_frames, 2)
    n_second = n_frames - n_first
    if n_first < 1 || n_second < 1
        return nothing
    end
    return (1:n_first, (n_first + 1):n_frames)
end

function compute_optimization_confidence_summary(
    targets::Vector{AbstractTrainingTarget},
    leg_artifacts::Vector{LegArtifacts},
    leg_energies_cache,
    leg_grads_phi,
    leg_volumes_cache,
    leg_endpoint_value_cache,
    leg_endpoint_gradient_cache,
    leg_log_mixture_denom_cache,
    observable_target_cache,
    trainable_param_names::Vector{String},
    objective_gradient::AbstractVector{AT},
    beta_val::BT,
    dG_std_corr::DT,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {AT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat}
    if opt_cfg.optimization_confidence_mode != :split_half
        return OptimizationConfidenceSummary(
            enabled=false,
            scale=one(AT),
            prediction_disagreement=zero(AT),
            objective_disagreement=zero(AT),
            endpoint_disagreement=zero(AT),
            cycle_disagreement=zero(AT),
            gradient_disagreement=zero(AT),
            additional_residual_requirement=zero(AT),
            eligible_targets=0,
            skipped_targets=Symbol[],
            eligible_legs=0,
            skipped_legs=Symbol[],
        )
    end

    prediction_disagreements = AT[]
    objective_half_1 = zero(AT)
    objective_half_2 = zero(AT)
    gradient_half_1 = zeros(AT, length(objective_gradient))
    gradient_half_2 = zeros(AT, length(objective_gradient))
    eligible_targets = 0
    skipped_targets = Symbol[]

    leg_artifacts_by_name = Dict(leg.name => leg for leg in leg_artifacts)

    for target in targets
        if target isa ResolvedCycleFreeEnergyTarget
            half_1_prediction = AT(dG_std_corr)
            half_2_prediction = AT(dG_std_corr)
            half_1_gradient = zeros(AT, length(objective_gradient))
            half_2_gradient = zeros(AT, length(objective_gradient))
            split_available = true

            for leg in leg_artifacts
                frame_split = split_half_frame_indices(
                    size(leg_energies_cache[leg.name], 1),
                    opt_cfg.optimization_confidence_min_frames,
                )
                if isnothing(frame_split)
                    split_available = false
                    break
                end

                first_half, second_half = frame_split
                _, dG_half_1 = compute_leg_endpoint_state(
                    leg,
                    trainable_param_names,
                    leg_grads_phi[leg.name],
                    leg_energies_cache[leg.name],
                    beta_val,
                    leg_volumes_cache[leg.name];
                    compute_gradients=false,
                    frame_indices=first_half,
                )
                _, dG_half_2 = compute_leg_endpoint_state(
                    leg,
                    trainable_param_names,
                    leg_grads_phi[leg.name],
                    leg_energies_cache[leg.name],
                    beta_val,
                    leg_volumes_cache[leg.name];
                    compute_gradients=false,
                    frame_indices=second_half,
                )
                grad_half_1, _ = compute_leg_endpoint_state(
                    leg,
                    trainable_param_names,
                    leg_grads_phi[leg.name],
                    leg_energies_cache[leg.name],
                    beta_val,
                    leg_volumes_cache[leg.name];
                    compute_gradients=true,
                    frame_indices=first_half,
                )
                grad_half_2, _ = compute_leg_endpoint_state(
                    leg,
                    trainable_param_names,
                    leg_grads_phi[leg.name],
                    leg_energies_cache[leg.name],
                    beta_val,
                    leg_volumes_cache[leg.name];
                    compute_gradients=true,
                    frame_indices=second_half,
                )

                coeff = AT(leg.coefficient)
                half_1_prediction += coeff * AT(dG_half_1)
                half_2_prediction += coeff * AT(dG_half_2)
                half_1_gradient .+= coeff .* AT.(grad_half_1)
                half_2_gradient .+= coeff .* AT.(grad_half_2)
            end

            if !split_available
                push!(skipped_targets, target.name)
                continue
            end

            eligible_targets += 1
            target_value = target_reference_value(target, beta_val, AT)
            loss_half_1 = evaluate_target_loss(
                target,
                half_1_prediction,
                target_value,
                beta_val,
                opt_cfg,
                AT,
            )
            loss_half_2 = evaluate_target_loss(
                target,
                half_2_prediction,
                target_value,
                beta_val,
                opt_cfg,
                AT,
            )
            objective_half_1 += loss_half_1.loss
            objective_half_2 += loss_half_2.loss
            gradient_half_1 .+= loss_half_1.derivative .* half_1_gradient
            gradient_half_2 .+= loss_half_2.derivative .* half_2_gradient
            prediction_gap = abs(half_1_prediction - half_2_prediction) / max(loss_half_1.tolerance, loss_half_2.tolerance)
            push!(prediction_disagreements, prediction_gap)
        elseif target isa ResolvedStateObservableTarget
            cache = get(observable_target_cache, target.name, nothing)
            if isnothing(cache)
                push!(skipped_targets, target.name)
                continue
            end

            frame_split = split_half_frame_indices(length(cache.values), opt_cfg.optimization_confidence_min_frames)
            if isnothing(frame_split)
                push!(skipped_targets, target.name)
                continue
            end

            eligible_targets += 1
            leg = leg_artifacts_by_name[target.leg]
            first_half, second_half = frame_split
            grad_half_1, prediction_half_1, _ = compute_leg_state_observable_estimate(
                leg,
                trainable_param_names,
                leg_grads_phi[leg.name],
                cache.values,
                cache.gradients_phi,
                leg_energies_cache[leg.name],
                leg_log_mixture_denom_cache[leg.name],
                target.state_idx,
                beta_val,
                leg_volumes_cache[leg.name];
                compute_gradients=true,
                frame_indices=first_half,
            )
            grad_half_2, prediction_half_2, _ = compute_leg_state_observable_estimate(
                leg,
                trainable_param_names,
                leg_grads_phi[leg.name],
                cache.values,
                cache.gradients_phi,
                leg_energies_cache[leg.name],
                leg_log_mixture_denom_cache[leg.name],
                target.state_idx,
                beta_val,
                leg_volumes_cache[leg.name];
                compute_gradients=true,
                frame_indices=second_half,
            )

            target_value = target_reference_value(target, beta_val, AT)
            loss_half_1 = evaluate_target_loss(
                target,
                prediction_half_1,
                target_value,
                beta_val,
                opt_cfg,
                AT,
            )
            loss_half_2 = evaluate_target_loss(
                target,
                prediction_half_2,
                target_value,
                beta_val,
                opt_cfg,
                AT,
            )
            objective_half_1 += loss_half_1.loss
            objective_half_2 += loss_half_2.loss
            gradient_half_1 .+= loss_half_1.derivative .* grad_half_1
            gradient_half_2 .+= loss_half_2.derivative .* grad_half_2
            prediction_gap = abs(prediction_half_1 - prediction_half_2) / max(loss_half_1.tolerance, loss_half_2.tolerance)
            push!(prediction_disagreements, prediction_gap)
        else
            throw(ArgumentError("Unsupported target type $(typeof(target)) in confidence summary."))
        end
    end

    if eligible_targets == 0
        return OptimizationConfidenceSummary(
            enabled=false,
            scale=one(AT),
            prediction_disagreement=zero(AT),
            objective_disagreement=zero(AT),
            endpoint_disagreement=zero(AT),
            cycle_disagreement=zero(AT),
            gradient_disagreement=zero(AT),
            additional_residual_requirement=zero(AT),
            eligible_targets=0,
            skipped_targets=skipped_targets,
            eligible_legs=0,
            skipped_legs=skipped_targets,
        )
    end

    prediction_disagreement = isempty(prediction_disagreements) ? zero(AT) : maximum(prediction_disagreements)
    objective_disagreement = abs(objective_half_1 - objective_half_2)
    gradient_disagreement = norm(gradient_half_1 - gradient_half_2) / max(norm(objective_gradient), sqrt(eps(AT)))
    raw_penalty = max(prediction_disagreement, objective_disagreement, gradient_disagreement)
    min_scale = clamp(AT(opt_cfg.optimization_confidence_min_scale), zero(AT), one(AT))
    strength = max(zero(AT), AT(opt_cfg.optimization_confidence_scale_strength))
    scale = max(min_scale, one(AT) / (one(AT) + strength * raw_penalty))
    additional_requirement =
        max(zero(AT), AT(opt_cfg.optimization_confidence_residual_requirement_strength)) *
        objective_disagreement
    return OptimizationConfidenceSummary(
        enabled=true,
        scale=scale,
        prediction_disagreement=prediction_disagreement,
        objective_disagreement=objective_disagreement,
        endpoint_disagreement=prediction_disagreement,
        cycle_disagreement=objective_disagreement,
        gradient_disagreement=gradient_disagreement,
        additional_residual_requirement=additional_requirement,
        eligible_targets=eligible_targets,
        skipped_targets=skipped_targets,
        eligible_legs=eligible_targets,
        skipped_legs=skipped_targets,
    )
end


"""
    run_optimization_phase!(phi_active, theta_active, leg_artifacts, param_names,
                            trainable_param_names, trainable_param_indices,
                            trainable_position_map, parameter_pools, theta_ref,
                            theta_min, theta_max, phi_0, beta_val, dG_std_corr,
                            targets, opt_cfg)

Perform the inner optimization loop for one macro epoch. Each inner epoch
builds one joint Fisher-preconditioned update over the full trainable vector,
then applies pool-specific step caps and drift checks during line search.
"""
function run_optimization_phase!(
    phi_active::Vector{FT},
    theta_active::Vector{FT},
    leg_artifacts::Vector{LegArtifacts},
    param_names::Vector{String},
    trainable_param_names::Vector{String},
    trainable_param_indices::Vector{Int},
    trainable_position_map::Dict{Int, Int},
    parameter_pools::Vector{ResolvedParameterPool},
    param_families::Vector{Symbol},
    theta_ref::Vector{FT},
    theta_min::Vector{FT},
    theta_max::Vector{FT},
    phi_0::Vector{FT},
    beta_val::BT,
    dG_std_corr::DT,
    targets::Vector{AbstractTrainingTarget},
    opt_cfg::OptimizationConfig,
) where {FT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat}
    if isempty(leg_artifacts)
        throw(ArgumentError("run_optimization_phase! requires at least one leg artifact."))
    end
    if isempty(trainable_param_indices)
        throw(ArgumentError("run_optimization_phase! received no trainable parameters."))
    end
    if isempty(targets)
        throw(ArgumentError("run_optimization_phase! requires at least one resolved training target."))
    end
    if opt_cfg.max_inner_epochs <= 0
        throw(ArgumentError("OptimizationConfig.max_inner_epochs must be positive."))
    end

    AT = optimization_analysis_type(leg_artifacts, targets, FT, beta_val, dG_std_corr)

    theoretical_ess_ratio = exp(AT(-2.0) * AT(opt_cfg.kl_target))
    ess_threshold_ratio = AT(opt_cfg.ess_threshold_scale) * theoretical_ess_ratio
    active_pools = optimization_active_pools(
        trainable_position_map,
        parameter_pools,
    )
    has_cycle_target = any(target -> target isa ResolvedCycleFreeEnergyTarget, targets)

    ess_thresholds = Dict{Symbol, AT}()
    N_base = Dict{Symbol, AT}()
    for leg in leg_artifacts
        M_leg = AT(length(leg.logger_prod.active_idx_history))
        if M_leg <= zero(AT)
            throw(ArgumentError("Leg $(leg.name) has no production frames."))
        end
        ess_thresholds[leg.name] = M_leg * ess_threshold_ratio
        N_base[leg.name] = AT(leg.awh_prod.initial_sampl_n + leg.awh_prod.state.N_eff)
    end

    tiny_alpha_hits = 0
    phase2_exit_reason = :running
    macro_start_residual = AT(NaN)
    macro_end_residual = AT(NaN)
    best_macro_abs_residual = AT(Inf)
    best_macro_residual = AT(NaN)
    best_macro_epoch = 0
    phi_best_macro = copy(phi_active)
    theta_best_macro = copy(theta_active)
    best_pool_drifts = Dict{Symbol, NamedTuple}()

    inner_epoch = 1
    while inner_epoch <= opt_cfg.max_inner_epochs
        pool_label = join(string.(getfield.(active_pools, :name)), ",")
        @info "  >> Optimization Epoch: Active Pools = $pool_label"

        chain_rule_multiplier = get_chain_rule_multiplier(phi_active, theta_active, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid, param_families)

        fim_joint = zeros(AT, length(trainable_param_names), length(trainable_param_names))
        ess_current = Dict{Symbol, AT}()
        N_active = Dict{Symbol, AT}()
        leg_energies_cache = Dict{Symbol, Any}()
        leg_grads_phi = Dict{Symbol, Dict{String, Matrix{FT}}}()
        leg_volumes_cache = Dict{Symbol, Vector{AT}}()
        leg_log_mixture_denom_cache = Dict{Symbol, Vector{AT}}()
        leg_endpoint_value_cache = Dict{Symbol, AT}()
        leg_endpoint_gradient_cache = Dict{Symbol, Vector{AT}}()

        ess_threshold_broken = false

        for leg in leg_artifacts
            u_eval, grads_eval_theta = evaluate_leg_ensemble(
                leg,
                theta_active,
                param_names,
                compute_gradients=true,
            )

            # Gradients are evaluated in θ-space, but the optimizer lives in the
            # bounded-transform coordinates ϕ.
            grads_eval_phi = Dict{String, Matrix{FT}}()
            for (i, p_key) in enumerate(param_names)
                grads_eval_phi[p_key] = grads_eval_theta[p_key] .* chain_rule_multiplier[i]
            end

            volumes = leg_volumes(leg, AT)
            _, ess_leg = compute_leg_weights_and_ess(leg, u_eval, beta_val, volumes)
            ess_current[leg.name] = ess_leg

            M_leg = AT(length(leg.logger_prod.active_idx_history))
            N_active[leg.name] = N_base[leg.name] * (ess_leg / M_leg)

            if ess_leg < ess_thresholds[leg.name]
                ess_threshold_broken = true
                break
            end

            w_norm_leg, _ = compute_leg_weights_and_ess(leg, u_eval, beta_val, volumes)
            _, fim_leg = compute_empirical_gradients_and_fim(
                trainable_param_names,
                grads_eval_phi,
                w_norm_leg,
                leg.logger_prod.active_idx_history,
                beta_val,
            )
            fim_joint .+= fim_leg

            leg_energies_cache[leg.name] = u_eval
            leg_grads_phi[leg.name] = grads_eval_phi
            leg_volumes_cache[leg.name] = volumes
            leg_log_mixture_denom_cache[leg.name] = AT.(compute_leg_log_mixture_denom(leg, beta_val, volumes))

            if has_cycle_target
                grad_dG_leg, dG_leg = compute_leg_endpoint_state(
                    leg,
                    trainable_param_names,
                    grads_eval_phi,
                    u_eval,
                    beta_val,
                    volumes;
                    compute_gradients=true,
                )
                leg_endpoint_value_cache[leg.name] = AT(dG_leg)
                leg_endpoint_gradient_cache[leg.name] = AT.(grad_dG_leg)
            end
        end

        if ess_threshold_broken
            println("  [!] ESS threshold broken during state evaluation. Exiting Phase 2.")
            phase2_exit_reason = :ess_threshold
            break
        end

        target_evaluations, objective_value, objective_gradient, observable_target_cache = evaluate_training_targets(
            targets,
            leg_artifacts,
            leg_energies_cache,
            leg_grads_phi,
            leg_volumes_cache,
            leg_endpoint_value_cache,
            leg_endpoint_gradient_cache,
            leg_log_mixture_denom_cache,
            ess_current,
            theta_active,
            param_names,
            trainable_param_names,
            chain_rule_multiplier,
            beta_val,
            dG_std_corr,
            opt_cfg,
            AT;
            compute_gradients=true,
        )
        all_targets_in_band = all(target_eval -> target_eval.in_band, target_evaluations)

        if isnan(macro_start_residual)
            macro_start_residual = objective_value
        end

        if objective_value < best_macro_abs_residual
            best_macro_abs_residual = objective_value
            best_macro_residual = objective_value
            best_macro_epoch = inner_epoch
            phi_best_macro .= phi_active
            theta_best_macro .= theta_active
        end

        if all_targets_in_band
            macro_end_residual = objective_value
            phase2_exit_reason = :targets_within_tolerance
            @info "  [!] All optimization targets are within tolerance at the current parameters. Skipping further inner-loop updates."
            break
        end

        confidence_summary = compute_optimization_confidence_summary(
            targets,
            leg_artifacts,
            leg_energies_cache,
            leg_grads_phi,
            leg_volumes_cache,
            leg_endpoint_value_cache,
            leg_endpoint_gradient_cache,
            leg_log_mixture_denom_cache,
            observable_target_cache,
            trainable_param_names,
            objective_gradient,
            beta_val,
            dG_std_corr,
            opt_cfg,
            AT,
        )
        effective_kl_target = AT(opt_cfg.kl_target) * confidence_summary.scale

        grad_loss = copy(objective_gradient)
        apply_pool_reference_penalty!(
            grad_loss,
            fim_joint,
            theta_active,
            theta_ref,
            chain_rule_multiplier,
            active_pools,
        )

        grad_loss_active = grad_loss
        fim_active = fim_joint

        fim_diag = diag(fim_active)
        variance_threshold = maximum(fim_diag) * AT(1e-5)
        D_vec = [d > variance_threshold ? one(AT) / sqrt(d) : zero(AT) for d in fim_diag]
        D_mat = Diagonal(D_vec)

        fim_corr = D_mat * fim_active * D_mat
        grad_loss_scaled = D_vec .* grad_loss_active

        # Use a Fisher-preconditioned step in the active block, truncating small
        # eigendirections that are too noisy to trust.
        decomp = eigen(Symmetric(fim_corr))
        vals, vecs = decomp.values, decomp.vectors

        eigenvalue_tol = maximum(vals) * AT(opt_cfg.eigenvalue_tol_scale)
        inv_vals = [v > eigenvalue_tol ? one(AT) / v : zero(AT) for v in vals]
        fim_corr_inv = vecs * Diagonal(inv_vals) * transpose(vecs)

        base_step_scaled = fim_corr_inv * grad_loss_scaled
        base_step_active = D_vec .* base_step_scaled

        estimated_KL = AT(0.5) * dot(base_step_active, fim_active * base_step_active)
        if estimated_KL > effective_kl_target
            kl_scaling = sqrt(effective_kl_target / estimated_KL)
            update_direction_active = base_step_active * kl_scaling
        else
            kl_scaling = one(AT)
            update_direction_active = base_step_active
        end

        n_truncated = count(v -> v <= eigenvalue_tol, vals)
        fim_cond_raw = cond(fim_corr)

        update_direction_train = copy(update_direction_active)
        pool_clip_stats = clip_update_by_pool!(update_direction_train, active_pools)
        update_direction = zeros(AT, length(param_names))
        for (i_local, i_global) in enumerate(trainable_param_indices)
            update_direction[i_global] = update_direction_train[i_local]
        end

        alpha = one(AT)
        phi_prop = copy(phi_active)
        theta_prop = copy(theta_active)
        line_search_success = false
        accepted_residual = objective_value
        accepted_alpha = zero(AT)
        last_trial_alpha = zero(AT)
        best_trial_alpha = zero(AT)
        best_trial_residual = objective_value
        best_trial_abs_residual = abs(objective_value)
        best_trial_ess_ok = false
        best_trial_drift_ok = false
        ls_trials_run = 0
        ess_prop = Dict{Symbol, AT}()
        target_evaluations_prop = TargetEvaluation[]
        accepted_pool_drifts = Dict{Symbol, NamedTuple}()
        residual_noise_tolerance = line_search_noise_tolerance(
            objective_value,
            AT(opt_cfg.line_search_noise_tolerance_fraction),
        )
        residual_acceptance_threshold = line_search_acceptance_threshold(
            objective_value,
            AT(opt_cfg.line_search_noise_tolerance_fraction),
            confidence_summary.additional_residual_requirement,
        )

        # Backtracking line search enforces both residual improvement and a
        # minimum effective sample size under the proposed reweighting, together
        # with optional per-pool drift caps relative to the reference model.
        for ls_iter in 1:7
            trial_alpha = alpha
            last_trial_alpha = trial_alpha
            ls_trials_run = ls_iter
            phi_prop .= phi_active .- alpha .* update_direction
            theta_prop .= map_phi_to_theta(phi_prop, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid, param_families)
            drift_ok, pool_drifts_prop = pool_drift_metrics(theta_prop, theta_ref, active_pools)

            ess_ok = true
            leg_energies_prop_cache = Dict{Symbol, Any}()
            leg_endpoint_value_prop_cache = Dict{Symbol, AT}()

            for leg in leg_artifacts
                u_prop, _ = evaluate_leg_ensemble(
                    leg,
                    theta_prop,
                    param_names,
                    compute_gradients=false,
                )

                volumes = leg_volumes_cache[leg.name]
                _, ess_leg_prop = compute_leg_weights_and_ess(leg, u_prop, beta_val, volumes)
                ess_prop[leg.name] = ess_leg_prop
                if ess_leg_prop < ess_thresholds[leg.name]
                    ess_ok = false
                end

                leg_energies_prop_cache[leg.name] = u_prop
                if has_cycle_target
                    _, dG_leg_prop = compute_leg_endpoint_state(
                        leg,
                        trainable_param_names,
                        leg_grads_phi[leg.name],
                        u_prop,
                        beta_val,
                        volumes;
                        compute_gradients=false,
                    )
                    leg_endpoint_value_prop_cache[leg.name] = AT(dG_leg_prop)
                end
            end

            target_evaluations_prop, objective_value_prop, _, _ = evaluate_training_targets(
                targets,
                leg_artifacts,
                leg_energies_prop_cache,
                leg_grads_phi,
                leg_volumes_cache,
                leg_endpoint_value_prop_cache,
                Dict{Symbol, Vector{AT}}(),
                leg_log_mixture_denom_cache,
                ess_prop,
                theta_prop,
                param_names,
                trainable_param_names,
                nothing,
                beta_val,
                dG_std_corr,
                opt_cfg,
                AT;
                compute_gradients=false,
            )

            residual_ok = objective_value_prop <= residual_acceptance_threshold
            if objective_value_prop < best_trial_abs_residual
                best_trial_abs_residual = objective_value_prop
                best_trial_residual = objective_value_prop
                best_trial_alpha = trial_alpha
                best_trial_ess_ok = ess_ok
                best_trial_drift_ok = drift_ok
            end
            drift_msg = join(
                [
                    "$(pool.name)=σ$(round(pool_drifts_prop[pool.name].sigma_drift, digits=5))/ϵ$(round(pool_drifts_prop[pool.name].epsilon_drift, digits=5))"
                    for pool in active_pools
                ],
                " | ",
            )
            ess_msg = join(
                [
                    "$(leg.name)=$(round(get(ess_prop, leg.name, zero(AT)), digits=1))"
                    for leg in leg_artifacts
                ],
                " | ",
            )
            target_msg = join(
                [
                    "$(eval.name)=$(round(eval.prediction, digits=4))"
                    for eval in target_evaluations_prop
                ],
                " | ",
            )
            @info "    LS Iter $ls_iter (α=$(trial_alpha)): ESS[$ess_msg] | Drift[$drift_msg] | Targets[$target_msg] | Obj = $(round(objective_value_prop, digits=5)) | Gate[ess=$(ess_ok) | drift=$(drift_ok) | objective=$(residual_ok) <= $(round(residual_acceptance_threshold, digits=4))]"

            if ess_ok && drift_ok && residual_ok
                line_search_success = true
                accepted_alpha = trial_alpha
                phi_active .= phi_prop
                theta_active .= theta_prop
                accepted_residual = objective_value_prop
                accepted_pool_drifts = pool_drifts_prop
                target_evaluations = target_evaluations_prop
                @info "    -> Line search converged at α=$(accepted_alpha)."
                break
            else
                alpha *= AT(0.5)
            end
        end

        alpha_for_tiny_check = line_search_success ? accepted_alpha : last_trial_alpha
        if line_search_success && alpha_for_tiny_check <= AT(opt_cfg.tiny_alpha_cutoff)
            tiny_alpha_hits += 1
        else
            tiny_alpha_hits = 0
        end

        macro_end_residual = accepted_residual
        if accepted_residual < best_macro_abs_residual
            best_macro_abs_residual = accepted_residual
            best_macro_residual = accepted_residual
            best_macro_epoch = inner_epoch
            phi_best_macro .= phi_active
            theta_best_macro .= theta_active
            best_pool_drifts = accepted_pool_drifts
        end

        norm_grad_loss = norm(grad_loss_active)
        max_grad_loss = maximum(abs.(grad_loss_active))
        actual_max_phi_step = line_search_success ? maximum(abs.(update_direction_train)) * accepted_alpha : zero(AT)
        accepted_targets_in_band = all(target_eval -> target_eval.in_band, target_evaluations)

        @info "--- Current Parameter State ---"
        for i in eachindex(param_names)
            @info "  $(param_names[i]): $(round(theta_active[i], digits=6))"
        end

        @info "--- Optimization Metrics (Epoch $inner_epoch - Joint Pools) ---"
        @info "  Objective:   value = $(round(objective_value, digits=5)) | Accepted = $(round(accepted_residual, digits=5)) | Extra req = $(round(confidence_summary.additional_residual_requirement, digits=4)) | Threshold = $(round(residual_acceptance_threshold, digits=4)) | in_band=$(accepted_targets_in_band) | step_accepted=$(line_search_success)"
        @info "  Gradients:   Norm = $(round(norm_grad_loss, digits=5)) | Max = $(round(max_grad_loss, digits=5))"
        @info "  FIM (Corr):  Raw Cond Number = $(round(fim_cond_raw, digits=2)) | Truncated Eigs = $n_truncated / $(length(vals))"
        @info "  Trust Reg.:  Confidence = $(round(confidence_summary.scale, digits=4)) | KL target = $(round(effective_kl_target, digits=4))"
        @info "  Confidence:  prediction = $(round(confidence_summary.prediction_disagreement, digits=4)) | objective = $(round(confidence_summary.objective_disagreement, digits=4)) | gradient = $(round(confidence_summary.gradient_disagreement, digits=4)) | eligible_targets = $(confidence_summary.eligible_targets)"
        if !isempty(confidence_summary.skipped_targets)
            @info "  Confidence:  skipped_targets = $(join(String.(confidence_summary.skipped_targets), ","))"
        end
        @info "  KL Bound:    Est. KL = $(round(estimated_KL, digits=4)) | Target = $(round(effective_kl_target, digits=4)) | Scaling = $(round(kl_scaling, digits=4))"
        if line_search_success
            @info "  Line Search: Converged α = $accepted_alpha | Noise tol = $(round(residual_noise_tolerance, digits=4))"
            @info "  Actual Step: Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) (α=$accepted_alpha)"
        else
            @info "  Line Search: Failed after $ls_trials_run trials | Best trial α = $(best_trial_alpha) | Best residual = $(round(best_trial_residual, digits=3)) | Last tried α = $(last_trial_alpha) | Noise tol = $(round(residual_noise_tolerance, digits=4)) | Best gates[ess=$(best_trial_ess_ok) | drift=$(best_trial_drift_ok)]"
            @info "  Actual Step: Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) (α=0.0) | Last tried α = $(last_trial_alpha)"
        end
        @info "  Params (σ,ϵ): Min = $(round(minimum(theta_active[trainable_param_indices]), digits=5)) | Max = $(round(maximum(theta_active[trainable_param_indices]), digits=5))"
        for pool in active_pools
            clip_stat = pool_clip_stats[pool.name]
            drift_stat = get(accepted_pool_drifts, pool.name, (sigma_drift=zero(AT), epsilon_drift=zero(AT), sigma_ready=true, epsilon_ready=true))
            @info "  Pool $(pool.name): max_ϕ=$(round(clip_stat.max_phi_update, digits=6)) | clip=$(round(clip_stat.clip_scaling, digits=4)) | σ_drift=$(round(drift_stat.sigma_drift, digits=6)) | ϵ_drift=$(round(drift_stat.epsilon_drift, digits=6))"
        end
        for target_eval in target_evaluations
            state_msg = isnothing(target_eval.state_label) ? "" : " | leg=$(target_eval.leg) | state=$(target_eval.state_label)"
            @info "  Target $(target_eval.name): kind=$(target_eval.kind)$state_msg | pred=$(round(target_eval.prediction, digits=5)) $(target_eval.unit_label) | ref=$(round(target_eval.target_value, digits=5)) | residual=$(round(target_eval.residual, digits=5)) | tol=$(round(target_eval.tolerance, digits=5)) | norm=$(round(target_eval.normalized_residual, digits=5)) | in_band=$(target_eval.in_band) | weight=$(round(target_eval.weight, digits=4)) | loss=$(round(target_eval.loss, digits=5)) | ESS=$(round(target_eval.ess, digits=2))"
        end
        for leg in leg_artifacts
            cycle_suffix = haskey(leg_endpoint_value_cache, leg.name) ? " | ΔG=$(round(leg_endpoint_value_cache[leg.name], digits=3)) kT" : ""
            @info "  Leg $(leg.name): coeff=$(round(leg.coefficient, digits=3))$cycle_suffix | ESS=$(round(ess_current[leg.name], digits=1)) / $(round(ess_thresholds[leg.name], digits=1)) | N_active=$(round(N_active[leg.name], digits=1))"
        end
        println("---------------------------------------------------\n")

        if !line_search_success
            @info "  [!] Line search accepted no trial. Triggering Phase 3 resimulation."
            phase2_exit_reason = :line_search_fail
            break
        end

        if accepted_targets_in_band
            @info "  [!] Accepted parameter state satisfies all target tolerances. Triggering Phase 3 resimulation."
            phase2_exit_reason = :targets_within_tolerance
            break
        end

        if tiny_alpha_hits >= opt_cfg.max_tiny_alpha_hits
            @info "  [!] Repeated tiny line-search α detected ($tiny_alpha_hits consecutive epochs with α <= $(opt_cfg.tiny_alpha_cutoff)). Triggering Phase 3 resimulation."
            phase2_exit_reason = :tiny_alpha
            break
        end

        if actual_max_phi_step < AT(opt_cfg.min_phi_step)
            @info "  [!] Accepted line-search step vanished (Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) < min_phi_step = $(opt_cfg.min_phi_step)). Triggering Phase 3 resimulation."
            phase2_exit_reason = :step_vanish
            break
        end

        inner_epoch += 1
    end

    if phase2_exit_reason == :running
        phase2_exit_reason = :inner_epoch_cap
        @info "  [!] Reached max_inner_epochs=$(opt_cfg.max_inner_epochs). Triggering Phase 3 resimulation."
    end

    if (
        phase2_exit_reason == :line_search_fail ||
        phase2_exit_reason == :step_vanish ||
        phase2_exit_reason == :tiny_alpha ||
        phase2_exit_reason == :inner_epoch_cap
    ) && best_macro_epoch > 0
        phi_active .= phi_best_macro
        theta_active .= theta_best_macro
        macro_end_residual = best_macro_residual
        @info "  [!] Restored best inner-loop state from Epoch $best_macro_epoch (Residual = $(round(best_macro_residual, digits=3)))."
    end

    if !isempty(best_pool_drifts)
        for pool in active_pools
            if haskey(best_pool_drifts, pool.name)
                @info "Pool Drift ($(pool.name)): σ=$(round(best_pool_drifts[pool.name].sigma_drift, digits=8)) | ϵ=$(round(best_pool_drifts[pool.name].epsilon_drift, digits=8))"
            end
        end
    end

    return (
        phase2_exit_reason = phase2_exit_reason,
        macro_start_residual = macro_start_residual,
        macro_end_residual = macro_end_residual,
        best_macro_residual = best_macro_residual,
        best_macro_epoch = best_macro_epoch,
        best_pool_drifts = best_pool_drifts,
    )
end
