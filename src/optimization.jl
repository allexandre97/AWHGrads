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


function line_search_residual_acceptable(
    error_residual::RT,
    error_residual_prop::PT,
    tolerance_fraction::TT,
    additional_improvement_requirement::ATX,
) where {RT <: AbstractFloat, PT <: AbstractFloat, TT <: AbstractFloat, ATX <: AbstractFloat}
    noise_tolerance = line_search_noise_tolerance(error_residual, tolerance_fraction)
    AT = promote_energy_analysis_type(error_residual, error_residual_prop, tolerance_fraction, additional_improvement_requirement)
    acceptance_threshold = max(
        zero(AT),
        abs(AT(error_residual)) + noise_tolerance - AT(additional_improvement_requirement),
    )
    return abs(AT(error_residual_prop)) <= acceptance_threshold
end

function optimization_analysis_type(
    leg_artifacts::Vector{LegArtifacts},
    ::Type{FT},
    beta_val::BT,
    dG_std_corr::DT,
    dG_exp::ET,
) where {FT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat, ET <: AbstractFloat}
    AT = promote_energy_analysis_type(FT, beta_val, dG_std_corr, dG_exp)
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
    return AT
end

Base.@kwdef struct OptimizationConfidenceSummary
    enabled::Bool = false
    scale::Any = 1.0
    endpoint_disagreement::Any = 0.0
    cycle_disagreement::Any = 0.0
    gradient_disagreement::Any = 0.0
    additional_residual_requirement::Any = 0.0
    eligible_legs::Int = 0
    skipped_legs::Vector{Symbol} = Symbol[]
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
    leg_artifacts::Vector{LegArtifacts},
    leg_energies_cache,
    leg_grads_phi,
    leg_volumes_cache,
    trainable_param_names::Vector{String},
    grad_cycle::AbstractVector{AT},
    beta_val::BT,
    dG_std_corr::DT,
    opt_cfg::OptimizationConfig,
    ::Type{AT},
) where {AT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat}
    if opt_cfg.optimization_confidence_mode != :split_half
        return OptimizationConfidenceSummary(
            enabled=false,
            scale=one(AT),
            endpoint_disagreement=zero(AT),
            cycle_disagreement=zero(AT),
            gradient_disagreement=zero(AT),
            additional_residual_requirement=zero(AT),
            eligible_legs=0,
            skipped_legs=Symbol[],
        )
    end

    endpoint_disagreements = AT[]
    cycle_half_1 = AT(dG_std_corr)
    cycle_half_2 = AT(dG_std_corr)
    grad_cycle_half_1 = zeros(AT, length(grad_cycle))
    grad_cycle_half_2 = zeros(AT, length(grad_cycle))
    eligible_legs = 0
    skipped_legs = Symbol[]

    for leg in leg_artifacts
        energies = leg_energies_cache[leg.name]
        frame_split = split_half_frame_indices(size(energies, 1), opt_cfg.optimization_confidence_min_frames)
        if isnothing(frame_split)
            push!(skipped_legs, leg.name)
            continue
        end

        eligible_legs += 1
        first_half, second_half = frame_split
        _, dG_half_1 = compute_leg_endpoint_state(
            leg,
            trainable_param_names,
            leg_grads_phi[leg.name],
            energies,
            beta_val,
            leg_volumes_cache[leg.name];
            compute_gradients=false,
            frame_indices=first_half,
        )
        _, dG_half_2 = compute_leg_endpoint_state(
            leg,
            trainable_param_names,
            leg_grads_phi[leg.name],
            energies,
            beta_val,
            leg_volumes_cache[leg.name];
            compute_gradients=false,
            frame_indices=second_half,
        )
        grad_half_1, _ = compute_leg_endpoint_state(
            leg,
            trainable_param_names,
            leg_grads_phi[leg.name],
            energies,
            beta_val,
            leg_volumes_cache[leg.name];
            compute_gradients=true,
            frame_indices=first_half,
        )
        grad_half_2, _ = compute_leg_endpoint_state(
            leg,
            trainable_param_names,
            leg_grads_phi[leg.name],
            energies,
            beta_val,
            leg_volumes_cache[leg.name];
            compute_gradients=true,
            frame_indices=second_half,
        )

        coeff = AT(leg.coefficient)
        cycle_half_1 += coeff * AT(dG_half_1)
        cycle_half_2 += coeff * AT(dG_half_2)
        grad_cycle_half_1 .+= coeff .* AT.(grad_half_1)
        grad_cycle_half_2 .+= coeff .* AT.(grad_half_2)
        push!(endpoint_disagreements, abs(AT(dG_half_1) - AT(dG_half_2)))
    end

    if eligible_legs == 0
        return OptimizationConfidenceSummary(
            enabled=false,
            scale=one(AT),
            endpoint_disagreement=zero(AT),
            cycle_disagreement=zero(AT),
            gradient_disagreement=zero(AT),
            additional_residual_requirement=zero(AT),
            eligible_legs=0,
            skipped_legs=skipped_legs,
        )
    end

    endpoint_disagreement = isempty(endpoint_disagreements) ? zero(AT) : maximum(endpoint_disagreements)
    cycle_disagreement = abs(cycle_half_1 - cycle_half_2)
    gradient_disagreement = norm(grad_cycle_half_1 - grad_cycle_half_2) / max(norm(grad_cycle), sqrt(eps(AT)))
    raw_penalty = max(endpoint_disagreement, cycle_disagreement, gradient_disagreement)
    min_scale = clamp(AT(opt_cfg.optimization_confidence_min_scale), zero(AT), one(AT))
    strength = max(zero(AT), AT(opt_cfg.optimization_confidence_scale_strength))
    scale = max(min_scale, one(AT) / (one(AT) + strength * raw_penalty))
    additional_requirement = max(zero(AT), AT(opt_cfg.optimization_confidence_residual_requirement_strength)) * cycle_disagreement
    return OptimizationConfidenceSummary(
        enabled=true,
        scale=scale,
        endpoint_disagreement=endpoint_disagreement,
        cycle_disagreement=cycle_disagreement,
        gradient_disagreement=gradient_disagreement,
        additional_residual_requirement=additional_requirement,
        eligible_legs=eligible_legs,
        skipped_legs=skipped_legs,
    )
end


"""
    run_optimization_phase!(phi_active, theta_active, leg_artifacts, param_names,
                            trainable_param_names, trainable_param_indices,
                            trainable_position_map, parameter_pools, theta_ref,
                            theta_min, theta_max, phi_0, beta_val, dG_std_corr,
                            dG_exp, opt_cfg)

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
    theta_ref::Vector{FT},
    theta_min::Vector{FT},
    theta_max::Vector{FT},
    phi_0::Vector{FT},
    beta_val::BT,
    dG_std_corr::DT,
    dG_exp::ETD,
    opt_cfg::OptimizationConfig,
) where {FT <: AbstractFloat, BT <: AbstractFloat, DT <: AbstractFloat, ETD <: AbstractFloat}
    if isempty(leg_artifacts)
        throw(ArgumentError("run_optimization_phase! requires at least one leg artifact."))
    end
    if isempty(trainable_param_indices)
        throw(ArgumentError("run_optimization_phase! received no trainable parameters."))
    end
    if opt_cfg.max_inner_epochs <= 0
        throw(ArgumentError("OptimizationConfig.max_inner_epochs must be positive."))
    end

    AT = optimization_analysis_type(leg_artifacts, FT, beta_val, dG_std_corr, dG_exp)

    theoretical_ess_ratio = exp(AT(-2.0) * AT(opt_cfg.kl_target))
    ess_threshold_ratio = AT(opt_cfg.ess_threshold_scale) * theoretical_ess_ratio
    active_pools = optimization_active_pools(
        trainable_position_map,
        parameter_pools,
    )

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

        chain_rule_multiplier = get_chain_rule_multiplier(theta_active, theta_min, theta_max, opt_cfg.k_sigmoid)

        fim_joint = zeros(AT, length(trainable_param_names), length(trainable_param_names))
        grad_cycle = zeros(AT, length(trainable_param_names))
        dG_pred = AT(dG_std_corr)

        ess_current = Dict{Symbol, AT}()
        N_active = Dict{Symbol, AT}()
        leg_dG_current = Dict{Symbol, AT}()
        leg_energies_cache = Dict{Symbol, Any}()
        leg_grads_phi = Dict{Symbol, Dict{String, Matrix{FT}}}()
        leg_volumes_cache = Dict{Symbol, Vector{AT}}()

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

            grad_dG_leg, dG_leg = compute_leg_endpoint_state(
                leg,
                trainable_param_names,
                grads_eval_phi,
                u_eval,
                beta_val,
                volumes;
                compute_gradients=true,
            )

            coeff = AT(leg.coefficient)
            grad_cycle .+= coeff .* grad_dG_leg
            dG_pred += coeff * dG_leg

            leg_dG_current[leg.name] = dG_leg
            leg_energies_cache[leg.name] = u_eval
            leg_grads_phi[leg.name] = grads_eval_phi
            leg_volumes_cache[leg.name] = volumes
        end

        if ess_threshold_broken
            println("  [!] ESS threshold broken during state evaluation. Exiting Phase 2.")
            phase2_exit_reason = :ess_threshold
            break
        end

        error_residual = dG_pred - dG_exp
        if isnan(macro_start_residual)
            macro_start_residual = error_residual
        end

        confidence_summary = compute_optimization_confidence_summary(
            leg_artifacts,
            leg_energies_cache,
            leg_grads_phi,
            leg_volumes_cache,
            trainable_param_names,
            grad_cycle,
            beta_val,
            dG_std_corr,
            opt_cfg,
            AT,
        )
        effective_kl_target = AT(opt_cfg.kl_target) * confidence_summary.scale

        dL_dE = abs(error_residual) <= AT(opt_cfg.huber_delta) ? error_residual : AT(opt_cfg.huber_delta) * sign(error_residual)
        grad_loss = dL_dE .* grad_cycle
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
        accepted_residual = error_residual
        ess_prop = Dict{Symbol, AT}()
        accepted_pool_drifts = Dict{Symbol, NamedTuple}()

        # Backtracking line search enforces both residual improvement and a
        # minimum effective sample size under the proposed reweighting, together
        # with optional per-pool drift caps relative to the reference model.
        for ls_iter in 1:7
            phi_prop .= phi_active .- alpha .* update_direction
            theta_prop .= map_phi_to_theta(phi_prop, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)
            drift_ok, pool_drifts_prop = pool_drift_metrics(theta_prop, theta_ref, active_pools)

            dG_pred_prop = AT(dG_std_corr)
            ess_ok = true

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

                _, dG_leg_prop = compute_leg_endpoint_state(
                    leg,
                    trainable_param_names,
                    leg_grads_phi[leg.name],
                    u_prop,
                    beta_val,
                    volumes;
                    compute_gradients=false,
                )
                dG_pred_prop += AT(leg.coefficient) * dG_leg_prop
            end

            error_residual_prop = dG_pred_prop - dG_exp
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
            @info "    LS Iter $ls_iter (α=$(alpha)): ESS[$ess_msg] | Drift[$drift_msg] | Res = $(round(error_residual_prop, digits=3))"

            if ess_ok && drift_ok && line_search_residual_acceptable(
                error_residual,
                error_residual_prop,
                AT(opt_cfg.line_search_noise_tolerance_fraction),
                confidence_summary.additional_residual_requirement,
            )
                line_search_success = true
                phi_active .= phi_prop
                theta_active .= theta_prop
                accepted_residual = error_residual_prop
                accepted_pool_drifts = pool_drifts_prop
                @info "    -> Line search converged."
                break
            else
                alpha *= AT(0.5)
            end
        end

        if alpha <= AT(opt_cfg.tiny_alpha_cutoff)
            tiny_alpha_hits += 1
        else
            tiny_alpha_hits = 0
        end

        macro_end_residual = accepted_residual
        if abs(accepted_residual) < best_macro_abs_residual
            best_macro_abs_residual = abs(accepted_residual)
            best_macro_residual = accepted_residual
            best_macro_epoch = inner_epoch
            phi_best_macro .= phi_active
            theta_best_macro .= theta_active
            best_pool_drifts = accepted_pool_drifts
        end

        norm_grad_loss = norm(grad_loss_active)
        max_grad_loss = maximum(abs.(grad_loss_active))
        actual_max_phi_step = maximum(abs.(update_direction_train)) * alpha

        @info "--- Current Parameter State ---"
        for i in eachindex(param_names)
            @info "  $(param_names[i]): $(round(theta_active[i], digits=6))"
        end

        @info "--- Optimization Metrics (Epoch $inner_epoch - Joint Pools) ---"
        @info "  Prediction:  ∆G_pred = $(round(dG_pred, digits=3)) kT | Target = $(round(dG_exp, digits=3)) kT"
        @info "  Error:       Residual = $(round(error_residual, digits=3)) | Huber dL/dE = $(round(dL_dE, digits=3))"
        @info "  Accepted:    Residual = $(round(accepted_residual, digits=3)) | Extra req = $(round(confidence_summary.additional_residual_requirement, digits=4))"
        @info "  Gradients:   Norm = $(round(norm_grad_loss, digits=5)) | Max = $(round(max_grad_loss, digits=5))"
        @info "  FIM (Corr):  Raw Cond Number = $(round(fim_cond_raw, digits=2)) | Truncated Eigs = $n_truncated / $(length(vals))"
        @info "  Trust Reg.:  Confidence = $(round(confidence_summary.scale, digits=4)) | KL target = $(round(effective_kl_target, digits=4))"
        @info "  Confidence:  endpoint_ΔG = $(round(confidence_summary.endpoint_disagreement, digits=4)) | cycle = $(round(confidence_summary.cycle_disagreement, digits=4)) | gradient = $(round(confidence_summary.gradient_disagreement, digits=4)) | eligible_legs = $(confidence_summary.eligible_legs)"
        if !isempty(confidence_summary.skipped_legs)
            @info "  Confidence:  skipped_legs = $(join(String.(confidence_summary.skipped_legs), ","))"
        end
        @info "  KL Bound:    Est. KL = $(round(estimated_KL, digits=4)) | Target = $(round(effective_kl_target, digits=4)) | Scaling = $(round(kl_scaling, digits=4))"
        @info "  Line Search: Converged α = $alpha"
        @info "  Actual Step: Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) (α=$alpha)"
        @info "  Params (σ,ϵ): Min = $(round(minimum(theta_active[trainable_param_indices]), digits=5)) | Max = $(round(maximum(theta_active[trainable_param_indices]), digits=5))"
        for pool in active_pools
            clip_stat = pool_clip_stats[pool.name]
            drift_stat = get(accepted_pool_drifts, pool.name, (sigma_drift=zero(AT), epsilon_drift=zero(AT), sigma_ready=true, epsilon_ready=true))
            @info "  Pool $(pool.name): max_ϕ=$(round(clip_stat.max_phi_update, digits=6)) | clip=$(round(clip_stat.clip_scaling, digits=4)) | σ_drift=$(round(drift_stat.sigma_drift, digits=6)) | ϵ_drift=$(round(drift_stat.epsilon_drift, digits=6))"
        end
        for leg in leg_artifacts
            @info "  Leg $(leg.name): coeff=$(round(leg.coefficient, digits=3)) | ΔG=$(round(leg_dG_current[leg.name], digits=3)) kT | ESS=$(round(ess_current[leg.name], digits=1)) / $(round(ess_thresholds[leg.name], digits=1)) | N_active=$(round(N_active[leg.name], digits=1))"
        end
        println("---------------------------------------------------\n")

        if tiny_alpha_hits >= opt_cfg.max_tiny_alpha_hits
            @info "  [!] Repeated tiny line-search α detected ($tiny_alpha_hits consecutive epochs with α <= $(opt_cfg.tiny_alpha_cutoff)). Triggering Phase 3 resimulation."
            phase2_exit_reason = :tiny_alpha
            break
        end

        if !line_search_success || actual_max_phi_step < AT(opt_cfg.min_phi_step)
            @info "  [!] Line search failed or step vanished. Triggering Phase 3 resimulation."
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
