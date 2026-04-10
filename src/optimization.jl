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
    volumes::AbstractVector{VT},
) where {ET <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat}
    idx_history = leg.logger_prod.active_idx_history
    if leg.include_pv
        return compute_weights_and_ess(
            energies_current,
            leg.u_ref,
            idx_history,
            beta_val,
            volumes,
            leg.p0_energy_per_vol,
        )
    end
    return compute_weights_and_ess(
        energies_current,
        leg.u_ref,
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
) where {ET <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat}
    idx_history = leg.logger_prod.active_idx_history
    # Endpoint MBAR conditions on arbitrary λ targets, so the denominator must
    # use Molly's fixed AWH Gibbs log-weights g_ref = f_ref + logρ_ref.
    log_gibbs_weights = awh_log_gibbs_weights(leg.active_bias)

    if leg.include_pv
        grad_F_1, F_1 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            leg.coupled_state_idx,
            beta_val,
            log_gibbs_weights,
            volumes,
            leg.p0_energy_per_vol;
            compute_gradients=compute_gradients,
        )
        grad_F_0, F_0 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            leg.decoupled_state_idx,
            beta_val,
            log_gibbs_weights,
            volumes,
            leg.p0_energy_per_vol;
            compute_gradients=compute_gradients,
        )
    else
        grad_F_1, F_1 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            leg.coupled_state_idx,
            beta_val,
            log_gibbs_weights;
            compute_gradients=compute_gradients,
        )
        grad_F_0, F_0 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            leg.decoupled_state_idx,
            beta_val,
            log_gibbs_weights;
            compute_gradients=compute_gradients,
        )
    end

    return grad_F_1 .- grad_F_0, F_1 - F_0
end


Base.@kwdef struct OptimizationBlock
    name::String
    kind::Symbol
    global_indices::Vector{Int}
    trainable_indices::Vector{Int}
end


"""
    optimization_active_blocks(param_names, trainable_param_indices,
                               trainable_position_map, solute_param_indices,
                               solvent_param_indices, optimize_solvent)

Build the ordered list of parameter blocks visited by the inner optimization
loop. When `optimize_solvent=false`, the full trainable solute vector is
updated together.
"""
function optimization_active_blocks(
    param_names::Vector{String},
    trainable_param_indices::Vector{Int},
    trainable_position_map::Dict{Int, Int},
    solute_param_indices::Vector{Int},
    solvent_param_indices::Vector{Int},
    optimize_solvent::Bool,
)
    blocks = OptimizationBlock[]

    if optimize_solvent
        for (name, kind, global_indices_raw) in (
            ("Solute", :solute, solute_param_indices),
            ("Solvent", :solvent, solvent_param_indices),
        )
            global_indices = [idx for idx in global_indices_raw if haskey(trainable_position_map, idx)]
            isempty(global_indices) && continue
            trainable_indices = [trainable_position_map[idx] for idx in global_indices]
            push!(
                blocks,
                OptimizationBlock(
                    name=name,
                    kind=kind,
                    global_indices=global_indices,
                    trainable_indices=trainable_indices,
                ),
            )
        end
    else
        global_indices = [idx for idx in trainable_param_indices if haskey(trainable_position_map, idx)]
        trainable_indices = [trainable_position_map[idx] for idx in global_indices]
        isempty(global_indices) || push!(
            blocks,
            OptimizationBlock(
                name="Solute",
                kind=:solute,
                global_indices=global_indices,
                trainable_indices=trainable_indices,
            ),
        )
    end

    isempty(blocks) && throw(ArgumentError("No optimization blocks were constructed."))
    return blocks
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
    noise_tolerance = line_search_noise_tolerance(error_residual, tolerance_fraction)
    AT = promote_energy_analysis_type(error_residual, error_residual_prop, tolerance_fraction)
    return abs(AT(error_residual_prop)) <= abs(AT(error_residual)) + noise_tolerance
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


"""
    run_optimization_phase!(phi_active, theta_active, leg_artifacts, param_names,
                            trainable_param_names, trainable_param_indices,
                            trainable_position_map, solute_param_indices,
                            solvent_param_indices, theta_min, theta_max, phi_0,
                            beta_val, dG_std_corr, dG_exp, opt_cfg)

Perform the inner optimization loop for one macro epoch. The loop alternates
between evaluating the current parameterization, building a Fisher-preconditioned
update direction, and line-searching until either progress stalls or the ESS
constraint is violated.
"""
function run_optimization_phase!(
    phi_active::Vector{FT},
    theta_active::Vector{FT},
    leg_artifacts::Vector{LegArtifacts},
    param_names::Vector{String},
    trainable_param_names::Vector{String},
    trainable_param_indices::Vector{Int},
    trainable_position_map::Dict{Int, Int},
    solute_param_indices::Vector{Int},
    solvent_param_indices::Vector{Int},
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
    active_blocks = optimization_active_blocks(
        param_names,
        trainable_param_indices,
        trainable_position_map,
        solute_param_indices,
        solvent_param_indices,
        opt_cfg.optimize_solvent,
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
    solvent_theta_start = isempty(solvent_param_indices) ? FT[] : copy(theta_active[solvent_param_indices])

    inner_epoch = 1
    while inner_epoch <= opt_cfg.max_inner_epochs
        block = active_blocks[((inner_epoch - 1) % length(active_blocks)) + 1]
        active_global_indices = block.global_indices
        active_trainable_indices = block.trainable_indices
        block_name = block.name
        @info "  >> Optimization Epoch: Active Block = $block_name"

        if isempty(active_trainable_indices)
            @info "  [!] No parameters active for this block. Skipping."
            inner_epoch += 1
            continue
        end

        chain_rule_multiplier = get_chain_rule_multiplier(theta_active, theta_min, theta_max, opt_cfg.k_sigmoid)

        fim_joint = zeros(AT, length(trainable_param_names), length(trainable_param_names))
        grad_cycle = zeros(AT, length(trainable_param_names))
        dG_pred = AT(dG_std_corr)

        ess_current = Dict{Symbol, AT}()
        N_active = Dict{Symbol, AT}()
        leg_dG_current = Dict{Symbol, AT}()
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

        dL_dE = abs(error_residual) <= AT(opt_cfg.huber_delta) ? error_residual : AT(opt_cfg.huber_delta) * sign(error_residual)
        grad_loss = dL_dE .* grad_cycle

        grad_loss_active = grad_loss[active_trainable_indices]
        fim_active = fim_joint[active_trainable_indices, active_trainable_indices]

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
        if estimated_KL > AT(opt_cfg.kl_target)
            kl_scaling = sqrt(AT(opt_cfg.kl_target) / estimated_KL)
            update_direction_active = base_step_active * kl_scaling
        else
            kl_scaling = one(AT)
            update_direction_active = base_step_active
        end

        n_truncated = count(v -> v <= eigenvalue_tol, vals)
        fim_cond_raw = cond(fim_corr)

        max_phi_update = maximum(abs.(update_direction_active))
        max_allowed_phi_step = block.kind == :solute ? AT(opt_cfg.max_phi_step_solute) : AT(opt_cfg.max_phi_step_solvent)

        if max_phi_update > max_allowed_phi_step
            clip_scaling = max_allowed_phi_step / max_phi_update
            update_direction_active .*= clip_scaling
            @info "  [!] Step clipped by infinity-norm (Scaling: $(round(clip_scaling, digits=4)))"
        end

        update_direction_train = zeros(AT, length(trainable_param_indices))
        update_direction_train[active_trainable_indices] = update_direction_active
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

        # Backtracking line search enforces both residual improvement and a
        # minimum effective sample size under the proposed reweighting.
        for ls_iter in 1:7
            phi_prop .= phi_active .- alpha .* update_direction
            theta_prop .= map_phi_to_theta(phi_prop, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)

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
            ess_msg = join(
                [
                    "$(leg.name)=$(round(get(ess_prop, leg.name, zero(AT)), digits=1))"
                    for leg in leg_artifacts
                ],
                " | ",
            )
            @info "    LS Iter $ls_iter (α=$(alpha)): ESS[$ess_msg] | Res = $(round(error_residual_prop, digits=3))"

            if ess_ok && line_search_residual_acceptable(
                error_residual,
                error_residual_prop,
                AT(opt_cfg.line_search_noise_tolerance_fraction),
            )
                line_search_success = true
                phi_active .= phi_prop
                theta_active .= theta_prop
                accepted_residual = error_residual_prop
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
        end

        norm_grad_loss = norm(grad_loss_active)
        max_grad_loss = maximum(abs.(grad_loss_active))
        actual_max_phi_step = maximum(abs.(update_direction_active)) * alpha

        @info "--- Current Parameter State ---"
        for i in eachindex(param_names)
            @info "  $(param_names[i]): $(round(theta_active[i], digits=6))"
        end

        @info "--- Optimization Metrics (Epoch $inner_epoch - Block: $block_name) ---"
        @info "  Prediction:  ∆G_pred = $(round(dG_pred, digits=3)) kT | Target = $(round(dG_exp, digits=3)) kT"
        @info "  Error:       Residual = $(round(error_residual, digits=3)) | Huber dL/dE = $(round(dL_dE, digits=3))"
        @info "  Gradients:   Norm = $(round(norm_grad_loss, digits=5)) | Max = $(round(max_grad_loss, digits=5))"
        @info "  FIM (Corr):  Raw Cond Number = $(round(fim_cond_raw, digits=2)) | Truncated Eigs = $n_truncated / $(length(vals))"
        @info "  KL Bound:    Est. KL = $(round(estimated_KL, digits=4)) | Target = $(opt_cfg.kl_target) | Scaling = $(round(kl_scaling, digits=4))"
        @info "  Line Search: Converged α = $alpha"
        @info "  Actual Step: Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) (α=$alpha)"
        @info "  Params (σ,ϵ): Min = $(round(minimum(theta_active[active_global_indices]), digits=5)) | Max = $(round(maximum(theta_active[active_global_indices]), digits=5))"
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

    solvent_drift = zero(FT)
    if !opt_cfg.optimize_solvent && !isempty(solvent_param_indices)
        solvent_drift = maximum(abs.(theta_active[solvent_param_indices] .- solvent_theta_start))
        @info "Solvent Invariant: max |Δθ_solvent| = $(round(solvent_drift, digits=8))"
    end

    return (
        phase2_exit_reason = phase2_exit_reason,
        macro_start_residual = macro_start_residual,
        macro_end_residual = macro_end_residual,
        best_macro_residual = best_macro_residual,
        best_macro_epoch = best_macro_epoch,
        solvent_drift = solvent_drift,
    )
end
