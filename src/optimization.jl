function run_optimization_phase!(
    phi_active::Vector{FT},
    theta_active::Vector{FT},
    active_bias_solv,
    active_bias_vac,
    logger_solv_prod,
    logger_vac_prod,
    nbrs_solv,
    nbrs_vac,
    awh_solv_prod,
    awh_vac_prod,
    sys_solv,
    sys_vac,
    u_solv_ref::Matrix{FT},
    u_vac_ref::Matrix{FT},
    param_names::Vector{String},
    trainable_param_names::Vector{String},
    trainable_param_indices::Vector{Int},
    trainable_position_map::Dict{Int, Int},
    solute_param_indices::Vector{Int},
    solvent_param_indices::Vector{Int},
    idxs_solv,
    idxs_vac,
    theta_min::Vector{FT},
    theta_max::Vector{FT},
    phi_0::Vector{FT},
    beta_val::FT,
    dG_std_corr::FT,
    dG_exp::FT,
    P0_energy_per_vol::FT,
    opt_cfg::OptimizationConfig,
) where {FT <: AbstractFloat}
    M_solv_FT = FT(length(logger_solv_prod.active_idx_history))
    M_vac_FT = FT(length(logger_vac_prod.active_idx_history))

    theoretical_ess_ratio = exp(FT(-2.0) * opt_cfg.kl_target)
    ess_threshold_ratio = opt_cfg.ess_threshold_scale * theoretical_ess_ratio
    ess_threshold_solv = M_solv_FT * ess_threshold_ratio
    ess_threshold_vac = M_vac_FT * ess_threshold_ratio

    N_base_solv = FT(awh_solv_prod.initial_sampl_n + awh_solv_prod.state.N_eff)
    N_base_vac = FT(awh_vac_prod.initial_sampl_n + awh_vac_prod.state.N_eff)

    tiny_alpha_hits = 0
    phase2_exit_reason = :running
    macro_start_residual = FT(NaN)
    macro_end_residual = FT(NaN)
    best_macro_abs_residual = FT(Inf)
    best_macro_residual = FT(NaN)
    best_macro_epoch = 0
    phi_best_macro = copy(phi_active)
    theta_best_macro = copy(theta_active)
    solvent_theta_start = isempty(solvent_param_indices) ? FT[] : copy(theta_active[solvent_param_indices])

    inner_epoch = 1
    while true
        if opt_cfg.optimize_solvent
            active_global_indices = (inner_epoch % 2 != 0) ? solute_param_indices : solvent_param_indices
            block_name = (inner_epoch % 2 != 0) ? "Solute" : "Solvent"
        else
            active_global_indices = trainable_param_indices
            block_name = "Solute"
        end

        active_trainable_indices = Int[]
        for idx_global in active_global_indices
            if haskey(trainable_position_map, idx_global)
                push!(active_trainable_indices, trainable_position_map[idx_global])
            end
        end
        @info "  >> Alternating Optimization: Active Block = $block_name"

        if isempty(active_trainable_indices)
            @info "  [!] No parameters active for this block. Skipping."
            inner_epoch += 1
            continue
        end

        u_solv_eval, grads_solv_eval_theta = evaluate_ensemble(
            logger_solv_prod, nbrs_solv, awh_solv_prod, sys_solv,
            theta_active, param_names, idxs_solv...; compute_gradients=true,
        )
        u_vac_eval, grads_vac_eval_theta = evaluate_ensemble(
            logger_vac_prod, nbrs_vac, awh_vac_prod, sys_vac,
            theta_active, param_names, idxs_vac...; compute_gradients=true,
        )

        grads_solv_eval_phi = Dict{String, Matrix{FT}}()
        grads_vac_eval_phi = Dict{String, Matrix{FT}}()

        chain_rule_multiplier = get_chain_rule_multiplier(theta_active, theta_min, theta_max, opt_cfg.k_sigmoid)
        for (i, p_key) in enumerate(param_names)
            grads_solv_eval_phi[p_key] = grads_solv_eval_theta[p_key] .* chain_rule_multiplier[i]
            grads_vac_eval_phi[p_key] = grads_vac_eval_theta[p_key] .* chain_rule_multiplier[i]
        end

        volumes_solv = FT.(ustrip.(logger_solv_prod.volume_history))

        w_norm_solv, ess_solv = compute_weights_and_ess(
            u_solv_eval, u_solv_ref, logger_solv_prod.active_idx_history, beta_val, volumes_solv, P0_energy_per_vol,
        )
        w_norm_vac, ess_vac = compute_weights_and_ess(
            u_vac_eval, u_vac_ref, logger_vac_prod.active_idx_history, beta_val,
        )

        N_active_solv = N_base_solv * (ess_solv / M_solv_FT)
        N_active_vac = N_base_vac * (ess_vac / M_vac_FT)

        if ess_solv < ess_threshold_solv || ess_vac < ess_threshold_vac
            println("  [!] ESS threshold broken during state evaluation. Exiting Phase 2.")
            phase2_exit_reason = :ess_threshold
            break
        end

        _, fim_solv = compute_empirical_gradients_and_fim(
            trainable_param_names, grads_solv_eval_phi, w_norm_solv, logger_solv_prod.active_idx_history, beta_val,
        )
        _, fim_vac = compute_empirical_gradients_and_fim(
            trainable_param_names, grads_vac_eval_phi, w_norm_vac, logger_vac_prod.active_idx_history, beta_val,
        )

        bias_solv = active_bias_solv.f .+ active_bias_solv.log_rho
        bias_vac = active_bias_vac.f .+ active_bias_vac.log_rho

        grad_F_solv_1, F_solv_1_current = compute_global_endpoint_gradients(
            trainable_param_names, grads_solv_eval_phi, u_solv_eval, u_solv_ref, logger_solv_prod.active_idx_history,
            1, beta_val, bias_solv, volumes_solv, P0_energy_per_vol,
        )
        grad_F_solv_0, F_solv_0_current = compute_global_endpoint_gradients(
            trainable_param_names, grads_solv_eval_phi, u_solv_eval, u_solv_ref, logger_solv_prod.active_idx_history,
            num_lambda_states, beta_val, bias_solv, volumes_solv, P0_energy_per_vol,
        )
        grad_F_vac_1, F_vac_1_current = compute_global_endpoint_gradients(
            trainable_param_names, grads_vac_eval_phi, u_vac_eval, u_vac_ref, logger_vac_prod.active_idx_history,
            1, beta_val, bias_vac,
        )
        grad_F_vac_0, F_vac_0_current = compute_global_endpoint_gradients(
            trainable_param_names, grads_vac_eval_phi, u_vac_eval, u_vac_ref, logger_vac_prod.active_idx_history,
            num_lambda_states, beta_val, bias_vac,
        )

        grad_dG_solv = grad_F_solv_1 .- grad_F_solv_0
        grad_dG_vac = grad_F_vac_1 .- grad_F_vac_0

        dG_solv_current = F_solv_1_current - F_solv_0_current
        dG_vac_current = F_vac_1_current - F_vac_0_current

        dG_pred = dG_solv_current - dG_vac_current + dG_std_corr
        error_residual = dG_pred - dG_exp
        if isnan(macro_start_residual)
            macro_start_residual = error_residual
        end

        dL_dE = abs(error_residual) <= opt_cfg.huber_delta ? error_residual : opt_cfg.huber_delta * sign(error_residual)
        grad_loss = dL_dE .* (grad_dG_solv .- grad_dG_vac)
        fim_joint = fim_solv .+ fim_vac

        grad_loss_active = grad_loss[active_trainable_indices]
        fim_active = fim_joint[active_trainable_indices, active_trainable_indices]

        fim_diag = diag(fim_active)
        variance_threshold = maximum(fim_diag) * FT(1e-5)
        D_vec = [d > variance_threshold ? FT(1.0) / sqrt(d) : zero(FT) for d in fim_diag]
        D_mat = Diagonal(D_vec)

        fim_corr = D_mat * fim_active * D_mat
        grad_loss_scaled = D_vec .* grad_loss_active

        decomp = eigen(Symmetric(fim_corr))
        vals, vecs = decomp.values, decomp.vectors

        eigenvalue_tol = maximum(vals) * opt_cfg.eigenvalue_tol_scale
        inv_vals = [v > eigenvalue_tol ? 1.0 / v : zero(FT) for v in vals]
        fim_corr_inv = vecs * Diagonal(inv_vals) * transpose(vecs)

        base_step_scaled = fim_corr_inv * grad_loss_scaled
        base_step_active = D_vec .* base_step_scaled

        estimated_KL = 0.5 * dot(base_step_active, fim_active * base_step_active)
        if estimated_KL > opt_cfg.kl_target
            kl_scaling = sqrt(opt_cfg.kl_target / estimated_KL)
            update_direction_active = base_step_active * kl_scaling
        else
            kl_scaling = 1.0
            update_direction_active = base_step_active
        end

        n_truncated = count(v -> v <= eigenvalue_tol, vals)
        fim_cond_raw = cond(fim_corr)

        max_phi_update = maximum(abs.(update_direction_active))
        max_allowed_phi_step = block_name == "Solute" ? opt_cfg.max_phi_step_solute : opt_cfg.max_phi_step_solvent

        if max_phi_update > max_allowed_phi_step
            clip_scaling = max_allowed_phi_step / max_phi_update
            update_direction_active .*= clip_scaling
            @info "  [!] Step clipped by infinity-norm (Scaling: $(round(clip_scaling, digits=4)))"
        end

        update_direction_train = zeros(FT, length(trainable_param_indices))
        update_direction_train[active_trainable_indices] = update_direction_active
        update_direction = zeros(FT, length(param_names))
        for (i_local, i_global) in enumerate(trainable_param_indices)
            update_direction[i_global] = update_direction_train[i_local]
        end

        alpha = FT(1.0)
        phi_prop = copy(phi_active)
        theta_prop = copy(theta_active)
        line_search_success = false

        ess_solv_prop = zero(FT)
        ess_vac_prop = zero(FT)
        accepted_residual = error_residual

        for ls_iter in 1:7
            phi_prop .= phi_active .- alpha .* update_direction
            theta_prop .= map_phi_to_theta(phi_prop, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)

            u_solv_prop, _ = evaluate_ensemble(
                logger_solv_prod, nbrs_solv, awh_solv_prod, sys_solv,
                theta_prop, param_names, idxs_solv...; compute_gradients=false,
            )
            u_vac_prop, _ = evaluate_ensemble(
                logger_vac_prod, nbrs_vac, awh_vac_prod, sys_vac,
                theta_prop, param_names, idxs_vac...; compute_gradients=false,
            )

            _, ess_solv_prop = compute_weights_and_ess(
                u_solv_prop, u_solv_ref, logger_solv_prod.active_idx_history, beta_val, volumes_solv, P0_energy_per_vol,
            )
            _, ess_vac_prop = compute_weights_and_ess(
                u_vac_prop, u_vac_ref, logger_vac_prod.active_idx_history, beta_val,
            )

            _, F_solv_1_prop = compute_global_endpoint_gradients(
                trainable_param_names, grads_solv_eval_phi, u_solv_prop, u_solv_ref, logger_solv_prod.active_idx_history,
                1, beta_val, bias_solv, volumes_solv, P0_energy_per_vol; compute_gradients=false,
            )
            _, F_solv_0_prop = compute_global_endpoint_gradients(
                trainable_param_names, grads_solv_eval_phi, u_solv_prop, u_solv_ref, logger_solv_prod.active_idx_history,
                num_lambda_states, beta_val, bias_solv, volumes_solv, P0_energy_per_vol; compute_gradients=false,
            )
            _, F_vac_1_prop = compute_global_endpoint_gradients(
                trainable_param_names, grads_vac_eval_phi, u_vac_prop, u_vac_ref, logger_vac_prod.active_idx_history,
                1, beta_val, bias_vac; compute_gradients=false,
            )
            _, F_vac_0_prop = compute_global_endpoint_gradients(
                trainable_param_names, grads_vac_eval_phi, u_vac_prop, u_vac_ref, logger_vac_prod.active_idx_history,
                num_lambda_states, beta_val, bias_vac; compute_gradients=false,
            )

            dG_pred_prop = (F_solv_1_prop - F_solv_0_prop) - (F_vac_1_prop - F_vac_0_prop) + dG_std_corr
            error_residual_prop = dG_pred_prop - dG_exp

            @info "    LS Iter $ls_iter (α=$(alpha)): Solv ESS = $(round(ess_solv_prop, digits=1)) | Vac ESS = $(round(ess_vac_prop, digits=1)) | Res = $(round(error_residual_prop, digits=3))"

            noise_tolerance = FT(0.05) * abs(error_residual)
            if ess_solv_prop >= ess_threshold_solv && ess_vac_prop >= ess_threshold_vac && abs(error_residual_prop) <= abs(error_residual) + noise_tolerance
                line_search_success = true
                phi_active .= phi_prop
                theta_active .= theta_prop
                accepted_residual = error_residual_prop
                @info "    -> Line search converged."
                break
            else
                alpha *= FT(0.5)
            end
        end

        if alpha <= opt_cfg.tiny_alpha_cutoff
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
        for i in 1:length(param_names)
            @info "  $(param_names[i]): $(round(theta_active[i], digits=6))"
        end

        @info "--- Optimization Metrics (Epoch $inner_epoch - Block: $block_name) ---"
        @info "  Prediction:  ∆G_pred = $(round(dG_pred, digits=3)) kT | Target = $(round(dG_exp, digits=3)) kT"
        @info "  Error:       Residual = $(round(error_residual, digits=3)) | Huber dL/dE = $(round(dL_dE, digits=3))"
        @info "  Gradients:   Norm = $(round(norm_grad_loss, digits=5)) | Max = $(round(max_grad_loss, digits=5))"
        @info "  FIM (Corr):  Raw Cond Number = $(round(fim_cond_raw, digits=2)) | Truncated Eigs = $n_truncated / $(length(vals))"
        @info "  Effective Samples: Solv = $(round(N_active_solv, digits=1)) | Vac = $(round(N_active_vac, digits=1))"
        @info "  KL Bound:    Est. KL = $(round(estimated_KL, digits=4)) | Target = $(opt_cfg.kl_target) | Scaling = $(round(kl_scaling, digits=4))"
        @info "  Line Search: Converged α = $alpha | Final Solv ESS = $(round(ess_solv_prop, digits=1)) | Vac ESS = $(round(ess_vac_prop, digits=1))"
        @info "  Actual Step: Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) (α=$alpha)"
        @info "  Params (σ,ϵ):Min = $(round(minimum(theta_active[active_global_indices]), digits=5)) | Max = $(round(maximum(theta_active[active_global_indices]), digits=5))"
        println("---------------------------------------------------\n")

        if tiny_alpha_hits >= opt_cfg.max_tiny_alpha_hits
            @info "  [!] Repeated tiny line-search α detected ($tiny_alpha_hits consecutive epochs with α <= $(opt_cfg.tiny_alpha_cutoff)). Triggering Phase 3 resimulation."
            phase2_exit_reason = :tiny_alpha
            break
        end

        if !line_search_success || actual_max_phi_step < opt_cfg.min_phi_step
            @info "  [!] Line search failed or step vanished. Triggering Phase 3 resimulation."
            phase2_exit_reason = :step_vanish
            break
        end

        inner_epoch += 1
    end

    if (phase2_exit_reason == :step_vanish || phase2_exit_reason == :tiny_alpha) && best_macro_epoch > 0
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
