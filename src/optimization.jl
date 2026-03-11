function leg_volumes(leg::LegArtifacts, ::Type{FT}) where {FT <: AbstractFloat}
    if !leg.include_pv
        return FT[]
    end
    return FT.(ustrip.(leg.logger_prod.volume_history))
end


function compute_leg_weights_and_ess(
    leg::LegArtifacts,
    energies_current::Matrix{FT},
    beta_val::FT,
    volumes::Vector{FT},
) where {FT <: AbstractFloat}
    idx_history = leg.logger_prod.active_idx_history
    if leg.include_pv
        return compute_weights_and_ess(
            energies_current,
            leg.u_ref,
            idx_history,
            beta_val,
            volumes,
            FT(leg.p0_energy_per_vol),
        )
    end
    return compute_weights_and_ess(
        energies_current,
        leg.u_ref,
        idx_history,
        beta_val,
    )
end


function compute_leg_endpoint_state(
    leg::LegArtifacts,
    trainable_param_names::Vector{String},
    gradients_phi::Dict{String, Matrix{FT}},
    energies_current::Matrix{FT},
    beta_val::FT,
    volumes::Vector{FT};
    compute_gradients::Bool=true,
) where {FT <: AbstractFloat}
    idx_history = leg.logger_prod.active_idx_history
    awh_bias = leg.active_bias.f .+ leg.active_bias.log_rho

    if leg.include_pv
        grad_F_1, F_1 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            1,
            beta_val,
            awh_bias,
            volumes,
            FT(leg.p0_energy_per_vol);
            compute_gradients=compute_gradients,
        )
        grad_F_0, F_0 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            num_lambda_states,
            beta_val,
            awh_bias,
            volumes,
            FT(leg.p0_energy_per_vol);
            compute_gradients=compute_gradients,
        )
    else
        grad_F_1, F_1 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            1,
            beta_val,
            awh_bias;
            compute_gradients=compute_gradients,
        )
        grad_F_0, F_0 = compute_global_endpoint_gradients(
            trainable_param_names,
            gradients_phi,
            energies_current,
            leg.u_ref,
            idx_history,
            num_lambda_states,
            beta_val,
            awh_bias;
            compute_gradients=compute_gradients,
        )
    end

    return grad_F_1 .- grad_F_0, F_1 - F_0
end


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
    beta_val::FT,
    dG_std_corr::FT,
    dG_exp::FT,
    opt_cfg::OptimizationConfig,
) where {FT <: AbstractFloat}
    if isempty(leg_artifacts)
        throw(ArgumentError("run_optimization_phase! requires at least one leg artifact."))
    end
    if isempty(trainable_param_indices)
        throw(ArgumentError("run_optimization_phase! received no trainable parameters."))
    end

    theoretical_ess_ratio = exp(FT(-2.0) * opt_cfg.kl_target)
    ess_threshold_ratio = opt_cfg.ess_threshold_scale * theoretical_ess_ratio

    ess_thresholds = Dict{Symbol, FT}()
    N_base = Dict{Symbol, FT}()
    for leg in leg_artifacts
        M_leg = FT(length(leg.logger_prod.active_idx_history))
        if M_leg <= zero(FT)
            throw(ArgumentError("Leg $(leg.name) has no production frames."))
        end
        ess_thresholds[leg.name] = M_leg * ess_threshold_ratio
        N_base[leg.name] = FT(leg.awh_prod.initial_sampl_n + leg.awh_prod.state.N_eff)
    end

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

        chain_rule_multiplier = get_chain_rule_multiplier(theta_active, theta_min, theta_max, opt_cfg.k_sigmoid)

        fim_joint = zeros(FT, length(trainable_param_names), length(trainable_param_names))
        grad_cycle = zeros(FT, length(trainable_param_names))
        dG_pred = dG_std_corr

        ess_current = Dict{Symbol, FT}()
        N_active = Dict{Symbol, FT}()
        leg_dG_current = Dict{Symbol, FT}()
        leg_grads_phi = Dict{Symbol, Dict{String, Matrix{FT}}}()
        leg_volumes_cache = Dict{Symbol, Vector{FT}}()

        ess_threshold_broken = false

        for leg in leg_artifacts
            u_eval, grads_eval_theta = evaluate_ensemble(
                leg.logger_prod,
                leg.neighbors,
                leg.awh_prod,
                leg.sys_base,
                theta_active,
                param_names,
                leg.idxs...;
                compute_gradients=true,
            )

            grads_eval_phi = Dict{String, Matrix{FT}}()
            for (i, p_key) in enumerate(param_names)
                grads_eval_phi[p_key] = grads_eval_theta[p_key] .* chain_rule_multiplier[i]
            end

            volumes = leg_volumes(leg, FT)
            _, ess_leg = compute_leg_weights_and_ess(leg, u_eval, beta_val, volumes)
            ess_current[leg.name] = ess_leg

            M_leg = FT(length(leg.logger_prod.active_idx_history))
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

            coeff = FT(leg.coefficient)
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

        dL_dE = abs(error_residual) <= opt_cfg.huber_delta ? error_residual : opt_cfg.huber_delta * sign(error_residual)
        grad_loss = dL_dE .* grad_cycle

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
        inv_vals = [v > eigenvalue_tol ? FT(1.0) / v : zero(FT) for v in vals]
        fim_corr_inv = vecs * Diagonal(inv_vals) * transpose(vecs)

        base_step_scaled = fim_corr_inv * grad_loss_scaled
        base_step_active = D_vec .* base_step_scaled

        estimated_KL = FT(0.5) * dot(base_step_active, fim_active * base_step_active)
        if estimated_KL > opt_cfg.kl_target
            kl_scaling = sqrt(opt_cfg.kl_target / estimated_KL)
            update_direction_active = base_step_active * kl_scaling
        else
            kl_scaling = FT(1.0)
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
        accepted_residual = error_residual
        ess_prop = Dict{Symbol, FT}()

        for ls_iter in 1:7
            phi_prop .= phi_active .- alpha .* update_direction
            theta_prop .= map_phi_to_theta(phi_prop, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)

            dG_pred_prop = dG_std_corr
            ess_ok = true

            for leg in leg_artifacts
                u_prop, _ = evaluate_ensemble(
                    leg.logger_prod,
                    leg.neighbors,
                    leg.awh_prod,
                    leg.sys_base,
                    theta_prop,
                    param_names,
                    leg.idxs...;
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
                dG_pred_prop += FT(leg.coefficient) * dG_leg_prop
            end

            error_residual_prop = dG_pred_prop - dG_exp
            ess_msg = join(
                [
                    "$(leg.name)=$(round(get(ess_prop, leg.name, zero(FT)), digits=1))"
                    for leg in leg_artifacts
                ],
                " | ",
            )
            @info "    LS Iter $ls_iter (α=$(alpha)): ESS[$ess_msg] | Res = $(round(error_residual_prop, digits=3))"

            noise_tolerance = FT(0.05) * abs(error_residual)
            if ess_ok && abs(error_residual_prop) <= abs(error_residual) + noise_tolerance
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
