function run_pipeline(; sim_cfg::SimulationConfig=default_simulation_config(), opt_cfg::OptimizationConfig=default_optimization_config())
    apply_simulation_config!(sim_cfg)
    FT = sim_cfg.FT
    runtime = RuntimeState()

    # Evaluate Standard State Correction (Gas to 1M Liquid)
    V_gas = ustrip(u"nm^3", Unitful.k * T0 / P0)
    V_std = ustrip(u"nm^3", 1.0u"L" / (1.0u"mol" * Unitful.Na))
    dG_std_corr = FT(log(V_gas / V_std))

awh_budget_time = sim_cfg.awh_budget_time
awh_block_time = sim_cfg.awh_block_time
awh_probe_time_solv = sim_cfg.awh_probe_time_solv
awh_probe_time_vac = sim_cfg.awh_probe_time_vac
md_time_production = sim_cfg.md_time_production

md_steps_budget = Int(floor(uconvert(unit(Δt), awh_budget_time) / Δt))
md_steps_block = max(1, Int(round(uconvert(unit(Δt), awh_block_time) / Δt)))
md_steps_probe_solv = max(1, Int(round(uconvert(unit(Δt), awh_probe_time_solv) / Δt)))
md_steps_probe_vac = max(1, Int(round(uconvert(unit(Δt), awh_probe_time_vac) / Δt)))
md_steps_prod = Int(floor(uconvert(unit(Δt), md_time_production) / Δt))
solute_idx = sim_cfg.solute_idx

production_log_interval = sim_cfg.production_log_interval
awh_convergence_tol = opt_cfg.awh_convergence_tol
rewarm_fraction = opt_cfg.rewarm_fraction
md_steps_rewarm = max(1, Int(round(md_steps_budget * rewarm_fraction)))
md_steps_rewarm = min(md_steps_rewarm, md_steps_budget)

T_coord = typeof(FT(1.0)u"nm")
T_vol = typeof(FT(1.0)u"nm^3")
T_en = typeof(FT(1.0)u"kJ * mol^-1")

max_macro_epochs = opt_cfg.max_macro_epochs
restart_rmsd_tol_nm = opt_cfg.restart_rmsd_tol_nm
awh_min_linear_neff = opt_cfg.awh_min_linear_neff
awh_split_tol_kT = opt_cfg.awh_split_tol_kT
awh_parity_tol_kT = opt_cfg.awh_parity_tol_kT
awh_tail_lag = opt_cfg.awh_tail_lag
awh_min_round_trips = opt_cfg.awh_min_round_trips
awh_endpoint_min_fraction = opt_cfg.awh_endpoint_min_fraction
awh_stageA_stable_blocks = opt_cfg.awh_stageA_stable_blocks

runtime.active_bias_solv = nothing
runtime.active_bias_vac = nothing
runtime.restart_cache_solv = nothing
runtime.restart_cache_vac = nothing

# ==============================================================================
# --- INITIALIZE PARAMETER MAPPINGS (RUNS ONCE) ---
# ==============================================================================
theta_ref    = Vector{FT}()
phi_active   = Vector{FT}()
theta_active = Vector{FT}()
param_names  = Vector{String}()  

sys_dummy = System(sim_cfg.pdb_solv, ff; array_type=AT, nonbonded_method=:none)
processed_atom_types = Dict{String, Int}()  
atoms_cpu = Molly.from_device(sys_dummy.atoms)  

solute_param_indices = Int[]
solvent_param_indices = Int[]

for idx in eachindex(atoms_cpu)  
    atom  = atoms_cpu[idx]  
    atype = String(sys_dummy.atoms_data[idx].atom_type)  
    
    if !haskey(processed_atom_types, atype)  
        sigma_raw = FT(ustrip(atom.σ))
        if sigma_raw <= FT(1e-6) || sigma_raw == one(FT)
            sigma_raw = FT(0.15)
        end
        
        push!(theta_ref, sigma_raw)
        push!(param_names, "atom_$(atype)_σ")  
        sigma_idx = length(theta_ref)  
        
        epsilon_raw = FT(ustrip(atom.ϵ))
        if epsilon_raw <= FT(1e-6)
            epsilon_raw = FT(1e-4)
        end
        
        push!(theta_ref, epsilon_raw)
        push!(param_names, "atom_$(atype)_ϵ") 
        epsilon_idx = length(theta_ref)

        processed_atom_types[atype] = sigma_idx  

        if idx ∈ solute_idx
            push!(solute_param_indices, sigma_idx, epsilon_idx)
        else
            push!(solvent_param_indices, sigma_idx, epsilon_idx)
        end
    end  
end  

trainable_param_indices = opt_cfg.optimize_solvent ? collect(eachindex(theta_ref)) : copy(solute_param_indices)
trainable_param_names = String[]
for idx in trainable_param_indices
    push!(trainable_param_names, param_names[idx])
end
trainable_position_map = Dict{Int, Int}()
for (i_local, i_global) in enumerate(trainable_param_indices)
    trainable_position_map[i_global] = i_local
end
theta_min = zeros(FT, length(theta_ref))
theta_max = zeros(FT, length(theta_ref))
phi_0     = zeros(FT, length(theta_ref))

for i in 1:length(theta_ref)
    p_name = param_names[i]
    
    atype_str = split(p_name, "_")[2]
    is_hydrogen = startswith(lowercase(atype_str), "h")
    
    if occursin("σ", p_name)
        if is_hydrogen
            theta_min[i] = FT(0.1)
            theta_max[i] = FT(0.4)
        else
            theta_min[i] = FT(0.2)
            theta_max[i] = FT(0.5)
        end
    elseif occursin("ϵ", p_name)
        if is_hydrogen
            theta_min[i] = FT(0.0)
            theta_max[i] = FT(0.5)
        else
            theta_min[i] = FT(0.0)
            theta_max[i] = FT(1.5)
        end
    end
    
    val_ref = clamp(theta_ref[i], theta_min[i] + FT(1e-4), theta_max[i] - FT(1e-4))
    val = (theta_max[i] - val_ref) / (val_ref - theta_min[i])
    phi_0[i] = (FT(1.0) / opt_cfg.k_sigmoid) * log(val)
end

phi_active = zeros(FT, length(theta_ref))
theta_active = map_phi_to_theta(phi_active, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)

idxs_solv = build_index_maps(sys_dummy, processed_atom_types)
idxs_vac = build_index_maps(System(sim_cfg.pdb_vac, ff; array_type=AT, nonbonded_method=:none), processed_atom_types)

# ==============================================================================
# --- MAIN PIPELINE ORCHESTRATION ---
# ==============================================================================

for macro_epoch in 1:max_macro_epochs
    println("\n>>> STARTING MACRO EPOCH $macro_epoch <<<")  

    warm_start_solv = macro_epoch > 1 && !isnothing(runtime.restart_cache_solv)
    warm_start_vac  = macro_epoch > 1 && !isnothing(runtime.restart_cache_vac)

    logger_solv = AWHEnsembleLogger(T_coord, T_vol, T_en, production_log_interval)
    awh_solv, sys_solv = setup_alchemical_awh(
        sim_cfg.pdb_solv,
        solute_idx;
        is_vacuum=false,
        logger=logger_solv,
        injected_bias=runtime.active_bias_solv,
        optimized_params=theta_active,
        param_idxs=idxs_solv,
        restart_state=warm_start_solv ? runtime.restart_cache_solv : nothing,
        restart_active_idx=warm_start_solv ? runtime.restart_cache_solv.active_idx : 1,
        warm_start=warm_start_solv
    )

    logger_vac = AWHEnsembleLogger(T_coord, T_vol, T_en, production_log_interval)
    awh_vac, sys_vac = setup_alchemical_awh(
        sim_cfg.pdb_vac,
        solute_idx;
        is_vacuum=true,
        logger=logger_vac,
        injected_bias=runtime.active_bias_vac,
        optimized_params=theta_active,
        param_idxs=idxs_vac,
        restart_state=warm_start_vac ? runtime.restart_cache_vac : nothing,
        restart_active_idx=warm_start_vac ? runtime.restart_cache_vac.active_idx : 1,
        warm_start=warm_start_vac
    )

    if warm_start_solv
        restart_rmsd = coords_rmsd_nm(runtime.restart_cache_solv.coords, awh_solv.state.active_sys.coords)
        prev_vol = FT(ustrip(volume(runtime.restart_cache_solv.boundary)))
        new_vol = FT(ustrip(volume(awh_solv.state.active_sys.boundary)))
        idx_match = runtime.restart_cache_solv.active_idx == awh_solv.state.active_idx
        @info "Macro Start (Solvent): warm_start=true | λ_prev=$(runtime.restart_cache_solv.active_idx) | λ_new=$(awh_solv.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6)) | volume_ratio=$(round(new_vol / prev_vol, digits=6))"
        if restart_rmsd > restart_rmsd_tol_nm
            @info "  [!] Solvent restart RMSD above tolerance ($(restart_rmsd_tol_nm) nm)."
        end
    else
        @info "Macro Start (Solvent): warm_start=false (cold-start from PDB)."
    end

    if warm_start_vac
        restart_rmsd = coords_rmsd_nm(runtime.restart_cache_vac.coords, awh_vac.state.active_sys.coords)
        idx_match = runtime.restart_cache_vac.active_idx == awh_vac.state.active_idx
        @info "Macro Start (Vacuum): warm_start=true | λ_prev=$(runtime.restart_cache_vac.active_idx) | λ_new=$(awh_vac.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6))"
        if restart_rmsd > restart_rmsd_tol_nm
            @info "  [!] Vacuum restart RMSD above tolerance ($(restart_rmsd_tol_nm) nm)."
        end
    else
        @info "Macro Start (Vacuum): warm_start=false (cold-start from PDB)."
    end

    e_unit = sys_solv.energy_units  
    beta_val = FT(1.0 / ustrip(uconvert(e_unit, Unitful.R * T0)))  
    P0_energy_per_vol = FT(ustrip(uconvert(e_unit, P0 * FT(1.0)u"nm^3" * Unitful.Na)))
    
    dG_exp_physical = FT(sim_cfg.dG_exp_kcal_mol) * FT(4.184)
    dG_exp = dG_exp_physical * beta_val   

    awh_ready = false
    awh_readiness_reason = :not_checked
    split_gap = FT(Inf)
    split_gap_solv = FT(Inf)
    split_gap_vac = FT(Inf)
    parity_gap_solv = FT(Inf)
    parity_gap_vac = FT(Inf)

    local awh_solv_prod
    local awh_vac_prod
    local logger_solv_prod
    local logger_vac_prod
    local nbrs_solv
    local nbrs_vac
    local u_solv_ref
    local u_vac_ref

    stageA_stats_default = StageAStats(df_mean=FT(Inf), linear_neff=zero(FT))
    stageB_stats_default = StageBStats(split_gap=FT(Inf), parity_gap=FT(Inf), dG_half_1=FT(NaN), dG_half_2=FT(NaN))

    stageA_stats_solv = stageA_stats_default
    stageA_stats_vac = stageA_stats_default
    stageB_stats_solv = stageB_stats_default
    stageB_stats_vac = stageB_stats_default

    stageA_streak_solv = 0
    stageA_streak_vac = 0
    spent_steps_solv = 0
    spent_steps_vac = 0
    leg_status_solv = :active
    leg_status_vac = :active

    runtime.active_bias_solv = extract_awh_data(awh_solv)
    runtime.active_bias_vac = extract_awh_data(awh_vac)

    if awh_solv.state.in_initial_stage
        rewarm_steps = min(md_steps_rewarm, md_steps_budget)
        println("Running Solvated AWH Leg (Initial Rewarm)...")
        simulate!(awh_solv, rewarm_steps)
        spent_steps_solv += rewarm_steps
        runtime.active_bias_solv = extract_awh_data(awh_solv)
    end

    if awh_vac.state.in_initial_stage
        rewarm_steps = min(md_steps_rewarm, md_steps_budget)
        println("Running Vacuum AWH Leg (Initial Rewarm)...")
        simulate!(awh_vac, rewarm_steps)
        spent_steps_vac += rewarm_steps
        runtime.active_bias_vac = extract_awh_data(awh_vac)
    end

    while true
        if leg_status_solv == :active
            remaining_steps_solv = md_steps_budget - spent_steps_solv
            if remaining_steps_solv <= 0
                leg_status_solv = :budget_exhausted
            else
                block_steps = min(md_steps_block, remaining_steps_solv)
                println("Running Solvated AWH Leg (Stage A Block)...")
                simulate!(awh_solv, block_steps)
                spent_steps_solv += block_steps
                runtime.active_bias_solv = extract_awh_data(awh_solv)

                stageA_stats_solv = StageAStats(evaluate_stage_a_readiness(
                    awh_solv, awh_convergence_tol;
                    tail_lag=awh_tail_lag,
                    min_linear_neff=awh_min_linear_neff,
                    min_round_trips=awh_min_round_trips,
                    endpoint_min_fraction=awh_endpoint_min_fraction,
                    high_idx=num_lambda_states,
                ))

                if stageA_stats_solv.ready
                    stageA_streak_solv += 1
                else
                    stageA_streak_solv = 0
                end

                if stageA_streak_solv >= awh_stageA_stable_blocks
                    stageB_stats_solv = StageBStats(run_stage_b_probe(
                        awh_solv, sys_solv, theta_active, param_names, idxs_solv, num_lambda_states,
                        beta_val, awh_split_tol_kT, awh_parity_tol_kT;
                        md_steps_probe=md_steps_probe_solv,
                        leg_name="solvent",
                        include_pv=true,
                        P0_energy_per_vol=P0_energy_per_vol
                    ))
                    split_gap_solv = stageB_stats_solv.split_gap
                    parity_gap_solv = stageB_stats_solv.parity_gap
                    if stageB_stats_solv.ready
                        leg_status_solv = :ready_frozen
                        @info "Solvent leg frozen after passing Stage B (split_gap=$(round(split_gap_solv, digits=4)) kT, parity_gap=$(round(parity_gap_solv, digits=4)) kT)."
                    else
                        stageA_streak_solv = 0
                    end
                end

                if leg_status_solv == :active && spent_steps_solv >= md_steps_budget
                    leg_status_solv = :budget_exhausted
                end
            end
        end

        if leg_status_vac == :active
            remaining_steps_vac = md_steps_budget - spent_steps_vac
            if remaining_steps_vac <= 0
                leg_status_vac = :budget_exhausted
            else
                block_steps = min(md_steps_block, remaining_steps_vac)
                println("Running Vacuum AWH Leg (Stage A Block)...")
                simulate!(awh_vac, block_steps)
                spent_steps_vac += block_steps
                runtime.active_bias_vac = extract_awh_data(awh_vac)

                stageA_stats_vac = StageAStats(evaluate_stage_a_readiness(
                    awh_vac, awh_convergence_tol;
                    tail_lag=awh_tail_lag,
                    min_linear_neff=awh_min_linear_neff,
                    min_round_trips=awh_min_round_trips,
                    endpoint_min_fraction=awh_endpoint_min_fraction,
                    high_idx=num_lambda_states,
                ))

                if stageA_stats_vac.ready
                    stageA_streak_vac += 1
                else
                    stageA_streak_vac = 0
                end

                if stageA_streak_vac >= awh_stageA_stable_blocks
                    stageB_stats_vac = StageBStats(run_stage_b_probe(
                        awh_vac, sys_vac, theta_active, param_names, idxs_vac, num_lambda_states,
                        beta_val, awh_split_tol_kT, awh_parity_tol_kT;
                        md_steps_probe=md_steps_probe_vac,
                        leg_name="vacuum",
                        include_pv=false
                    ))
                    split_gap_vac = stageB_stats_vac.split_gap
                    parity_gap_vac = stageB_stats_vac.parity_gap
                    if stageB_stats_vac.ready
                        leg_status_vac = :ready_frozen
                        @info "Vacuum leg frozen after passing Stage B (split_gap=$(round(split_gap_vac, digits=4)) kT, parity_gap=$(round(parity_gap_vac, digits=4)) kT)."
                    else
                        stageA_streak_vac = 0
                    end
                end

                if leg_status_vac == :active && spent_steps_vac >= md_steps_budget
                    leg_status_vac = :budget_exhausted
                end
            end
        end

        @info "AWH Block Status: solv=$(leg_status_solv) vac=$(leg_status_vac) | spent_ns_solv=$(round(steps_to_ns(spent_steps_solv), digits=2))/$(round(ustrip(u"ns", awh_budget_time), digits=2)) | spent_ns_vac=$(round(steps_to_ns(spent_steps_vac), digits=2))/$(round(ustrip(u"ns", awh_budget_time), digits=2))"
        @info "  Stage A (Solv): df=$(round(stageA_stats_solv.df_mean, digits=6)) (ok=$(stageA_stats_solv.df_ready)) | neff=$(round(stageA_stats_solv.linear_neff, digits=1)) (ok=$(stageA_stats_solv.neff_ready)) | rt=$(stageA_stats_solv.round_trips) (ok=$(stageA_stats_solv.round_trip_ready)) | endpt=($(round(stageA_stats_solv.endpoint_low, digits=3)), $(round(stageA_stats_solv.endpoint_high, digits=3))) (ok=$(stageA_stats_solv.endpoint_ready)) | streak=$stageA_streak_solv"
        @info "  Stage A (Vac):  df=$(round(stageA_stats_vac.df_mean, digits=6)) (ok=$(stageA_stats_vac.df_ready)) | neff=$(round(stageA_stats_vac.linear_neff, digits=1)) (ok=$(stageA_stats_vac.neff_ready)) | rt=$(stageA_stats_vac.round_trips) (ok=$(stageA_stats_vac.round_trip_ready)) | endpt=($(round(stageA_stats_vac.endpoint_low, digits=3)), $(round(stageA_stats_vac.endpoint_high, digits=3))) (ok=$(stageA_stats_vac.endpoint_ready)) | streak=$stageA_streak_vac"
        if stageB_stats_solv.n_frames > 0
            @info "  Stage B (Solv): split_gap=$(round(stageB_stats_solv.split_gap, digits=4)) kT (ok=$(stageB_stats_solv.split_ready)) | parity_gap=$(round(stageB_stats_solv.parity_gap, digits=4)) kT (ok=$(stageB_stats_solv.parity_ready)) | frames=$(stageB_stats_solv.n_frames)"
        end
        if stageB_stats_vac.n_frames > 0
            @info "  Stage B (Vac):  split_gap=$(round(stageB_stats_vac.split_gap, digits=4)) kT (ok=$(stageB_stats_vac.split_ready)) | parity_gap=$(round(stageB_stats_vac.parity_gap, digits=4)) kT (ok=$(stageB_stats_vac.parity_ready)) | frames=$(stageB_stats_vac.n_frames)"
        end

        if leg_status_solv == :ready_frozen && leg_status_vac == :ready_frozen
            awh_ready = true
            awh_readiness_reason = :ready
            break
        end

        if leg_status_solv == :budget_exhausted || leg_status_vac == :budget_exhausted
            awh_ready = false
            awh_readiness_reason = :awh_not_ready_budget
            break
        end
    end

    if !awh_ready
        @info "Macro $macro_epoch skipped: strict AWH readiness failed ($awh_readiness_reason). Status=(solv=$leg_status_solv, vac=$leg_status_vac), StageA df=(solv=$(round(stageA_stats_solv.df_mean, digits=6)), vac=$(round(stageA_stats_vac.df_mean, digits=6))), neff=(solv=$(round(stageA_stats_solv.linear_neff, digits=1)), vac=$(round(stageA_stats_vac.linear_neff, digits=1))), rt=(solv=$(stageA_stats_solv.round_trips), vac=$(stageA_stats_vac.round_trips)), parity=(solv=$(round(parity_gap_solv, digits=4)), vac=$(round(parity_gap_vac, digits=4))) kT"
        continue
    end

    split_gap = max(split_gap_solv, split_gap_vac)
    runtime.active_bias_solv = extract_awh_data(awh_solv)
    runtime.active_bias_vac = extract_awh_data(awh_vac)

    println("Running Production Runs (Extended Ensemble) [Readiness-Passed Dataset]...")
    clear_awh_logger_histories!(awh_solv)
    clear_awh_logger_histories!(awh_vac)

    awh_solv_prod = AWHSimulation(awh_solv.state; num_md_steps=awh_solv.n_md_steps, update_freq=typemax(Int), well_tempered_factor=Inf)
    awh_solv_prod.state.active_sys.loggers.awh_logger.should_log = true
    simulate!(awh_solv_prod, md_steps_prod)

    awh_vac_prod = AWHSimulation(awh_vac.state; num_md_steps=awh_vac.n_md_steps, update_freq=typemax(Int), well_tempered_factor=Inf)
    awh_vac_prod.state.active_sys.loggers.awh_logger.should_log = true
    simulate!(awh_vac_prod, md_steps_prod)

    runtime.restart_cache_solv = capture_restart_state(awh_solv_prod)
    runtime.restart_cache_vac  = capture_restart_state(awh_vac_prod)
    @info "Captured restart state for next macro epoch: λ_solv=$(runtime.restart_cache_solv.active_idx) | λ_vac=$(runtime.restart_cache_vac.active_idx)"

    logger_solv_prod = get_production_logger(awh_solv_prod, "solvent")
    logger_vac_prod  = get_production_logger(awh_vac_prod, "vacuum")

    nbrs_solv = precompute_neighbors(logger_solv_prod, awh_solv_prod.state.active_sys)
    nbrs_vac  = precompute_neighbors(logger_vac_prod, awh_vac_prod.state.active_sys)

    u_solv_ref, _ = evaluate_ensemble(
        logger_solv_prod, nbrs_solv, awh_solv_prod, sys_solv,
        theta_active, param_names, idxs_solv...; compute_gradients=false
    )
    u_vac_ref, _  = evaluate_ensemble(
        logger_vac_prod, nbrs_vac, awh_vac_prod, sys_vac,
        theta_active, param_names, idxs_vac...; compute_gradients=false
    )
    GC.gc()

    opt_result = run_optimization_phase!(
        phi_active,
        theta_active,
        runtime.active_bias_solv,
        runtime.active_bias_vac,
        logger_solv_prod,
        logger_vac_prod,
        nbrs_solv,
        nbrs_vac,
        awh_solv_prod,
        awh_vac_prod,
        sys_solv,
        sys_vac,
        u_solv_ref,
        u_vac_ref,
        param_names,
        trainable_param_names,
        trainable_param_indices,
        trainable_position_map,
        solute_param_indices,
        solvent_param_indices,
        idxs_solv,
        idxs_vac,
        theta_min,
        theta_max,
        phi_0,
        beta_val,
        dG_std_corr,
        dG_exp,
        P0_energy_per_vol,
        opt_cfg,
    )

    @info "Macro $macro_epoch Residual Summary: Start = $(round(opt_result.macro_start_residual, digits=3)) | Best = $(round(opt_result.best_macro_residual, digits=3)) (Epoch $(opt_result.best_macro_epoch)) | End = $(round(opt_result.macro_end_residual, digits=3)) | Exit = $(opt_result.phase2_exit_reason) | AWH split-gap(max) = $(round(split_gap, digits=4)) kT | Parity gaps = (solv=$(round(parity_gap_solv, digits=4)), vac=$(round(parity_gap_vac, digits=4))) kT"

end

    runtime.phi_active = phi_active
    runtime.theta_active = theta_active
    return runtime
end
