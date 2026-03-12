function sync_runtime_aliases!(runtime::RuntimeState)
    runtime.active_bias_solv = get(runtime.active_bias, :solvent, nothing)
    runtime.active_bias_vac = get(runtime.active_bias, :vacuum, nothing)
    runtime.restart_cache_solv = get(runtime.restart_cache, :solvent, nothing)
    runtime.restart_cache_vac = get(runtime.restart_cache, :vacuum, nothing)
    return runtime
end


time_to_steps_floor(duration) = Int(floor(uconvert(unit(Δt), duration) / Δt))
time_to_steps_round(duration) = max(1, Int(round(uconvert(unit(Δt), duration) / Δt)))


function compute_standard_state_correction(cycle_cfg::ThermodynamicCycleConfig, ::Type{FT}) where {FT <: AbstractFloat}
    if !cycle_cfg.include_standard_state_correction
        return zero(FT)
    end
    V_gas = ustrip(u"nm^3", Unitful.k * T0 / P0)
    V_std = ustrip(u"nm^3", 1.0u"L" / (1.0u"mol" * Unitful.Na))
    return FT(log(V_gas / V_std))
end


function initialize_parameter_state(
    sim_cfg::SimulationConfig,
    opt_cfg::OptimizationConfig,
    cycle_cfg::ThermodynamicCycleConfig,
)
    FT = sim_cfg.FT
    bounds_cfg = sim_cfg.parameter_bounds
    solute_idx = sim_cfg.solute_idx

    ref_leg = resolve_parameter_reference_leg(sim_cfg, cycle_cfg)
    sys_ref = System(ref_leg.pdb, ff; array_type=AT, nonbonded_method=:none)
    atoms_cpu = Molly.from_device(sys_ref.atoms)

    theta_ref = Vector{FT}()
    param_names = String[]
    param_kind = Symbol[]
    param_is_hydrogen = Bool[]
    processed_atom_types = Dict{String, Int}()
    solute_param_indices = Int[]
    solvent_param_indices = Int[]

    sigma_floor = FT(bounds_cfg.sigma_floor)
    epsilon_floor = FT(bounds_cfg.epsilon_floor)
    clamp_eps = FT(bounds_cfg.reference_clamp_eps)

    for idx in eachindex(atoms_cpu)
        atom = atoms_cpu[idx]
        atype = String(sys_ref.atoms_data[idx].atom_type)
        if haskey(processed_atom_types, atype)
            continue
        end

        is_hydrogen = startswith(lowercase(atype), "h")

        sigma_raw = FT(ustrip(atom.σ))
        if sigma_raw <= FT(1e-6) || sigma_raw == one(FT)
            sigma_raw = sigma_floor
        end
        push!(theta_ref, sigma_raw)
        push!(param_names, "atom_$(atype)_σ")
        push!(param_kind, :sigma)
        push!(param_is_hydrogen, is_hydrogen)
        sigma_idx = length(theta_ref)

        epsilon_raw = FT(ustrip(atom.ϵ))
        if epsilon_raw <= FT(1e-6)
            epsilon_raw = epsilon_floor
        end
        push!(theta_ref, epsilon_raw)
        push!(param_names, "atom_$(atype)_ϵ")
        push!(param_kind, :epsilon)
        push!(param_is_hydrogen, is_hydrogen)
        epsilon_idx = length(theta_ref)

        processed_atom_types[atype] = sigma_idx
        if idx ∈ solute_idx
            push!(solute_param_indices, sigma_idx, epsilon_idx)
        else
            push!(solvent_param_indices, sigma_idx, epsilon_idx)
        end
    end

    trainable_param_indices = opt_cfg.optimize_solvent ? collect(eachindex(theta_ref)) : copy(solute_param_indices)
    trainable_param_names = [param_names[idx] for idx in trainable_param_indices]
    trainable_position_map = Dict{Int, Int}()
    for (i_local, i_global) in enumerate(trainable_param_indices)
        trainable_position_map[i_global] = i_local
    end

    theta_min = zeros(FT, length(theta_ref))
    theta_max = zeros(FT, length(theta_ref))
    phi_0 = zeros(FT, length(theta_ref))

    for i in eachindex(theta_ref)
        if param_kind[i] == :sigma
            if param_is_hydrogen[i]
                theta_min[i] = FT(bounds_cfg.sigma_hydrogen_min)
                theta_max[i] = FT(bounds_cfg.sigma_hydrogen_max)
            else
                theta_min[i] = FT(bounds_cfg.sigma_heavy_min)
                theta_max[i] = FT(bounds_cfg.sigma_heavy_max)
            end
        else
            if param_is_hydrogen[i]
                theta_min[i] = FT(bounds_cfg.epsilon_hydrogen_min)
                theta_max[i] = FT(bounds_cfg.epsilon_hydrogen_max)
            else
                theta_min[i] = FT(bounds_cfg.epsilon_heavy_min)
                theta_max[i] = FT(bounds_cfg.epsilon_heavy_max)
            end
        end

        val_ref = clamp(theta_ref[i], theta_min[i] + clamp_eps, theta_max[i] - clamp_eps)
        val = (theta_max[i] - val_ref) / (val_ref - theta_min[i])
        phi_0[i] = (FT(1.0) / opt_cfg.k_sigmoid) * log(val)
    end

    phi_active = zeros(FT, length(theta_ref))
    theta_active = map_phi_to_theta(phi_active, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)

    idxs_by_leg = Dict{Symbol, Any}()
    for leg in cycle_cfg.legs
        sys_leg = leg.name == ref_leg.name ? sys_ref : System(leg.pdb, ff; array_type=AT, nonbonded_method=:none)
        idxs_by_leg[leg.name] = build_index_maps(sys_leg, processed_atom_types)
    end

    return (
        phi_active=phi_active,
        theta_active=theta_active,
        param_names=param_names,
        trainable_param_names=trainable_param_names,
        trainable_param_indices=trainable_param_indices,
        trainable_position_map=trainable_position_map,
        solute_param_indices=solute_param_indices,
        solvent_param_indices=solvent_param_indices,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_0=phi_0,
        idxs_by_leg=idxs_by_leg,
    )
end


function setup_macro_legs(
    cycle_cfg::ThermodynamicCycleConfig,
    sim_cfg::SimulationConfig,
    runtime::RuntimeState,
    theta_active::Vector{FT},
    idxs_by_leg::Dict{Symbol, Any},
    T_coord,
    T_vol,
    T_en,
    macro_epoch::Int,
    restart_rmsd_tol_nm::FT,
) where {FT <: AbstractFloat}
    awh_by_leg = Dict{Symbol, Any}()
    sys_by_leg = Dict{Symbol, Any}()
    bounds_cfg = sim_cfg.parameter_bounds
    sigma_seed = FT(bounds_cfg.sigma_floor)u"nm"
    epsilon_seed = FT(bounds_cfg.epsilon_floor)u"kJ/mol"

    for leg in cycle_cfg.legs
        warm_start = macro_epoch > 1 && haskey(runtime.restart_cache, leg.name) && !isnothing(runtime.restart_cache[leg.name])
        restart_state = warm_start ? runtime.restart_cache[leg.name] : nothing

        logger = AWHEnsembleLogger(T_coord, T_vol, T_en, sim_cfg.production_log_interval)
        awh_leg, sys_leg = setup_alchemical_awh(
            leg.pdb,
            sim_cfg.solute_idx;
            is_vacuum=leg.is_vacuum,
            logger=logger,
            injected_bias=get(runtime.active_bias, leg.name, nothing),
            optimized_params=theta_active,
            param_idxs=idxs_by_leg[leg.name],
            restart_state=restart_state,
            restart_active_idx=warm_start ? restart_state.active_idx : 1,
            warm_start=warm_start,
            sigma_seed=sigma_seed,
            epsilon_seed=epsilon_seed,
        )
        awh_by_leg[leg.name] = awh_leg
        sys_by_leg[leg.name] = sys_leg

        leg_label = uppercasefirst(String(leg.name))
        if warm_start
            restart_rmsd = coords_rmsd_nm(restart_state.coords, awh_leg.state.active_sys.coords)
            idx_match = restart_state.active_idx == awh_leg.state.active_idx
            if leg.is_vacuum
                @info "Macro Start ($leg_label): warm_start=true | λ_prev=$(restart_state.active_idx) | λ_new=$(awh_leg.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6))"
            else
                prev_vol = FT(ustrip(volume(restart_state.boundary)))
                new_vol = FT(ustrip(volume(awh_leg.state.active_sys.boundary)))
                @info "Macro Start ($leg_label): warm_start=true | λ_prev=$(restart_state.active_idx) | λ_new=$(awh_leg.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6)) | volume_ratio=$(round(new_vol / prev_vol, digits=6))"
            end
            if restart_rmsd > restart_rmsd_tol_nm
                @info "  [!] $leg_label restart RMSD above tolerance ($(restart_rmsd_tol_nm) nm)."
            end
        else
            @info "Macro Start ($leg_label): warm_start=false (cold-start from PDB)."
        end
    end

    return awh_by_leg, sys_by_leg
end


function run_readiness_loop!(
    cycle_cfg::ThermodynamicCycleConfig,
    awh_by_leg::Dict{Symbol, Any},
    sys_by_leg::Dict{Symbol, Any},
    idxs_by_leg::Dict{Symbol, Any},
    runtime::RuntimeState,
    theta_active::Vector{FT},
    param_names::Vector{String},
    md_steps_budget::Int,
    md_steps_block::Int,
    md_steps_rewarm::Int,
    probe_steps_by_leg::Dict{Symbol, Int},
    probe_stride_by_leg::Dict{Symbol, Int},
    probe_min_frames_by_leg::Dict{Symbol, Int},
    probe_max_frames_by_leg::Dict{Symbol, Int},
    awh_budget_time,
    awh_convergence_tol::FT,
    awh_min_linear_neff::Int,
    awh_split_tol_kT::FT,
    awh_parity_tol_kT::FT,
    awh_tail_lag::Int,
    awh_min_round_trips::Int,
    awh_endpoint_min_fraction::FT,
    awh_stageA_stable_blocks::Int,
    awh_stageB_cooldown_blocks::Int,
    beta_val::FT,
    p0_energy_per_vol::FT,
) where {FT <: AbstractFloat}
    stageA_default = StageAStats(df_mean=FT(Inf), linear_neff=zero(FT))
    stageB_default = StageBStats(split_gap=FT(Inf), parity_gap=FT(Inf), dG_half_1=FT(NaN), dG_half_2=FT(NaN))

    stageA_stats_by_leg = Dict{Symbol, StageAStats}(leg.name => stageA_default for leg in cycle_cfg.legs)
    stageB_stats_by_leg = Dict{Symbol, StageBStats}(leg.name => stageB_default for leg in cycle_cfg.legs)
    stageA_streak = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    stageB_cooldown = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    spent_steps = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    leg_status = Dict{Symbol, Symbol}(leg.name => :active for leg in cycle_cfg.legs)
    split_gap_by_leg = Dict{Symbol, FT}(leg.name => FT(Inf) for leg in cycle_cfg.legs)
    parity_gap_by_leg = Dict{Symbol, FT}(leg.name => FT(Inf) for leg in cycle_cfg.legs)
    cooldown_blocks = max(0, awh_stageB_cooldown_blocks)

    for leg in cycle_cfg.legs
        runtime.active_bias[leg.name] = extract_awh_data(awh_by_leg[leg.name])
    end
    sync_runtime_aliases!(runtime)

    for leg in cycle_cfg.legs
        awh_leg = awh_by_leg[leg.name]
        if !awh_leg.state.in_initial_stage
            continue
        end
        rewarm_steps = min(md_steps_rewarm, md_steps_budget)
        println("Running $(uppercasefirst(String(leg.name))) AWH Leg (Initial Rewarm)...")
        timed_phase("Initial Rewarm", String(leg.name); md_steps=rewarm_steps) do
            simulate!(awh_leg, rewarm_steps)
        end
        spent_steps[leg.name] += rewarm_steps
        runtime.active_bias[leg.name] = extract_awh_data(awh_leg)
    end
    sync_runtime_aliases!(runtime)

    awh_ready = false
    awh_readiness_reason = :not_checked

    while true
        for leg in cycle_cfg.legs
            name = leg.name
            if leg_status[name] != :active
                continue
            end

            awh_leg = awh_by_leg[name]
            remaining_steps = md_steps_budget - spent_steps[name]
            if remaining_steps <= 0
                leg_status[name] = :budget_exhausted
                continue
            end

            block_steps = min(md_steps_block, remaining_steps)
            println("Running $(uppercasefirst(String(name))) AWH Leg (Stage A Block)...")
            timed_phase("Stage A Block", String(name); md_steps=block_steps) do
                simulate!(awh_leg, block_steps)
            end
            spent_steps[name] += block_steps
            runtime.active_bias[name] = extract_awh_data(awh_leg)

            stageA_stats_by_leg[name] = StageAStats(evaluate_stage_a_readiness(
                awh_leg,
                awh_convergence_tol;
                tail_lag=awh_tail_lag,
                min_linear_neff=awh_min_linear_neff,
                min_round_trips=awh_min_round_trips,
                endpoint_min_fraction=awh_endpoint_min_fraction,
                high_idx=num_lambda_states,
            ))

            if stageA_stats_by_leg[name].ready
                stageA_streak[name] += 1
            else
                stageA_streak[name] = 0
            end

            if stageA_streak[name] >= awh_stageA_stable_blocks
                if stageB_cooldown[name] > 0
                    @info "Stage B ($(name)) cooldown active: remaining_checks=$(stageB_cooldown[name]); skipping probe this block."
                    stageB_cooldown[name] -= 1
                else
                    probe_steps = probe_steps_by_leg[name]
                    @info "Stage A ($(name)) reached stable streak ($(stageA_streak[name])/$(awh_stageA_stable_blocks)); entering Stage B probe | probe_steps=$probe_steps | probe_ns=$(round(steps_to_ns(probe_steps), digits=4))"
                    stageB_stats_by_leg[name] = StageBStats(run_stage_b_probe(
                        awh_leg,
                        sys_by_leg[name],
                        theta_active,
                        param_names,
                        idxs_by_leg[name],
                        num_lambda_states,
                        beta_val,
                        awh_split_tol_kT,
                        awh_parity_tol_kT;
                        md_steps_probe=probe_steps_by_leg[name],
                        leg_name=String(name),
                        include_pv=leg.include_pv,
                        P0_energy_per_vol=leg.include_pv ? p0_energy_per_vol : zero(FT),
                        probe_frame_stride=probe_stride_by_leg[name],
                        probe_min_frames=probe_min_frames_by_leg[name],
                        probe_max_frames=probe_max_frames_by_leg[name],
                    ))

                    split_gap_by_leg[name] = stageB_stats_by_leg[name].split_gap
                    parity_gap_by_leg[name] = stageB_stats_by_leg[name].parity_gap
                    if stageB_stats_by_leg[name].ready
                        leg_status[name] = :ready_frozen
                        @info "$(uppercasefirst(String(name))) leg frozen after passing Stage B (split_gap=$(round(split_gap_by_leg[name], digits=4)) kT, parity_gap=$(round(parity_gap_by_leg[name], digits=4)) kT)."
                    else
                        stageA_streak[name] = 0
                        stageB_cooldown[name] = cooldown_blocks
                        if cooldown_blocks > 0
                            @info "Stage B ($(name)) failed; scheduling cooldown for $cooldown_blocks stable-check opportunities."
                        end
                    end
                end
            end

            if leg_status[name] == :active && spent_steps[name] >= md_steps_budget
                leg_status[name] = :budget_exhausted
            end
        end

        sync_runtime_aliases!(runtime)

        status_msg = join(["$(leg.name)=$(leg_status[leg.name])" for leg in cycle_cfg.legs], " ")
        spent_msg = join(
            [
                "$(leg.name)=$(round(steps_to_ns(spent_steps[leg.name]), digits=2))/$(round(ustrip(u"ns", awh_budget_time), digits=2))"
                for leg in cycle_cfg.legs
            ],
            " | ",
        )
        @info "AWH Block Status: $status_msg | spent_ns: $spent_msg"

        for leg in cycle_cfg.legs
            name = leg.name
            statsA = stageA_stats_by_leg[name]
            @info "  Stage A ($(name)): df=$(round(statsA.df_mean, digits=6)) (ok=$(statsA.df_ready)) | neff=$(round(statsA.linear_neff, digits=1)) (ok=$(statsA.neff_ready)) | rt=$(statsA.round_trips) (ok=$(statsA.round_trip_ready)) | endpt=($(round(statsA.endpoint_low, digits=3)), $(round(statsA.endpoint_high, digits=3))) (ok=$(statsA.endpoint_ready)) | n_hist=$(statsA.n_hist) | streak=$(stageA_streak[name]) | cooldown=$(stageB_cooldown[name])"

            statsB = stageB_stats_by_leg[name]
            if statsB.n_frames > 0
                @info "  Stage B ($(name)): split_gap=$(round(statsB.split_gap, digits=4)) kT (ok=$(statsB.split_ready)) | parity_gap=$(round(statsB.parity_gap, digits=4)) kT (ok=$(statsB.parity_ready)) | frames=$(statsB.n_frames)"
            end
        end

        all_ready = all(leg_status[leg.name] == :ready_frozen for leg in cycle_cfg.legs)
        if all_ready
            awh_ready = true
            awh_readiness_reason = :ready
            break
        end

        any_budget_exhausted = any(leg_status[leg.name] == :budget_exhausted for leg in cycle_cfg.legs)
        if any_budget_exhausted
            awh_ready = false
            awh_readiness_reason = :awh_not_ready_budget
            break
        end
    end

    return (
        awh_ready=awh_ready,
        awh_readiness_reason=awh_readiness_reason,
        stageA_stats_by_leg=stageA_stats_by_leg,
        stageB_stats_by_leg=stageB_stats_by_leg,
        split_gap_by_leg=split_gap_by_leg,
        parity_gap_by_leg=parity_gap_by_leg,
        leg_status=leg_status,
    )
end


function collect_production_artifacts!(
    cycle_cfg::ThermodynamicCycleConfig,
    awh_by_leg::Dict{Symbol, Any},
    sys_by_leg::Dict{Symbol, Any},
    idxs_by_leg::Dict{Symbol, Any},
    runtime::RuntimeState,
    theta_active::Vector{FT},
    param_names::Vector{String},
    md_steps_prod::Int,
    p0_energy_per_vol::FT,
) where {FT <: AbstractFloat}
    for leg in cycle_cfg.legs
        runtime.active_bias[leg.name] = extract_awh_data(awh_by_leg[leg.name])
    end
    sync_runtime_aliases!(runtime)

    println("Running Production Runs (Extended Ensemble) [Readiness-Passed Dataset]...")
    for leg in cycle_cfg.legs
        clear_awh_logger_histories!(awh_by_leg[leg.name])
    end

    artifacts = LegArtifacts[]
    for leg in cycle_cfg.legs
        name = leg.name
        awh_prod = AWHSimulation(
            awh_by_leg[name].state;
            num_md_steps=awh_by_leg[name].n_md_steps,
            update_freq=typemax(Int),
            well_tempered_factor=Inf,
            coverage_type=:physical
        )
        awh_prod.state.active_sys.loggers.awh_logger.should_log = true
        simulate!(awh_prod, md_steps_prod)

        runtime.restart_cache[name] = capture_restart_state(awh_prod)

        logger_prod = get_production_logger(awh_prod, String(name))
        neighbors = precompute_neighbors(logger_prod, awh_prod.state.active_sys)
        u_ref, _ = evaluate_ensemble(
            logger_prod,
            neighbors,
            awh_prod,
            sys_by_leg[name],
            theta_active,
            param_names,
            idxs_by_leg[name]...;
            compute_gradients=false,
        )

        push!(artifacts, LegArtifacts(
            name=name,
            coefficient=FT(leg.coefficient),
            include_pv=leg.include_pv,
            p0_energy_per_vol=leg.include_pv ? p0_energy_per_vol : zero(FT),
            awh_prod=awh_prod,
            logger_prod=logger_prod,
            neighbors=neighbors,
            u_ref=u_ref,
            sys_base=sys_by_leg[name],
            active_bias=runtime.active_bias[name],
            idxs=idxs_by_leg[name],
        ))
    end

    sync_runtime_aliases!(runtime)
    return artifacts
end


function run_pipeline(; sim_cfg::SimulationConfig=default_simulation_config(), opt_cfg::OptimizationConfig=default_optimization_config())
    apply_simulation_config!(sim_cfg)
    FT = sim_cfg.FT
    cycle_cfg = validate_cycle_config(resolved_cycle_config(sim_cfg))
    runtime = RuntimeState()

    dG_std_corr = compute_standard_state_correction(cycle_cfg, FT)

    md_steps_budget = time_to_steps_floor(sim_cfg.awh_budget_time)
    md_steps_block = time_to_steps_round(sim_cfg.awh_block_time)
    md_steps_prod = time_to_steps_floor(sim_cfg.md_time_production)
    md_steps_rewarm = min(
        max(1, Int(round(md_steps_budget * opt_cfg.rewarm_fraction))),
        md_steps_budget,
    )

    probe_steps_by_leg = Dict{Symbol, Int}()
    probe_stride_by_leg = Dict{Symbol, Int}()
    probe_min_frames_by_leg = Dict{Symbol, Int}()
    probe_max_frames_by_leg = Dict{Symbol, Int}()
    for leg in cycle_cfg.legs
        probe_steps_by_leg[leg.name] = time_to_steps_round(leg.probe_time)
        if leg.is_vacuum
            probe_stride_by_leg[leg.name] = max(1, sim_cfg.awh_probe_reweight_stride_vac)
            probe_min_frames_by_leg[leg.name] = max(2, sim_cfg.awh_probe_reweight_min_frames_vac)
            probe_max_frames_by_leg[leg.name] = max(0, sim_cfg.awh_probe_reweight_max_frames_vac)
        else
            probe_stride_by_leg[leg.name] = max(1, sim_cfg.awh_probe_reweight_stride_solv)
            probe_min_frames_by_leg[leg.name] = max(2, sim_cfg.awh_probe_reweight_min_frames_solv)
            probe_max_frames_by_leg[leg.name] = max(0, sim_cfg.awh_probe_reweight_max_frames_solv)
        end
    end

    T_coord = typeof(FT(1.0)u"nm")
    T_vol = typeof(FT(1.0)u"nm^3")
    T_en = typeof(FT(1.0)u"kJ * mol^-1")

    pstate = initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)
    phi_active = pstate.phi_active
    theta_active = pstate.theta_active
    param_names = pstate.param_names
    trainable_param_names = pstate.trainable_param_names
    trainable_param_indices = pstate.trainable_param_indices
    trainable_position_map = pstate.trainable_position_map
    solute_param_indices = pstate.solute_param_indices
    solvent_param_indices = pstate.solvent_param_indices
    theta_min = pstate.theta_min
    theta_max = pstate.theta_max
    phi_0 = pstate.phi_0
    idxs_by_leg = pstate.idxs_by_leg

    for macro_epoch in 1:opt_cfg.max_macro_epochs
        println("\n>>> STARTING MACRO EPOCH $macro_epoch <<<")

        awh_by_leg, sys_by_leg = setup_macro_legs(
            cycle_cfg,
            sim_cfg,
            runtime,
            theta_active,
            idxs_by_leg,
            T_coord,
            T_vol,
            T_en,
            macro_epoch,
            opt_cfg.restart_rmsd_tol_nm,
        )

        first_leg = first(cycle_cfg.legs)
        e_unit = sys_by_leg[first_leg.name].energy_units
        beta_val = FT(1.0 / ustrip(uconvert(e_unit, Unitful.R * T0)))
        p0_energy_per_vol = FT(ustrip(uconvert(e_unit, P0 * FT(1.0)u"nm^3" * Unitful.Na)))

        dG_exp_physical = FT(cycle_cfg.target_dG_kcal_mol) * FT(4.184)
        dG_exp = dG_exp_physical * beta_val

        readiness_result = run_readiness_loop!(
            cycle_cfg,
            awh_by_leg,
            sys_by_leg,
            idxs_by_leg,
            runtime,
            theta_active,
            param_names,
            md_steps_budget,
            md_steps_block,
            md_steps_rewarm,
            probe_steps_by_leg,
            probe_stride_by_leg,
            probe_min_frames_by_leg,
            probe_max_frames_by_leg,
            sim_cfg.awh_budget_time,
            opt_cfg.awh_convergence_tol,
            opt_cfg.awh_min_linear_neff,
            opt_cfg.awh_split_tol_kT,
            opt_cfg.awh_parity_tol_kT,
            opt_cfg.awh_tail_lag,
            opt_cfg.awh_min_round_trips,
            opt_cfg.awh_endpoint_min_fraction,
            opt_cfg.awh_stageA_stable_blocks,
            opt_cfg.awh_stageB_cooldown_blocks,
            beta_val,
            p0_energy_per_vol,
        )

        if !readiness_result.awh_ready
            stageA_summary = join(
                [
                    "$(leg.name)=df:$(round(readiness_result.stageA_stats_by_leg[leg.name].df_mean, digits=6)) neff:$(round(readiness_result.stageA_stats_by_leg[leg.name].linear_neff, digits=1)) rt:$(readiness_result.stageA_stats_by_leg[leg.name].round_trips)"
                    for leg in cycle_cfg.legs
                ],
                " | ",
            )
            parity_summary = join(
                [
                    "$(leg.name)=$(round(readiness_result.parity_gap_by_leg[leg.name], digits=4))"
                    for leg in cycle_cfg.legs
                ],
                ", ",
            )
            status_summary = join(
                ["$(leg.name)=$(readiness_result.leg_status[leg.name])" for leg in cycle_cfg.legs],
                " ",
            )
            @info "Macro $macro_epoch skipped: strict AWH readiness failed ($(readiness_result.awh_readiness_reason)). Status=[$status_summary] | StageA=[$stageA_summary] | Parity gaps=[$parity_summary] kT"
            continue
        end

        leg_artifacts = collect_production_artifacts!(
            cycle_cfg,
            awh_by_leg,
            sys_by_leg,
            idxs_by_leg,
            runtime,
            theta_active,
            param_names,
            md_steps_prod,
            p0_energy_per_vol,
        )

        GC.gc()

        opt_result = run_optimization_phase!(
            phi_active,
            theta_active,
            leg_artifacts,
            param_names,
            trainable_param_names,
            trainable_param_indices,
            trainable_position_map,
            solute_param_indices,
            solvent_param_indices,
            theta_min,
            theta_max,
            phi_0,
            beta_val,
            dG_std_corr,
            dG_exp,
            opt_cfg,
        )

        split_gap_max = maximum(values(readiness_result.split_gap_by_leg))
        parity_summary = join(
            [
                "$(leg.name)=$(round(readiness_result.parity_gap_by_leg[leg.name], digits=4))"
                for leg in cycle_cfg.legs
            ],
            ", ",
        )

        @info "Macro $macro_epoch Residual Summary: Start = $(round(opt_result.macro_start_residual, digits=3)) | Best = $(round(opt_result.best_macro_residual, digits=3)) (Epoch $(opt_result.best_macro_epoch)) | End = $(round(opt_result.macro_end_residual, digits=3)) | Exit = $(opt_result.phase2_exit_reason) | AWH split-gap(max) = $(round(split_gap_max, digits=4)) kT | Parity gaps = [$parity_summary] kT"
    end

    runtime.phi_active = phi_active
    runtime.theta_active = theta_active
    sync_runtime_aliases!(runtime)
    return runtime
end
