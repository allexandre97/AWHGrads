#!/usr/bin/env julia

# Solvent-leg ensemble-eval benchmark:
# 1) reuse the full-example solvent setup
# 2) run a single solvent AWH segment for 0.5 ns
# 3) clone a frozen-bias production segment for 0.1 ns
# 4) collect at least 500 production frames
# 5) warm up the exact Enzyme gradient path on one cached frame/template
# 6) benchmark `evaluate_ensemble(...; compute_gradients=true)`
# 7) run one actual optimization phase on the same frozen-bias artifact
#
# This script exists to exercise and profile the expensive optimization replay
# path without waiting for the full readiness + optimization pipeline.
#
# Run with Julia 1.11:
#   julia +1.11 scripts/run_optimization_epoch_smoke.jl

using Unitful

include(joinpath(@__DIR__, "run_alch_full_example.jl"))

function parameter_pool_with(pool::AWHGrads.ParameterPoolConfig; kwargs...)
    fields = [name => getfield(pool, name) for name in fieldnames(AWHGrads.ParameterPoolConfig)]
    for (k, v) in kwargs
        idx = findfirst(p -> first(p) == k, fields)
        isnothing(idx) && throw(ArgumentError("Unknown ParameterPoolConfig field override: $(k)."))
        fields[idx] = k => v
    end
    return AWHGrads.ParameterPoolConfig(; fields...)
end

function charge_enabled_smoke_parameter_pools(pool_cfgs::Vector{AWHGrads.ParameterPoolConfig})
    return [
        pool.name == :inserted_region ?
            parameter_pool_with(
                pool;
                trainable_families=unique(vcat(pool.trainable_families, [:charge_chi, :charge_eta])),
            ) :
            pool
        for pool in pool_cfgs
    ]
end

function solvent_only_cycle_from_example(cycle_cfg)
    solvent_leg = only(filter(leg -> leg.name == :solvent, cycle_cfg.legs))
    return AWHGrads.ThermodynamicCycleConfig(
        legs=[solvent_leg],
        include_standard_state_correction=cycle_cfg.include_standard_state_correction,
        target_dG_kcal_mol=cycle_cfg.target_dG_kcal_mol,
    )
end

function production_interval_for_min_frames(production_steps::Int, min_frames::Int)
    min_frames > 0 || throw(ArgumentError("`min_frames` must be positive, got $min_frames."))
    return max(1, fld(production_steps, min_frames))
end

function expected_logged_frames(md_steps::Int, log_interval::Int)
    log_interval > 0 || throw(ArgumentError("`log_interval` must be positive, got $log_interval."))
    return fld(md_steps, log_interval)
end

function build_optimization_smoke_configs(; enable_charge_training::Bool=true)
    sim_cfg, opt_cfg = build_example_configs()
    FT = sim_cfg.FT

    awh_time = FT(0.1)u"ns"
    production_time = FT(0.1)u"ns"
    min_frames = 500
    awh_steps = AWHGrads.time_to_steps_floor(awh_time)
    production_steps = AWHGrads.time_to_steps_floor(production_time)
    production_log_interval = production_interval_for_min_frames(production_steps, min_frames)
    expected_frames = expected_logged_frames(production_steps, production_log_interval)
    parameter_pools = enable_charge_training ?
        charge_enabled_smoke_parameter_pools(sim_cfg.parameter_pools) :
        sim_cfg.parameter_pools
    charge_training_cfg = enable_charge_training ?
        AWHGrads.ChargeTrainingConfig(enabled=true) :
        AWHGrads.ChargeTrainingConfig(enabled=false)

    sim_cfg = AWHGrads.simulation_config_with(
        sim_cfg;
        cycle=solvent_only_cycle_from_example(sim_cfg.cycle),
        md_time_production=production_time,
        production_log_interval=production_log_interval,
        parameter_pools=parameter_pools,
        charge_training=charge_training_cfg,
        ensemble_eval=AWHGrads.EnsembleEvalConfig(
            threads=min(Threads.nthreads(), 8),
            lambda_tile=4,
            schedule=:dynamic,
            cache_unitless_frames=true,
            cache_unitless_templates=true,
            progress=true,
            progress_interval_seconds=10.0,
        ),
    )

    opt_cfg = AWHGrads.optimization_config_with(
        opt_cfg;
        max_macro_epochs=1,
        max_inner_epochs=1,
        optimize_solvent=false,
    )

    bench_cfg = (
        awh_time=awh_time,
        awh_steps=awh_steps,
        production_time=production_time,
        production_steps=production_steps,
        min_frames=min_frames,
        expected_frames=expected_frames,
        enable_charge_training=enable_charge_training,
    )

    return sim_cfg, opt_cfg, bench_cfg
end

function main(; dry_run::Bool=false, enable_charge_training::Bool=true, run_optimization::Bool=true)
    sim_cfg, opt_cfg, bench_cfg = build_optimization_smoke_configs(; enable_charge_training=enable_charge_training)
    FT = sim_cfg.FT

    println("Solvent ensemble-eval benchmark prepared.")
    println("  AWH time: ", bench_cfg.awh_time, " (", bench_cfg.awh_steps, " steps)")
    println("  Frozen-bias production time: ", bench_cfg.production_time, " (", bench_cfg.production_steps, " steps)")
    println("  Production log interval: ", sim_cfg.production_log_interval)
    println("  Expected logged frames: ", bench_cfg.expected_frames)
    println("  Minimum required frames: ", bench_cfg.min_frames)
    println("  Eval threads: ", sim_cfg.ensemble_eval.threads)
    println("  Eval lambda tile: ", sim_cfg.ensemble_eval.lambda_tile)
    println("  Eval schedule: ", sim_cfg.ensemble_eval.schedule)
    println("  Charge training enabled: ", sim_cfg.charge_training.enabled)
    println("  Optimization phase enabled: ", run_optimization)

    if dry_run
        println("Dry run only; benchmark not executed.")
        return nothing
    end

    log_io = AWHGrads.setup_logging("logs_optimization_smoke.log"; append=false)
    AWHGrads.apply_simulation_config!(sim_cfg)

    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=FT,
    )
    leg = only(cycle_cfg.legs)
    state_schedules_by_leg = Dict{Symbol, AWHGrads.ResolvedLegStateSchedule{FT}}(
        leg.name => AWHGrads.resolve_leg_state_schedule(leg, sim_cfg.lambda_schedule, FT)
        for leg in cycle_cfg.legs
    )
    runtime = AWHGrads.RuntimeState()
    T_coord, T_vol, T_en = AWHGrads.awh_logger_value_types(sim_cfg)
    pstate = AWHGrads.initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)

    awh_by_leg, sys_by_leg = AWHGrads.setup_macro_legs(
        cycle_cfg,
        sim_cfg,
        state_schedules_by_leg,
        runtime,
        pstate.theta_active,
        pstate.idxs_by_leg,
        T_coord,
        T_vol,
        T_en,
        1,
        opt_cfg.restart_rmsd_tol_nm,
    )

    awh_leg = awh_by_leg[leg.name]
    sys_base = sys_by_leg[leg.name]
    idxs = pstate.idxs_by_leg[leg.name]
    beta_val = AWHGrads.default_energy_analysis_type(sim_cfg)(
        1.0 / ustrip(uconvert(sys_base.energy_units, Unitful.R * AWHGrads.T0)),
    )

    println("Running solvent AWH segment...")
    awh_elapsed = @elapsed AWHGrads.simulate!(awh_leg, bench_cfg.awh_steps)
    awh_samples = length(AWHGrads.get_awh_active_idx_history(awh_leg))
    println("  AWH wall time: ", round(awh_elapsed, digits=3), " s")
    println("  AWH sampled λ indices recorded: ", awh_samples)

    println("Running frozen-bias production segment...")
    awh_prod, bias_data = AWHGrads.build_frozen_bias_awh_sim(awh_leg, bench_cfg.production_steps)
    AWHGrads.clear_awh_logger_histories!(awh_leg)
    production_elapsed = @elapsed AWHGrads.simulate!(awh_prod, bench_cfg.production_steps)
    logger_prod = AWHGrads.get_production_logger(awh_prod, "solvent benchmark")
    frame_count = length(logger_prod.active_idx_history)
    λ_states = length(awh_prod.state.partition.λ_atoms)

    println("  Production wall time: ", round(production_elapsed, digits=3), " s")
    println("  Production frames collected: ", frame_count)
    println("  λ states: ", λ_states)
    println("  Parameter count: ", length(pstate.param_names), " total / ", length(pstate.trainable_param_names), " trainable")
    println("  Charge parameter count: ", count(family -> family in (:charge_chi, :charge_eta), pstate.param_families))

    if frame_count < bench_cfg.min_frames
        throw(ArgumentError(
            "Collected only $frame_count production frames for the solvent benchmark; expected at least $(bench_cfg.min_frames). Increase `md_time_production` or lower `production_log_interval`.",
        ))
    end

    neighbors_elapsed = @elapsed neighbors = AWHGrads.precompute_neighbors(logger_prod, awh_prod.state.active_sys)
    cache_elapsed = @elapsed eval_cache = AWHGrads.build_ensemble_eval_cache(
        logger_prod,
        neighbors,
        awh_prod,
        sys_base,
        sim_cfg.ensemble_eval,
    )

    println("Precomputed replay state:")
    println("  Neighbor build time: ", round(neighbors_elapsed, digits=3), " s")
    println("  Eval cache build time: ", round(cache_elapsed, digits=3), " s")
    println("  Cached template count: ", isnothing(eval_cache.template_cache) ? 0 : length(eval_cache.template_cache))

    coords, box = AWHGrads.ensemble_eval_frame_state(eval_cache, 1)
    warmup_grads = zeros(FT, length(pstate.theta_active))
    println("Warming up Enzyme on one cached frame/template...")
    warmup_elapsed = @elapsed warmup_energy = AWHGrads.evaluate_frame_gradients(
        first(eval_cache.template_cache),
        coords,
        box,
        first(neighbors),
        copy(pstate.theta_active),
        warmup_grads,
        idxs...,
    )
    println("  Warmup wall time: ", round(warmup_elapsed, digits=3), " s")
    println("  Warmup energy: ", warmup_energy)

    GC.gc()

    println("Running full gradient ensemble evaluation...")
    eval_stats = @timed AWHGrads.evaluate_ensemble(
        eval_cache,
        pstate.theta_active,
        pstate.param_names,
        idxs...;
        compute_gradients=true,
    )
    energies, gradients = eval_stats.value

    frame_state_evals = frame_count * λ_states
    grad_entry_count = sum(length, values(gradients))
    eval_rate = eval_stats.time > 0 ? frame_state_evals / eval_stats.time : Inf

    println("Gradient ensemble evaluation finished.")
    println("  Eval wall time: ", round(eval_stats.time, digits=3), " s")
    println("  Eval GC time: ", round(eval_stats.gctime, digits=3), " s")
    println("  Eval allocations: ", round(eval_stats.bytes / 2.0^30, digits=3), " GiB")
    println("  Frame-state evaluations: ", frame_state_evals)
    println("  Throughput: ", round(eval_rate, digits=3), " frame-states/s")
    println("  Energy matrix size: ", size(energies))
    println("  Gradient matrices: ", length(gradients))
    println("  Gradient scalar entries: ", grad_entry_count)
    println("  Frozen bias states stored: ", length(bias_data.f))

    optimization_elapsed = 0.0
    optimization_result = nothing
    phi_optimized = nothing
    theta_optimized = nothing

    if run_optimization
        thermo_AT = AWHGrads.default_energy_analysis_type(sim_cfg)
        dG_std_corr = AWHGrads.compute_standard_state_correction(cycle_cfg, thermo_AT)
        p0_energy_per_vol = thermo_AT(ustrip(uconvert(sys_base.energy_units, AWHGrads.P0 * thermo_AT(1.0)u"nm^3" * Unitful.Na)))
        state_schedule = state_schedules_by_leg[leg.name]
        targets = AWHGrads.resolve_training_targets(sim_cfg, cycle_cfg, state_schedules_by_leg)
        leg_artifacts = [
            AWHGrads.LegArtifacts(
                name=leg.name,
                coefficient=FT(leg.coefficient),
                include_pv=leg.include_pv,
                p0_energy_per_vol=leg.include_pv ? p0_energy_per_vol : zero(p0_energy_per_vol),
                n_states=length(state_schedule.lambda),
                coupled_state_idx=state_schedule.coupled_state_idx,
                decoupled_state_idx=state_schedule.decoupled_state_idx,
                awh_prod=awh_prod,
                logger_prod=logger_prod,
                neighbors=neighbors,
                u_ref=energies,
                sys_base=sys_base,
                active_bias=bias_data,
                idxs=idxs,
                eval_cache=eval_cache,
            ),
        ]
        phi_optimized = copy(pstate.phi_active)
        theta_optimized = copy(pstate.theta_active)

        println("Running one optimization phase...")
        optimization_elapsed = @elapsed optimization_result = AWHGrads.run_optimization_phase!(
            phi_optimized,
            theta_optimized,
            leg_artifacts,
            pstate.param_names,
            pstate.trainable_param_names,
            pstate.trainable_param_indices,
            pstate.trainable_position_map,
            pstate.parameter_pools,
            pstate.param_families,
            pstate.theta_ref,
            pstate.theta_min,
            pstate.theta_max,
            pstate.phi_0,
            beta_val,
            dG_std_corr,
            targets,
            opt_cfg,
        )
        println("Optimization phase finished.")
        println("  Optimization wall time: ", round(optimization_elapsed, digits=3), " s")
        println("  Exit reason: ", optimization_result.phase2_exit_reason)
        println("  Best residual: ", optimization_result.best_macro_residual)
        println("  Best epoch: ", optimization_result.best_macro_epoch)
    end

    metrics = (
        leg=leg.name,
        frame_count=frame_count,
        lambda_states=λ_states,
        frame_state_evals=frame_state_evals,
        awh_elapsed_s=awh_elapsed,
        production_elapsed_s=production_elapsed,
        neighbor_elapsed_s=neighbors_elapsed,
        cache_elapsed_s=cache_elapsed,
        warmup_elapsed_s=warmup_elapsed,
        eval_elapsed_s=eval_stats.time,
        eval_gc_s=eval_stats.gctime,
        eval_alloc_bytes=eval_stats.bytes,
        eval_rate=eval_rate,
        charge_training_enabled=sim_cfg.charge_training.enabled,
        optimization_elapsed_s=optimization_elapsed,
        optimization_exit_reason=isnothing(optimization_result) ? nothing : optimization_result.phase2_exit_reason,
        optimized_parameter_count=isnothing(theta_optimized) ? 0 : length(theta_optimized),
    )

    return metrics, log_io
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
