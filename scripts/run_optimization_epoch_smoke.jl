#!/usr/bin/env julia

# Solvent-leg ensemble-eval benchmark:
# 1) reuse the full-example solvent setup
# 2) run a single solvent AWH segment for 0.5 ns
# 3) clone a frozen-bias production segment for 0.1 ns
# 4) collect at least 500 production frames
# 5) warm up the exact Enzyme gradient path on one cached frame/template
# 6) benchmark `evaluate_ensemble(...; compute_gradients=true)`
#
# This script exists to exercise and profile the expensive optimization replay
# path without waiting for the full readiness + optimization pipeline.
#
# Run with Julia 1.11:
#   julia +1.11 scripts/run_optimization_epoch_smoke.jl

using Unitful

include(joinpath(@__DIR__, "run_alch_full_example.jl"))

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

function build_optimization_smoke_configs()
    sim_cfg, opt_cfg = build_example_configs()
    FT = sim_cfg.FT

    awh_time = FT(0.1)u"ns"
    production_time = FT(0.1)u"ns"
    min_frames = 500
    awh_steps = AWHGrads.time_to_steps_floor(awh_time)
    production_steps = AWHGrads.time_to_steps_floor(production_time)
    production_log_interval = production_interval_for_min_frames(production_steps, min_frames)
    expected_frames = expected_logged_frames(production_steps, production_log_interval)

    sim_cfg = AWHGrads.simulation_config_with(
        sim_cfg;
        cycle=solvent_only_cycle_from_example(sim_cfg.cycle),
        md_time_production=production_time,
        production_log_interval=production_log_interval,
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
    )

    return sim_cfg, opt_cfg, bench_cfg
end

function main(; dry_run::Bool=false)
    sim_cfg, opt_cfg, bench_cfg = build_optimization_smoke_configs()
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
    )

    return metrics, log_io
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
