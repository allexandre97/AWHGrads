#!/usr/bin/env julia

# Smoke-test end-to-end workflow:
# 1) use the same thermodynamic cycle structure as the full example
# 2) relax readiness/Stage B thresholds so the pipeline advances quickly
# 3) confirm readiness, production, and optimization code paths do not crash
#
# This script is intentionally not scientifically meaningful. Use it only as a
# fast preflight before running `scripts/run_alch_full_example.jl`.
#
# Run with Julia 1.11:
#   julia +1.11 scripts/run_alch_full_smoke.jl

using Unitful

include(joinpath(@__DIR__, "run_alch_full_example.jl"))

function smoke_cycle_from_example(cycle_cfg, solvent_probe_time, vacuum_probe_time)
    smoke_legs = map(cycle_cfg.legs) do leg
        probe_time = leg.name == :solvent ? solvent_probe_time : vacuum_probe_time
        return AWHGrads.ThermodynamicLegConfig(
            name=leg.name,
            pdb=leg.pdb,
            coefficient=leg.coefficient,
            is_vacuum=leg.is_vacuum,
            include_pv=leg.include_pv,
            probe_time=probe_time,
            lambda_schedule=leg.lambda_schedule,
            ensemble=leg.ensemble,
            awh_seed_num_md_steps=leg.awh_seed_num_md_steps,
            awh_bias_update_interval_md_steps=leg.awh_bias_update_interval_md_steps,
            probe_awh_seed_num_md_steps=leg.probe_awh_seed_num_md_steps,
            electrostatics_method=leg.electrostatics_method,
            lambda_scheduler=leg.lambda_scheduler,
            coulomb_softcore_model=leg.coulomb_softcore_model,
            lj_softcore_model=leg.lj_softcore_model,
            readiness_policy=leg.readiness_policy,
        )
    end
    return AWHGrads.ThermodynamicCycleConfig(
        legs=smoke_legs,
        include_standard_state_correction=cycle_cfg.include_standard_state_correction,
        target_dG_kcal_mol=cycle_cfg.target_dG_kcal_mol,
    )
end

function build_smoke_configs()
    sim_cfg, opt_cfg = build_example_configs()
    FT = sim_cfg.FT
    solvent_probe_time = FT(0.05)u"ns"
    vacuum_probe_time = FT(0.02)u"ns"
    cycle_cfg = smoke_cycle_from_example(sim_cfg.cycle, solvent_probe_time, vacuum_probe_time)

    sim_cfg = AWHGrads.simulation_config_with(
        sim_cfg;
        awh_budget_time=FT(0.30)u"ns",
        awh_block_time=FT(0.05)u"ns",
        md_time_production=FT(0.02)u"ns",
        production_log_interval=50,
        awh_probe_time_solv=solvent_probe_time,
        awh_probe_time_vac=vacuum_probe_time,
        awh_probe_reweight_stride_solv=10,
        awh_probe_reweight_stride_vac=10,
        awh_probe_reweight_min_frames_solv=20,
        awh_probe_reweight_min_frames_vac=10,
        awh_probe_reweight_max_frames_solv=100,
        awh_probe_reweight_max_frames_vac=50,
        awh_probe_discard_fraction=0.0,
        cycle=cycle_cfg,
    )

    opt_cfg = AWHGrads.optimization_config_with(
        opt_cfg;
        max_macro_epochs=1,
        max_inner_epochs=2,
        optimize_solvent=false,
        awh_split_tol_kT=FT(25.0),
        awh_parity_tol_kT=FT(25.0),
        awh_convergence_tol=FT(0.5),
        awh_min_lambda_ess=10,
        awh_min_linear_neff=10,
        awh_parity_support_threshold=FT(1.0),
        awh_stageB_support_allow_missing=100,
        awh_min_round_trips=0,
        awh_endpoint_target_ratio=FT(0.0),
        awh_stageA_stable_blocks=1,
        awh_stageB_cooldown_blocks=0,
        awh_stageB_near_pass_cooldown_blocks=0,
        awh_stageB_probe_growth_ns=FT(0.0),
        awh_stageB_probe_near_pass_scale=FT(1.0),
        awh_stageB_probe_max_factor=FT(1.0),
        awh_stageA_streak_growth_factor=FT(1.0),
        awh_stageB_cooldown_growth_factor=FT(1.0),
        awh_stageA_max_streak=1,
        awh_stageB_max_cooldown=0,
    )

    return sim_cfg, opt_cfg
end


function main(; dry_run::Bool=false)
    sim_cfg, opt_cfg = build_smoke_configs()

    println("Smoke-test configuration prepared.")
    println("  AWH budget: ", sim_cfg.awh_budget_time)
    println("  AWH block: ", sim_cfg.awh_block_time)
    println("  Production time: ", sim_cfg.md_time_production)
    println("  Max macro epochs: ", opt_cfg.max_macro_epochs)
    println("  Max inner epochs: ", opt_cfg.max_inner_epochs)

    if dry_run
        println("Dry run only; pipeline not executed.")
        return nothing
    end

    log_io = AWHGrads.setup_logging("logs_smoke.log"; append=false)
    runtime = AWHGrads.run_pipeline(; sim_cfg=sim_cfg, opt_cfg=opt_cfg)

    println("Smoke-test run finished.")
    println("  tuned parameter count: ", isnothing(runtime.theta_active) ? 0 : length(runtime.theta_active))
    println("  latent parameter count: ", isnothing(runtime.phi_active) ? 0 : length(runtime.phi_active))

    return runtime, log_io
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
