#!/usr/bin/env julia

# Example end-to-end workflow:
# 1) define a thermodynamic cycle
# 2) build simulation and optimization configs
# 3) run the full AWH + optimization pipeline
#
# Run with Julia 1.11:
#   julia +1.11 scripts/run_alch_full_example.jl

using Unitful

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))

function build_example_configs()
    base_sim = AWHGrads.default_simulation_config()
    base_opt = AWHGrads.default_optimization_config()

    # Explicit global lambda window schedule used by the vacuum leg and as the
    # fallback for any leg that does not provide its own schedule.
    lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
    dense_solvent_lambda_schedule = AWHGrads.dense_solvent_leg_lambda_schedule(base_sim.FT; lambda_scheduler=:ele_scaled)
    cycle_cfg = AWHGrads.default_cycle_config(; target_dG_kcal_mol=base_sim.dG_exp_kcal_mol, FT=base_sim.FT)

    # Ensure the solvent leg in the cycle object matches our desired R&D sampling depth.
    for leg in cycle_cfg.legs
        if leg.name == :solvent
            new_leg = AWHGrads.ThermodynamicLegConfig(
                name=leg.name,
                pdb=leg.pdb,
                coefficient=leg.coefficient,
                is_vacuum=leg.is_vacuum,
                include_pv=leg.include_pv,

                lambda_schedule=dense_solvent_lambda_schedule,
                ensemble=leg.ensemble,
                probe_awh_seed_num_md_steps=1000,
                electrostatics_method=:cutoff,
                lambda_scheduler=:ele_scaled,
                coulomb_softcore_model=:gapsys,
                lj_softcore_model=:gapsys,
            )
            idx = findfirst(l -> l.name == :solvent, cycle_cfg.legs)
            cycle_cfg.legs[idx] = new_leg
        end
    end

    sim_cfg = AWHGrads.simulation_config_with(
        base_sim;
        device_id=1,
        solute_idx=1:9,
        lambda_schedule=lambda_schedule,
        # Keep the project default mixed-precision path: Float32 dynamics with
        # Float64 nonbonded potential-energy accumulation.
        nonbonded_energy_type=AWHGrads.default_nonbonded_energy_type(base_sim.FT),
        awh_budget_time=base_sim.FT(60)u"ns",
        awh_probe_reweight_stride_solv=2,
        awh_probe_reweight_min_frames_solv=2000,
        awh_probe_reweight_max_frames_solv=6000,
        awh_probe_discard_fraction=0.1,
        force_field=AWHGrads.ForceFieldConfig(
            xml_files=["tip3p_standard.xml", "gaff.xml", "ethanol.xml"],
        ),
        awh_control=AWHGrads.AWHControlConfig(
            lj_softcore_alpha=0.85,
            coul_softcore_alpha=0.3,
            seed_num_md_steps=250,
            bias_update_interval_md_steps=25_000,
            stats_log_every_updates=1,
            coverage_threshold=1.0,
            significant_weight=0.1,
            initial_n_bias=100,
            well_tempered_factor=Inf,
            coverage_type=:physical,
        ),
        cycle=cycle_cfg,
        parameter_reference_leg=:solvent,
    )

    opt_cfg = AWHGrads.optimization_config_with(
        base_opt;
        max_macro_epochs=30,
        optimize_solvent=false,
        awh_split_tol_kT=1.0,
        awh_parity_tol_kT=0.5,
        awh_convergence_tol=5e-3,
    )

    return sim_cfg, opt_cfg
end

function main()
    log_io = AWHGrads.setup_logging("logs.log"; append=false)
    sim_cfg, opt_cfg = build_example_configs()
    runtime = AWHGrads.run_pipeline(; sim_cfg=sim_cfg, opt_cfg=opt_cfg)

    println("Run finished.")
    println("  tuned parameter count: ", isnothing(runtime.theta_active) ? 0 : length(runtime.theta_active))
    println("  latent parameter count: ", isnothing(runtime.phi_active) ? 0 : length(runtime.phi_active))

    return runtime, log_io
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
