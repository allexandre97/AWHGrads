#!/usr/bin/env julia

# Example end-to-end workflow:
# 1) define a thermodynamic cycle
# 2) build simulation and optimization configs
# 3) run the full AWH + optimization pipeline
#
# Run with Julia 1.11:
#   julia +1.11 scripts/run_alch_full_example.jl

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))

base_sim = AWHGrads.default_simulation_config()
base_opt = AWHGrads.default_optimization_config()

# Explicit global lambda window schedule used by the vacuum leg and as the
# fallback for any leg that does not provide its own schedule.
lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
cycle_cfg = AWHGrads.default_cycle_config(; target_dG_kcal_mol=base_sim.dG_exp_kcal_mol, FT=base_sim.FT)

sim_cfg = AWHGrads.simulation_config_with(
    base_sim;
    device_id=1,
    solute_idx=1:9,
    lambda_schedule=lambda_schedule,
    force_field=AWHGrads.ForceFieldConfig(
        xml_files=["tip3p_standard.xml", "gaff.xml", "ethanol.xml"],
    ),
    awh_control=AWHGrads.AWHControlConfig(
        lj_softcore_alpha=0.85,
        coul_softcore_alpha=0.3,
        seed_num_md_steps=10,
        seed_log_freq=100,
        update_freq=100,
        coverage_threshold=0.8,
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
)

runtime = AWHGrads.run_pipeline(; sim_cfg=sim_cfg, opt_cfg=opt_cfg)

println("Run finished.")
println("  tuned parameter count: ", isnothing(runtime.theta_active) ? 0 : length(runtime.theta_active))
println("  latent parameter count: ", isnothing(runtime.phi_active) ? 0 : length(runtime.phi_active))
