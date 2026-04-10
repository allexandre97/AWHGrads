#
# Example config file consumed by `scripts/run_alch.jl`. Returning a NamedTuple
# keeps it easy to override either simulation or optimization settings.
#
base_sim = AWHGrads.default_simulation_config()
base_opt = AWHGrads.default_optimization_config()

# Vacuum keeps the legacy global λ schedule. The solvent leg uses the staged
# default path defined on the cycle itself.
lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
custom_cycle = AWHGrads.default_cycle_config(; target_dG_kcal_mol=-5.01, FT=base_sim.FT)

sim_cfg = AWHGrads.simulation_config_with(
    base_sim;
    device_id=1,
    lambda_schedule=lambda_schedule,
    nonbonded_energy_type=AWHGrads.default_nonbonded_energy_type(base_sim.FT),
    cycle=custom_cycle,
    solute_idx=1:9,
    force_field=AWHGrads.ForceFieldConfig(
        xml_files=["tip3p_standard.xml", "gaff.xml", "ethanol.xml"],
    ),
    awh_control=AWHGrads.AWHControlConfig(
        lj_softcore_alpha=1.5,
        coul_softcore_alpha=0.3,
        bias_update_interval_md_steps=1000,
        stats_log_every_updates=1,
        coverage_threshold=1.0,
        significant_weight=0.1,
        initial_n_bias=100,
    ),
)

opt_cfg = AWHGrads.optimization_config_with(
    base_opt;
    max_macro_epochs=30,
    optimize_solvent=false,
)

(sim_cfg=sim_cfg, opt_cfg=opt_cfg)
