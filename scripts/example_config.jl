base_sim = AWHGrads.default_simulation_config()
base_opt = AWHGrads.default_optimization_config()
lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))

custom_cycle = AWHGrads.ThermodynamicCycleConfig(
    legs=[
        AWHGrads.ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            coefficient=1.0,
            is_vacuum=false,
            include_pv=true,
            probe_time=base_sim.awh_probe_time_solv,
        ),
        AWHGrads.ThermodynamicLegConfig(
            name=:vacuum,
            pdb="ethanol_vac.pdb",
            coefficient=-1.0,
            is_vacuum=true,
            include_pv=false,
            probe_time=base_sim.awh_probe_time_vac,
        ),
    ],
    include_standard_state_correction=true,
    target_dG_kcal_mol=-5.01,
)

sim_cfg = AWHGrads.simulation_config_with(
    base_sim;
    device_id=1,
    lambda_schedule=lambda_schedule,
    cycle=custom_cycle,
    solute_idx=1:9,
    force_field=AWHGrads.ForceFieldConfig(
        xml_files=["tip3p_standard.xml", "gaff.xml", "ethanol.xml"],
    ),
    awh_control=AWHGrads.AWHControlConfig(
        lj_softcore_alpha=0.85,
        coul_softcore_alpha=0.3,
        update_freq=100,
        coverage_threshold=0.8,
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
