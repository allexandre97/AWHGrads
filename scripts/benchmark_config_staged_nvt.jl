#
# Staged solvent benchmark config in NVT. The solvent leg disables the pV term.
#
base_sim = AWHGrads.default_simulation_config()
base_opt = AWHGrads.default_optimization_config()

lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
solvent_schedule = AWHGrads.default_solvent_leg_schedules(base_sim.FT)

cycle_cfg = AWHGrads.ThermodynamicCycleConfig(
    legs=[
        AWHGrads.ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            coefficient=1.0,
            is_vacuum=false,
            include_pv=false,
            probe_time=base_sim.awh_probe_time_solv,
            coulomb_lambda_schedule=solvent_schedule.coulomb_lambda_schedule,
            lj_lambda_schedule=solvent_schedule.lj_lambda_schedule,
            ensemble=:nvt,
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
    target_dG_kcal_mol=base_sim.dG_exp_kcal_mol,
)

sim_cfg = AWHGrads.simulation_config_with(
    base_sim;
    device_id=1,
    lambda_schedule=lambda_schedule,
    cycle=cycle_cfg,
    solute_idx=1:9,
    force_field=AWHGrads.ForceFieldConfig(
        xml_files=["tip3p_standard.xml", "gaff.xml", "ethanol.xml"],
    ),
)

opt_cfg = AWHGrads.optimization_config_with(
    base_opt;
    optimize_solvent=false,
)

(sim_cfg=sim_cfg, opt_cfg=opt_cfg)
