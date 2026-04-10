#
# Baseline benchmark config: both solvent and vacuum use the legacy global
# λ schedule defined on `sim_cfg.lambda_schedule`.
#
base_sim = AWHGrads.default_simulation_config()
base_opt = AWHGrads.default_optimization_config()

lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))

cycle_cfg = AWHGrads.ThermodynamicCycleConfig(
    legs=[
        AWHGrads.ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            coefficient=1.0,
            is_vacuum=false,
            include_pv=true,
            probe_time=base_sim.awh_probe_time_solv,
            ensemble=:npt,
            electrostatics_method=:pme,
            coulomb_softcore_model=:gapsys,
            lj_softcore_model=:gapsys,
        ),
        AWHGrads.ThermodynamicLegConfig(
            name=:vacuum,
            pdb="ethanol_vac.pdb",
            coefficient=-1.0,
            is_vacuum=true,
            include_pv=false,
            probe_time=base_sim.awh_probe_time_vac,
            electrostatics_method=:none,
            coulomb_softcore_model=:gapsys,
            lj_softcore_model=:gapsys,
        ),
    ],
    include_standard_state_correction=true,
    target_dG_kcal_mol=base_sim.dG_exp_kcal_mol,
)

sim_cfg = AWHGrads.simulation_config_with(
    base_sim;
    device_id=1,
    lambda_schedule=lambda_schedule,
    nonbonded_energy_type=AWHGrads.default_nonbonded_energy_type(base_sim.FT),
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
