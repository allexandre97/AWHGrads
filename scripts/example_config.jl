#
# Example config file consumed by `scripts/run_alch.jl`. Returning a NamedTuple
# keeps it easy to override either simulation or optimization settings.
#
base_sim = AWHGrads.default_simulation_config()
base_opt = AWHGrads.default_optimization_config()
parameter_pools = [
    AWHGrads.ParameterPoolConfig(
        name=:inserted_region,
        atom_indices=collect(1:9),
        max_phi_step=0.25,
        max_sigma_drift=0.08,
        max_epsilon_drift=0.25,
    ),
    AWHGrads.ParameterPoolConfig(
        name=:background_oxygen,
        residue_names=["HOH"],
        atom_types=["tip3p-O", "tip3p-H"],
        max_phi_step=Float64(base_opt.max_phi_step_solvent),
        max_sigma_drift=0.03,
        max_epsilon_drift=0.08,
    ),
]

# Vacuum keeps the legacy global λ schedule. The solvent leg uses the staged
# default path defined on the cycle itself.
lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
custom_cycle = AWHGrads.default_cycle_config(; target_dG_kcal_mol=-5.01, FT=base_sim.FT)
targets = AWHGrads.AbstractTrainingTarget[
    AWHGrads.CycleFreeEnergyTarget(
        name=:hydration_free_energy,
        target_dG_kcal_mol=custom_cycle.target_dG_kcal_mol,
    ),
    AWHGrads.StateObservableTarget(
        name=:solvent_density,
        leg=:solvent,
        state=:coupled,
        observable=AWHGrads.MassDensityObservable(),
        target_value=0.99815,
        weight=1.0,
        unit_label="g/mL",
    ),
]

sim_cfg = AWHGrads.simulation_config_with(
    base_sim;
    device_id=1,
    lambda_schedule=lambda_schedule,
    nonbonded_energy_type=AWHGrads.default_nonbonded_energy_type(base_sim.FT),
    cycle=custom_cycle,
    targets=targets,
    solute_idx=1:9,
    parameter_pools=parameter_pools,
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
)

(sim_cfg=sim_cfg, opt_cfg=opt_cfg)
