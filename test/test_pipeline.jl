using Test
using Molly
using Unitful
# No need for using AWHGrads here as it is included in runtests.jl

@testset "mixed precision defaults" begin
    @test AWHGrads.default_nonbonded_energy_type(Float32) == Float64
    @test AWHGrads.default_nonbonded_energy_type(Float64) == Float64

    sim_cfg = AWHGrads.default_simulation_config()
    @test sim_cfg.FT == Float32
    @test sim_cfg.nonbonded_energy_type == Float64
    @test AWHGrads.effective_nonbonded_energy_type(sim_cfg) == Float64
    @test AWHGrads.default_energy_analysis_type(sim_cfg) == Float64

    _, _, T_en = AWHGrads.awh_logger_value_types(sim_cfg)
    @test T_en == typeof(Float64(1.0)u"kJ * mol^-1")
end

@testset "pipeline orchestration" begin
    FT = Float32

    base_controls = AWHGrads.stage_b_retry_controls(2, 2, 0, FT(1.5), FT(2.0), 8, 10)
    @test base_controls.target_streak == 2
    @test base_controls.cooldown_blocks == 2

    escalated_controls = AWHGrads.stage_b_retry_controls(2, 2, 2, FT(1.5), FT(2.0), 8, 10)
    @test escalated_controls.target_streak == 4
    @test escalated_controls.cooldown_blocks == 8

    capped_controls = AWHGrads.stage_b_retry_controls(2, 2, 5, FT(1.5), FT(2.0), 6, 9)
    @test capped_controls.target_streak == 6
    @test capped_controls.cooldown_blocks == 9

    @test AWHGrads.scaled_probe_frame_cap(0, 5_000_000, 20_000_000) == 0
    @test AWHGrads.scaled_probe_frame_cap(6_000, 5_000_000, 5_000_000) == 6_000
    @test AWHGrads.scaled_probe_frame_cap(6_000, 5_000_000, 7_500_000) == 9_000
    @test AWHGrads.scaled_probe_frame_cap(6_000, 5_000_000, 11_250_000) == 13_500
end

@testset "parameter pool selectors and optimizer metadata" begin
    sim_cfg = AWHGrads.default_simulation_config(AT=Array)
    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=sim_cfg.FT,
    )
    ref_leg = AWHGrads.resolve_parameter_reference_leg(sim_cfg, cycle_cfg)
    ref_method = AWHGrads.resolve_base_nonbonded_method(
        ref_leg.electrostatics_method,
        ref_leg.is_vacuum,
        ref_leg.coulomb_softcore_model,
    )
    sys_ref = Molly.System(
        ref_leg.pdb,
        AWHGrads.ff;
        array_type=Array,
        nonbonded_method=ref_method,
        nonbonded_energy_type=sim_cfg.nonbonded_energy_type,
    )

    pool_cfgs = [
        AWHGrads.ParameterPoolConfig(
            name=:inserted_region,
            atom_indices=collect(1:9),
            max_phi_step=0.25,
        ),
        AWHGrads.ParameterPoolConfig(
            name=:background_oxygen,
            residue_names=["HOH"],
            atom_types=["tip3p-O"],
            max_phi_step=0.035,
            max_sigma_drift=0.03,
            max_epsilon_drift=0.08,
        ),
    ]
    memberships = AWHGrads.resolve_system_parameter_pool_names(sys_ref, pool_cfgs)
    oxygen_indices = findall(ad -> ad.res_name == "HOH" && ad.atom_type == "tip3p-O", sys_ref.atoms_data)
    hydrogen_indices = findall(ad -> ad.res_name == "HOH" && ad.atom_type == "tip3p-H", sys_ref.atoms_data)

    @test all(memberships[1:9] .== Ref(:inserted_region))
    @test !isempty(oxygen_indices)
    @test all(memberships[oxygen_indices] .== Ref(:background_oxygen))
    @test all(isnothing, memberships[hydrogen_indices])

    sim_cfg_joint = AWHGrads.simulation_config_with(
        sim_cfg;
        parameter_pools=pool_cfgs,
        parameter_reference_leg=:solvent,
    )
    pstate = AWHGrads.initialize_parameter_state(
        sim_cfg_joint,
        AWHGrads.default_optimization_config(FT=sim_cfg.FT),
        cycle_cfg,
    )

    @test any(contains("pool_inserted_region"), pstate.param_names)
    @test any(contains("pool_background_oxygen"), pstate.param_names)
    @test !any(contains("tip3p-H"), pstate.param_names)

    active_pools = AWHGrads.optimization_active_pools(
        pstate.trainable_position_map,
        pstate.parameter_pools,
    )
    @test [pool.name for pool in active_pools] == [:inserted_region, :background_oxygen]
    @test all(!isempty(pool.global_indices) for pool in active_pools)
end

@testset "parameter initialization classifies hydrogen atom types from metadata" begin
    sim_cfg = AWHGrads.default_simulation_config(AT=Array)
    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=sim_cfg.FT,
    )
    ref_leg = AWHGrads.resolve_parameter_reference_leg(sim_cfg, cycle_cfg)
    ref_method = AWHGrads.resolve_base_nonbonded_method(
        ref_leg.electrostatics_method,
        ref_leg.is_vacuum,
        ref_leg.coulomb_softcore_model,
    )
    sys_ref = Molly.System(
        ref_leg.pdb,
        AWHGrads.ff;
        array_type=Array,
        nonbonded_method=ref_method,
        nonbonded_energy_type=sim_cfg.nonbonded_energy_type,
    )

    water_h = first(ad for ad in sys_ref.atoms_data if ad.atom_type == "tip3p-H")
    water_o = first(ad for ad in sys_ref.atoms_data if ad.atom_type == "tip3p-O")
    @test AWHGrads.atom_data_is_hydrogen(water_h)
    @test !AWHGrads.atom_data_is_hydrogen(water_o)
    @test AWHGrads.atom_data_is_hydrogen((; atom_type="tip3p-H", atom_name="", element=""))

    pool_cfgs = [
        AWHGrads.ParameterPoolConfig(
            name=:inserted_region,
            atom_indices=collect(1:9),
            max_phi_step=0.25,
        ),
        AWHGrads.ParameterPoolConfig(
            name=:background_water,
            residue_names=["HOH"],
            atom_types=["tip3p-O", "tip3p-H"],
            max_phi_step=0.035,
            max_sigma_drift=0.03,
            max_epsilon_drift=0.08,
        ),
    ]
    sim_cfg_joint = AWHGrads.simulation_config_with(
        sim_cfg;
        parameter_pools=pool_cfgs,
        parameter_reference_leg=:solvent,
    )
    opt_cfg = AWHGrads.default_optimization_config(FT=sim_cfg.FT)
    pstate = AWHGrads.initialize_parameter_state(sim_cfg_joint, opt_cfg, cycle_cfg)
    bounds_cfg = sim_cfg_joint.parameter_bounds

    h_sigma_idx = findfirst(==("pool_background_water_atom_tip3p-H_σ"), pstate.param_names)
    h_epsilon_idx = findfirst(==("pool_background_water_atom_tip3p-H_ϵ"), pstate.param_names)
    @test !isnothing(h_sigma_idx)
    @test !isnothing(h_epsilon_idx)
    @test pstate.theta_min[h_sigma_idx] == sim_cfg.FT(bounds_cfg.sigma_hydrogen_min)
    @test pstate.theta_max[h_sigma_idx] == sim_cfg.FT(bounds_cfg.sigma_hydrogen_max)
    @test pstate.theta_min[h_epsilon_idx] == sim_cfg.FT(bounds_cfg.epsilon_hydrogen_min)
    @test pstate.theta_max[h_epsilon_idx] == sim_cfg.FT(bounds_cfg.epsilon_hydrogen_max)
    @test isapprox(pstate.theta_active[h_sigma_idx], pstate.theta_ref[h_sigma_idx]; atol=sim_cfg.FT(1e-4))
    @test isapprox(pstate.theta_active[h_epsilon_idx], pstate.theta_ref[h_epsilon_idx]; atol=sim_cfg.FT(1e-4))
end

@testset "charge-training parameter state and index maps" begin
    sim_cfg = AWHGrads.default_simulation_config(AT=Array)
    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=sim_cfg.FT,
    )

    pool_cfgs = [
        AWHGrads.ParameterPoolConfig(
            name=:inserted_region,
            atom_indices=collect(1:9),
            trainable_families=Symbol[:sigma, :epsilon, :charge_chi, :charge_eta],
            max_phi_step=0.25,
        ),
    ]
    sim_cfg_charge = AWHGrads.simulation_config_with(
        sim_cfg;
        parameter_pools=pool_cfgs,
        parameter_reference_leg=:vacuum,
        charge_training=AWHGrads.ChargeTrainingConfig(enabled=true),
    )
    opt_cfg = AWHGrads.default_optimization_config(FT=sim_cfg.FT)
    pstate = AWHGrads.initialize_parameter_state(sim_cfg_charge, opt_cfg, cycle_cfg)

    @test any(contains("_charge_"), pstate.param_names)
    @test count(family -> family == :charge_chi, pstate.param_families) > 0
    @test count(family -> family == :charge_eta, pstate.param_families) > 0
    @test all(isfinite, pstate.theta_active)
    @test all(isapprox.(pstate.theta_active, pstate.theta_ref; atol=sim_cfg.FT(1e-4)))

    atom_idxs = pstate.idxs_by_leg[:vacuum][1]
    @test any(>(0), atom_idxs.charge_chi[1:9])
    @test any(>(0), atom_idxs.charge_eta[1:9])
    @test length(atom_idxs.molecule_ids) == length(atom_idxs.reference_charges)
    @test !isempty(atom_idxs.molecule_charge_targets)
end

@testset "line search residual tolerance" begin
    FT = Float32
    @test AWHGrads.line_search_noise_tolerance(FT(-5.0), FT(0.1)) == FT(0.5)
    @test AWHGrads.line_search_residual_acceptable(FT(10.0), FT(10.9), FT(0.1))
    @test !AWHGrads.line_search_residual_acceptable(FT(10.0), FT(11.2), FT(0.1))
    @test !AWHGrads.line_search_residual_acceptable(FT(10.0), FT(10.9), FT(0.1), FT(0.8))
    threshold = AWHGrads.line_search_acceptance_threshold(FT(0.011), FT(0.1), FT(0.214))
    @test threshold == AWHGrads.line_search_noise_tolerance(FT(0.011), FT(0.1))
    @test threshold > zero(FT)
    @test AWHGrads.line_search_acceptance_threshold(zero(FT), FT(0.1), FT(1.0)) > zero(FT)
    @test AWHGrads.line_search_residual_acceptable(FT(0.011), FT(0.001), FT(0.1), FT(0.214))
    @test !AWHGrads.line_search_residual_acceptable(FT(0.011), FT(0.005), FT(0.1), FT(0.214))
    @test AWHGrads.default_optimization_config().max_inner_epochs == 10
end

@testset "pipeline stabilization helpers" begin
    FT = Float32

    @test AWHGrads.stage_b_near_pass_scale(:keep, FT(0.5)) == FT(1.0)
    @test AWHGrads.stage_b_near_pass_scale(:shrink, FT(0.5)) == FT(0.5)

    split_only = AWHGrads.StageBStats(
        split_ready=false,
        parity_ready=true,
        endpoint_parity_ready=true,
        support_coverage_ready=true,
    )
    @test AWHGrads.stage_b_is_split_only_candidate(split_only)
    @test !AWHGrads.stage_b_is_split_only_candidate(AWHGrads.StageBStats(
        split_ready=false,
        parity_ready=false,
        endpoint_parity_ready=true,
        support_coverage_ready=true,
    ))

    @test :opt_stageB_health_scaling_enabled ∉ fieldnames(AWHGrads.OptimizationConfig)
    @test :opt_stageB_health_min_scale ∉ fieldnames(AWHGrads.OptimizationConfig)
    @test :opt_stageB_health_comfort_fraction ∉ fieldnames(AWHGrads.OptimizationConfig)

    @test isnothing(AWHGrads.split_half_frame_indices(3, 4))
    first_half, second_half = AWHGrads.split_half_frame_indices(4, 4)
    @test collect(first_half) == [1, 2]
    @test collect(second_half) == [3, 4]
end

@testset "optimization split-half confidence scaling" begin
    FT = Float32
    trainable_param_names = ["atom_c3_σ"]
    energies = FT[
        0.0  2.0;
        0.0  2.0;
        0.0  0.2;
        0.0  0.2;
    ]
    gradients = Dict(
        "atom_c3_σ" => FT[
            0.0  1.0;
            0.0  1.0;
            0.0 -1.0;
            0.0 -1.0;
        ],
    )
    leg = AWHGrads.LegArtifacts(
        name=:solvent,
        coefficient=FT(1.0),
        include_pv=false,
        n_states=2,
        coupled_state_idx=1,
        decoupled_state_idx=2,
        logger_prod=(active_idx_history=[1, 1, 1, 1],),
        u_ref=copy(energies),
        active_bias=(f=zeros(FT, 2), log_rho=zeros(FT, 2)),
    )
    grad_cycle, _ = AWHGrads.compute_leg_endpoint_state(
        leg,
        trainable_param_names,
        gradients,
        energies,
        FT(1.0),
        FT[];
        compute_gradients=true,
    )
    opt_cfg = AWHGrads.optimization_config_with(
        AWHGrads.default_optimization_config(FT=FT);
        optimization_confidence_min_frames=4,
        optimization_confidence_min_scale=FT(0.25),
        optimization_confidence_scale_strength=FT(1.0),
        optimization_confidence_residual_requirement_strength=FT(0.5),
    )
    summary = AWHGrads.compute_optimization_confidence_summary(
        [leg],
        Dict(:solvent => energies),
        Dict(:solvent => gradients),
        Dict(:solvent => FT[]),
        trainable_param_names,
        grad_cycle,
        FT(1.0),
        FT(0.0),
        opt_cfg,
        FT,
    )

    @test summary.enabled
    @test summary.eligible_legs == 1
    @test isempty(summary.skipped_legs)
    @test summary.cycle_disagreement > 0
    @test summary.gradient_disagreement > 0
    @test FT(0.25) <= summary.scale < FT(1.0)
    @test summary.additional_residual_requirement > 0
    @test opt_cfg.kl_target * summary.scale < opt_cfg.kl_target
    @test opt_cfg.max_phi_step_solute * summary.scale < opt_cfg.max_phi_step_solute
end

@testset "Molly dispersion correction unit stripping" begin
    inter = Molly.LJDispersionCorrection(1.25f0u"kJ*nm^3/mol")
    stripped = ustrip(inter)

    @test stripped isa Molly.LJDispersionCorrection
    @test stripped.factor == 1.25f0
end

@testset "Molly System reconstruction bypasses launch_config forwarding" begin
    custom_launch = Molly.CUDALaunchConfig(force_block_y=4)
    sys = Molly.System(
        "ethanol_vac.pdb",
        AWHGrads.ff;
        array_type=Array,
        nonbonded_method=:none,
        launch_config=custom_launch,
    )
    sys_nounits = ustrip(Molly.from_device(sys))

    rebuilt = AWHGrads._rebuild_system_like(
        sys_nounits,
        sys_nounits.atoms,
        sys_nounits.coords,
        sys_nounits.boundary,
        sys_nounits.pairwise_inters,
        sys_nounits.specific_inter_lists,
        sys_nounits.general_inters,
    )

    @test sys_nounits.launch_config == custom_launch
    @test rebuilt.launch_config == Molly.CUDALaunchConfig()
    @test rebuilt.coords == sys_nounits.coords
    @test rebuilt.general_inters == sys_nounits.general_inters
    @test rebuilt.masses == sys_nounits.masses
    @test rebuilt.total_mass == sys_nounits.total_mass
end
