using Test
using Molly
using Unitful

function build_observable_eval_fixture(; prod_steps::Int=1)
    FT = Float32
    lambda_values = FT[1.0, 0.5, 0.0]
    vacuum_leg = AWHGrads.ThermodynamicLegConfig(
        name=:vacuum,
        pdb="ethanol_vac.pdb",
        coefficient=1.0,
        is_vacuum=true,
        include_pv=false,
        probe_time=FT(0.01)u"ns",
        lambda_schedule=lambda_values,
        electrostatics_method=:none,
        coulomb_softcore_model=:gapsys,
        lj_softcore_model=:gapsys,
    )
    cycle_cfg = AWHGrads.ThermodynamicCycleConfig(
        legs=[vacuum_leg],
        include_standard_state_correction=false,
        target_dG_kcal_mol=-5.01,
    )
    sim_cfg = AWHGrads.simulation_config_with(
        AWHGrads.default_simulation_config(; FT=FT, AT=Array);
        cycle=cycle_cfg,
        parameter_reference_leg=:vacuum,
        ensemble_eval=AWHGrads.EnsembleEvalConfig(
            threads=max(1, min(Threads.nthreads(), 2)),
            lambda_tile=2,
            schedule=:dynamic,
            cache_unitless_frames=true,
            cache_unitless_templates=true,
        ),
    )
    opt_cfg = AWHGrads.default_optimization_config(; FT=FT)

    AWHGrads.apply_simulation_config!(sim_cfg)
    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=FT,
    )
    leg = only(cycle_cfg.legs)
    state_schedule = AWHGrads.resolve_leg_state_schedule(leg, sim_cfg.lambda_schedule, FT)
    pstate = AWHGrads.initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)
    T_coord, T_vol, T_en = AWHGrads.awh_logger_value_types(sim_cfg)
    logger = Molly.AWHEnsembleLogger(T_coord, T_vol, T_en, 1)

    awh_sim, sys_base = AWHGrads.setup_alchemical_awh(
        leg.pdb,
        sim_cfg.solute_idx;
        lambda_values=state_schedule.lambda,
        awh_control=sim_cfg.awh_control,
        is_vacuum=leg.is_vacuum,
        ensemble=leg.ensemble,
        logger=logger,
        optimized_params=pstate.theta_active,
        param_idxs=pstate.idxs_by_leg[leg.name],
        electrostatics_method=leg.electrostatics_method,
        lambda_scheduler=leg.lambda_scheduler,
        coulomb_softcore_model=leg.coulomb_softcore_model,
        lj_softcore_model=leg.lj_softcore_model,
        array_type=Array,
        nonbonded_energy_type=sim_cfg.nonbonded_energy_type,
    )

    awh_prod, bias_data = AWHGrads.build_frozen_bias_awh_sim(awh_sim, prod_steps)
    simulate!(awh_prod, prod_steps)

    logger_prod = AWHGrads.get_production_logger(awh_prod, "observable test")
    neighbors = AWHGrads.precompute_neighbors(logger_prod, awh_prod.state.active_sys)

    return (
        sim_cfg=sim_cfg,
        opt_cfg=opt_cfg,
        cycle_cfg=cycle_cfg,
        state_schedule=state_schedule,
        pstate=pstate,
        awh_prod=awh_prod,
        bias_data=bias_data,
        logger_prod=logger_prod,
        neighbors=neighbors,
        sys_base=sys_base,
    )
end

@testset "training target resolution" begin
    sim_cfg = AWHGrads.default_simulation_config(AT=Array)
    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=sim_cfg.FT,
    )
    state_schedules_by_leg = Dict(
        leg.name => AWHGrads.resolve_leg_state_schedule(leg, sim_cfg.lambda_schedule, sim_cfg.FT)
        for leg in cycle_cfg.legs
    )

    default_targets = AWHGrads.resolve_training_targets(sim_cfg, cycle_cfg, state_schedules_by_leg)
    @test length(default_targets) == 1
    @test only(default_targets) isa AWHGrads.ResolvedCycleFreeEnergyTarget
    @test only(default_targets).target_dG_kcal_mol == cycle_cfg.target_dG_kcal_mol
    @test isnothing(only(default_targets).tolerance_kcal_mol)

    explicit_targets = AWHGrads.AbstractTrainingTarget[
        AWHGrads.CycleFreeEnergyTarget(name=:hydration_free_energy, tolerance_kcal_mol=0.25),
        AWHGrads.StateObservableTarget(
            name=:solvent_density,
            leg=:solvent,
            state=:coupled,
            observable=AWHGrads.MassDensityObservable(),
            target_value=0.99815,
            tolerance=0.02,
            unit_label="g/mL",
        ),
        AWHGrads.StateObservableTarget(
            name=:solvent_dielectric,
            leg=:solvent,
            state=:coupled,
            observable=AWHGrads.DielectricConstantObservable(),
            target_value=80.35,
            tolerance=0.5,
            unit_label="epsilon_r",
        ),
    ]
    sim_cfg_targets = AWHGrads.simulation_config_with(
        sim_cfg;
        cycle=cycle_cfg,
        targets=explicit_targets,
    )
    resolved_targets = AWHGrads.resolve_training_targets(
        sim_cfg_targets,
        cycle_cfg,
        state_schedules_by_leg,
    )

    @test length(resolved_targets) == 3
    cycle_target = only(filter(target -> target isa AWHGrads.ResolvedCycleFreeEnergyTarget, resolved_targets))
    @test cycle_target.tolerance_kcal_mol == 0.25
    density_target = only(filter(target -> target isa AWHGrads.ResolvedStateObservableTarget && target.name == :solvent_density, resolved_targets))
    @test density_target.leg == :solvent
    @test density_target.state_idx == state_schedules_by_leg[:solvent].coupled_state_idx
    @test density_target.unit_label == "g/mL"
    @test density_target.target_value == 0.99815
    @test density_target.tolerance == 0.02
    dielectric_target = only(filter(target -> target isa AWHGrads.ResolvedStateObservableTarget && target.name == :solvent_dielectric, resolved_targets))
    @test dielectric_target.leg == :solvent
    @test dielectric_target.state_label == :coupled
    @test dielectric_target.target_value == 80.35
    @test dielectric_target.tolerance == 0.5
    @test dielectric_target.unit_label == "epsilon_r"
end

@testset "density example config uses generic targets" begin
    example_cfg = include(joinpath(@__DIR__, "..", "scripts", "example_config.jl"))
    @test !isnothing(example_cfg.sim_cfg.targets)

    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(example_cfg.sim_cfg);
        default_lambda_schedule=example_cfg.sim_cfg.lambda_schedule,
        FT=example_cfg.sim_cfg.FT,
    )
    state_schedules_by_leg = Dict(
        leg.name => AWHGrads.resolve_leg_state_schedule(leg, example_cfg.sim_cfg.lambda_schedule, example_cfg.sim_cfg.FT)
        for leg in cycle_cfg.legs
    )
    resolved_targets = AWHGrads.resolve_training_targets(
        example_cfg.sim_cfg,
        cycle_cfg,
        state_schedules_by_leg,
    )

    @test length(resolved_targets) == 3
    @test any(target -> target isa AWHGrads.ResolvedCycleFreeEnergyTarget, resolved_targets)
    density_target = only(filter(target -> target isa AWHGrads.ResolvedStateObservableTarget && target.name == :solvent_density, resolved_targets))
    @test density_target.name == :solvent_density
    @test density_target.leg == :solvent
    @test density_target.state_label == :coupled
    @test density_target.target_value == 0.99815
    dielectric_target = only(filter(target -> target isa AWHGrads.ResolvedStateObservableTarget && target.name == :solvent_dielectric, resolved_targets))
    @test dielectric_target.leg == :solvent
    @test dielectric_target.state_label == :coupled
    @test dielectric_target.target_value == 80.35
end

@testset "target tolerances default to 0.1 percent with floors" begin
    opt_cfg = AWHGrads.default_optimization_config(FT=Float64)
    cycle_target = AWHGrads.ResolvedCycleFreeEnergyTarget(
        :cycle_free_energy,
        -5.01,
        nothing,
        1.0,
    )
    cycle_tol = AWHGrads.target_tolerance_value(cycle_target, 1.0, opt_cfg, Float64)
    @test isapprox(cycle_tol, 0.001 * abs(cycle_target.target_dG_kcal_mol) * 4.184; rtol=1e-8)

    zero_observable_target = AWHGrads.ResolvedStateObservableTarget(
        :zero_observable,
        :solvent,
        1,
        :coupled,
        AWHGrads.MassDensityObservable(),
        0.0,
        nothing,
        1.0,
        "g/mL",
    )
    zero_tol = AWHGrads.target_tolerance_value(zero_observable_target, 1.0, opt_cfg, Float64)
    @test zero_tol == opt_cfg.observable_target_absolute_tolerance
end

@testset "normalized target losses match across scales" begin
    opt_cfg = AWHGrads.default_optimization_config(FT=Float64)
    observable_target_small = AWHGrads.ResolvedStateObservableTarget(
        :small_scale,
        :solvent,
        1,
        :coupled,
        AWHGrads.MassDensityObservable(),
        1.0,
        nothing,
        1.0,
        "g/mL",
    )
    observable_target_large = AWHGrads.ResolvedStateObservableTarget(
        :large_scale,
        :solvent,
        1,
        :coupled,
        AWHGrads.MassDensityObservable(),
        100.0,
        nothing,
        1.0,
        "arb",
    )

    loss_small = AWHGrads.evaluate_target_loss(
        observable_target_small,
        1.04,
        AWHGrads.target_reference_value(observable_target_small, 1.0, Float64),
        1.0,
        opt_cfg,
        Float64,
    )
    loss_large = AWHGrads.evaluate_target_loss(
        observable_target_large,
        104.0,
        AWHGrads.target_reference_value(observable_target_large, 1.0, Float64),
        1.0,
        opt_cfg,
        Float64,
    )

    @test isapprox(loss_small.normalized_residual, 40.0; atol=1e-10)
    @test isapprox(loss_large.normalized_residual, 40.0; atol=1e-10)
    @test isapprox(loss_small.loss, loss_large.loss; atol=1e-10)
end

@testset "mass density observable returns expected ethanol-water density" begin
    sim_cfg = AWHGrads.default_simulation_config(AT=Array)
    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=sim_cfg.FT,
    )
    solvent_leg = only(filter(leg -> leg.name == :solvent, cycle_cfg.legs))
    ref_method = AWHGrads.resolve_base_nonbonded_method(
        solvent_leg.electrostatics_method,
        solvent_leg.is_vacuum,
        solvent_leg.coulomb_softcore_model,
    )
    sys = Molly.System(
        solvent_leg.pdb,
        AWHGrads.ff;
        array_type=Array,
        nonbonded_method=ref_method,
        nonbonded_energy_type=sim_cfg.nonbonded_energy_type,
    )

    density = AWHGrads.MassDensityObservable()(sys)
    @test isapprox(density, 0.97858413; rtol=1e-5, atol=1e-6)
end

@testset "dielectric constant observable estimator matches dipole-fluctuation formula" begin
    FT = Float64
    trainable_param_names = ["atom_c3_σ"]
    leg = AWHGrads.LegArtifacts(
        name=:solvent,
        coefficient=FT(1.0),
        include_pv=false,
        n_states=2,
        coupled_state_idx=1,
        decoupled_state_idx=2,
        logger_prod=(active_idx_history=[1, 1],),
        u_ref=FT[
            0.0  1.0;
            0.0  1.0;
        ],
        active_bias=(f=zeros(FT, 2), log_rho=zeros(FT, 2)),
    )
    observable = AWHGrads.DielectricConstantObservable()
    energies = copy(leg.u_ref)
    energy_gradients_phi = Dict(
        "atom_c3_σ" => zeros(FT, size(energies)),
    )
    dipole_sq_values = FT[1.0, 9.0]
    dipole_sq_gradients_phi = Dict("atom_c3_σ" => FT[0.0, 0.0])
    dipole_component_values = (
        FT[1.0, 3.0],
        FT[0.0, 0.0],
        FT[0.0, 0.0],
    )
    dipole_component_gradients_phi = (
        Dict("atom_c3_σ" => FT[0.0, 0.0]),
        Dict("atom_c3_σ" => FT[0.0, 0.0]),
        Dict("atom_c3_σ" => FT[0.0, 0.0]),
    )
    volumes = FT[2.0, 4.0]
    log_mixture_denom = AWHGrads.compute_leg_log_mixture_denom(leg, FT(1.0), FT[])

    dielectric_grad, dielectric_prediction, state_ess = AWHGrads.compute_leg_dielectric_constant_estimate(
        leg,
        observable,
        trainable_param_names,
        energy_gradients_phi,
        dipole_sq_values,
        dipole_sq_gradients_phi,
        dipole_component_values,
        dipole_component_gradients_phi,
        energies,
        log_mixture_denom,
        1,
        FT(1.0),
        volumes;
        compute_gradients=true,
    )

    expected_prediction = 1 + (4 * π / 3) * observable.coulomb_const * ((5.0 - 4.0) / 3.0)
    @test isapprox(dielectric_prediction, expected_prediction; rtol=1e-12, atol=1e-12)
    @test state_ess == 2.0
    @test dielectric_grad == [0.0]
end

@testset "replayed Molly.System observables compile and differentiate" begin
    fixture = build_observable_eval_fixture(; prod_steps=1)
    leg = only(fixture.cycle_cfg.legs)
    idxs = fixture.pstate.idxs_by_leg[leg.name]
    cache = AWHGrads.build_ensemble_eval_cache(
        fixture.logger_prod,
        fixture.neighbors,
        fixture.awh_prod,
        fixture.sys_base,
        fixture.sim_cfg.ensemble_eval,
    )

    state_idx = fixture.state_schedule.coupled_state_idx
    template = AWHGrads.ensemble_eval_template(cache, state_idx)
    coords, box = AWHGrads.ensemble_eval_frame_state(cache, 1)
    observable = sys -> sum(atom.σ for atom in sys.atoms)

    params = copy(fixture.pstate.theta_active)
    grads = zeros(eltype(params), length(params))
    obs_value = AWHGrads.evaluate_frame_observable_gradients(
        observable,
        template,
        coords,
        box,
        first(fixture.neighbors),
        params,
        grads,
        idxs...,
    )

    @test isfinite(obs_value)
    @test all(isfinite, grads)

    dipole_observable = sys -> sum(abs2, Molly.dipole_moment(sys))
    dipole_grads = zeros(eltype(params), length(params))
    dipole_value = AWHGrads.evaluate_frame_observable_gradients(
        dipole_observable,
        template,
        coords,
        box,
        first(fixture.neighbors),
        params,
        dipole_grads,
        idxs...,
    )

    @test isfinite(dipole_value)
    @test all(isfinite, dipole_grads)

    sigma_idx = findfirst(name -> occursin("_σ", name), fixture.pstate.param_names)
    @test !isnothing(sigma_idx)

    eps = eltype(params)(1e-3)
    params_hi = copy(params)
    params_lo = copy(params)
    params_hi[sigma_idx] += eps
    params_lo[sigma_idx] -= eps

    obs_hi = AWHGrads.evaluate_frame_observable(
        params_hi,
        observable,
        template,
        coords,
        box,
        first(fixture.neighbors),
        idxs...,
    )
    obs_lo = AWHGrads.evaluate_frame_observable(
        params_lo,
        observable,
        template,
        coords,
        box,
        first(fixture.neighbors),
        idxs...,
    )
    fd_grad = (obs_hi - obs_lo) / (2 * eps)
    @test isapprox(grads[sigma_idx], fd_grad; rtol=5e-2, atol=5e-3)

    values, value_grads = AWHGrads.evaluate_state_observable(
        cache,
        fixture.pstate.theta_active,
        fixture.pstate.param_names,
        observable,
        state_idx,
        idxs...;
        compute_gradients=true,
    )
    @test length(values) == length(fixture.logger_prod.active_idx_history)
    @test haskey(value_grads, fixture.pstate.param_names[sigma_idx])
    @test all(isfinite, values)
    @test all(isfinite, value_grads[fixture.pstate.param_names[sigma_idx]])
end

@testset "state observable targets support non-gradient proposal evaluation" begin
    fixture = build_observable_eval_fixture(; prod_steps=1)
    leg_cfg = only(fixture.cycle_cfg.legs)
    idxs = fixture.pstate.idxs_by_leg[leg_cfg.name]
    eval_cache = AWHGrads.build_ensemble_eval_cache(
        fixture.logger_prod,
        fixture.neighbors,
        fixture.awh_prod,
        fixture.sys_base,
        fixture.sim_cfg.ensemble_eval,
    )

    params = copy(fixture.pstate.theta_active)
    u_ref, _ = AWHGrads.evaluate_ensemble(
        eval_cache,
        params,
        fixture.pstate.param_names,
        idxs...;
        compute_gradients=false,
    )
    leg = AWHGrads.LegArtifacts(
        name=leg_cfg.name,
        coefficient=fixture.sim_cfg.FT(leg_cfg.coefficient),
        include_pv=leg_cfg.include_pv,
        p0_energy_per_vol=zero(fixture.sim_cfg.FT),
        n_states=length(fixture.state_schedule.lambda),
        coupled_state_idx=fixture.state_schedule.coupled_state_idx,
        decoupled_state_idx=fixture.state_schedule.decoupled_state_idx,
        logger_prod=fixture.logger_prod,
        u_ref=u_ref,
        active_bias=fixture.bias_data,
        idxs=idxs,
        eval_cache=eval_cache,
    )
    observable = sys -> sum(atom.σ for atom in sys.atoms)
    target = AWHGrads.ResolvedStateObservableTarget(
        :sigma_sum,
        leg_cfg.name,
        fixture.state_schedule.coupled_state_idx,
        :coupled,
        observable,
        0.0,
        nothing,
        1.0,
        "arb",
    )
    log_mixture_denom = AWHGrads.compute_leg_log_mixture_denom(leg, fixture.sim_cfg.FT(1.0), fixture.sim_cfg.FT[])

    gradient, prediction, state_ess, cache = AWHGrads.evaluate_resolved_state_observable_target(
        target,
        leg,
        u_ref,
        Dict{String, Matrix{fixture.sim_cfg.FT}}(),
        log_mixture_denom,
        fixture.sim_cfg.FT[],
        params,
        fixture.pstate.param_names,
        fixture.pstate.trainable_param_names,
        nothing,
        fixture.sim_cfg.FT(1.0);
        compute_gradients=false,
    )

    @test cache isa AWHGrads.ScalarStateObservableCache
    @test length(gradient) == length(fixture.pstate.trainable_param_names)
    @test all(iszero, gradient)
    @test isfinite(prediction)
    @test isfinite(state_ess)
    @test state_ess > zero(state_ess)
    @test_throws ArgumentError AWHGrads.evaluate_resolved_state_observable_target(
        target,
        leg,
        u_ref,
        Dict{String, Matrix{fixture.sim_cfg.FT}}(),
        log_mixture_denom,
        fixture.sim_cfg.FT[],
        params,
        fixture.pstate.param_names,
        fixture.pstate.trainable_param_names,
        nothing,
        fixture.sim_cfg.FT(1.0);
        compute_gradients=true,
    )
end
