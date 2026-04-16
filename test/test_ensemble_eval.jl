using Test
using Molly
using Unitful

function build_vacuum_eval_fixture(; prod_steps::Int=20)
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

    logger_prod = AWHGrads.get_production_logger(awh_prod, "vacuum test")
    neighbors = AWHGrads.precompute_neighbors(logger_prod, awh_prod.state.active_sys)
    beta_val = AWHGrads.default_energy_analysis_type(sim_cfg)(
        1.0 / ustrip(uconvert(sys_base.energy_units, Unitful.R * AWHGrads.T0)),
    )

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
        beta_val=beta_val,
    )
end

@testset "ensemble evaluator energy cache and tile equivalence" begin
    fixture = build_vacuum_eval_fixture()
    leg = only(fixture.cycle_cfg.legs)
    idxs = fixture.pstate.idxs_by_leg[leg.name]
    num_lambda = length(fixture.state_schedule.lambda)

    full_tile_cfg = AWHGrads.EnsembleEvalConfig(
        threads=1,
        lambda_tile=num_lambda,
        schedule=:static,
        cache_unitless_frames=true,
        cache_unitless_templates=true,
    )
    tiled_cfg = AWHGrads.EnsembleEvalConfig(
        threads=max(1, min(Threads.nthreads(), 2)),
        lambda_tile=1,
        schedule=:dynamic,
        cache_unitless_frames=true,
        cache_unitless_templates=true,
    )

    cache_full = AWHGrads.build_ensemble_eval_cache(
        fixture.logger_prod,
        fixture.neighbors,
        fixture.awh_prod,
        fixture.sys_base,
        full_tile_cfg,
    )
    u_full, g_full = AWHGrads.evaluate_ensemble(
        cache_full,
        fixture.pstate.theta_active,
        fixture.pstate.param_names,
        idxs...;
        compute_gradients=false,
    )
    cache = AWHGrads.build_ensemble_eval_cache(
        fixture.logger_prod,
        fixture.neighbors,
        fixture.awh_prod,
        fixture.sys_base,
        tiled_cfg,
    )
    u_tiled, g_tiled = AWHGrads.evaluate_ensemble(
        cache,
        fixture.pstate.theta_active,
        fixture.pstate.param_names,
        idxs...;
        compute_gradients=false,
    )

    @test !isnothing(cache_full.frame_cache)
    @test !isnothing(cache_full.template_cache)
    @test size(u_full) == size(u_tiled)
    @test isapprox(u_full, u_tiled; rtol=1e-6, atol=1e-6)
    @test isempty(g_full)
    @test isempty(g_tiled)
end

@testset "ensemble evaluator templates are CPU and unitless before AD" begin
    fixture = build_vacuum_eval_fixture()
    cache = AWHGrads.build_ensemble_eval_cache(
        fixture.logger_prod,
        fixture.neighbors,
        fixture.awh_prod,
        fixture.sys_base,
        fixture.sim_cfg.ensemble_eval,
    )

    @test !isnothing(cache.template_cache)
    @test !isempty(cache.template_cache)

    for template in cache.template_cache
        @test typeof(template) == typeof(AWHGrads.prepare_enzyme_ready_system(template))
        @test Molly.array_type(template.coords) == Array
        @test !(first(first(template.coords)) isa Unitful.AbstractQuantity)
        @test propertynames(template.loggers) == ()
    end
end

@testset "ensemble evaluator single-frame gradients compile on cached template" begin
    fixture = build_vacuum_eval_fixture(; prod_steps=1)
    leg = only(fixture.cycle_cfg.legs)
    idxs = fixture.pstate.idxs_by_leg[leg.name]
    cache = AWHGrads.build_ensemble_eval_cache(
        fixture.logger_prod,
        fixture.neighbors,
        fixture.awh_prod,
        fixture.sys_base,
        fixture.sim_cfg.ensemble_eval,
    )

    coords, box = AWHGrads.ensemble_eval_frame_state(cache, 1)
    params = copy(fixture.pstate.theta_active)
    grads = zeros(eltype(params), length(params))
    energy = AWHGrads.evaluate_frame_gradients(
        first(cache.template_cache),
        coords,
        box,
        first(fixture.neighbors),
        params,
        grads,
        idxs...,
    )

    @test isfinite(energy)
    @test all(isfinite, grads)
end

@testset "optimization smoke script config" begin
    include(joinpath(@__DIR__, "..", "scripts", "run_optimization_epoch_smoke.jl"))
    sim_cfg, opt_cfg, bench_cfg = build_optimization_smoke_configs()
    prod_steps = AWHGrads.time_to_steps_floor(sim_cfg.md_time_production)

    @test length(sim_cfg.cycle.legs) == 1
    @test only(sim_cfg.cycle.legs).name == :solvent
    @test bench_cfg.awh_time == sim_cfg.FT(0.1)u"ns"
    @test sim_cfg.md_time_production == sim_cfg.FT(0.1)u"ns"
    @test sim_cfg.production_log_interval == fld(prod_steps, bench_cfg.min_frames)
    @test bench_cfg.expected_frames >= bench_cfg.min_frames
    @test sim_cfg.ensemble_eval.lambda_tile == 4
    @test sim_cfg.ensemble_eval.progress
    @test sim_cfg.charge_training.enabled
    inserted_region_pool = only(filter(pool -> pool.name == :inserted_region, sim_cfg.parameter_pools))
    @test :charge_chi in inserted_region_pool.trainable_families
    @test :charge_eta in inserted_region_pool.trainable_families
    @test opt_cfg.max_macro_epochs == 1
    @test opt_cfg.max_inner_epochs == 1
end
