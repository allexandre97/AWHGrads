using Random
using Molly

@testset "readiness helpers" begin
    a, b = AWHGrads.split_half_ranges(10)
    @test collect(a) == collect(1:5)
    @test collect(b) == collect(6:10)

    history = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 1]
    @test AWHGrads.count_full_round_trips(history, 1, 4) >= 1

    low_frac, high_frac = AWHGrads.endpoint_occupancy_fractions(history, 1, 4)
    @test low_frac > 0
    @test high_frac > 0
end

@testset "probe frame selection and thinning" begin
    @test isempty(AWHGrads.probe_frame_indices(0; frame_stride=4, min_frames=10, max_frames=20))

    idxs_stride = AWHGrads.probe_frame_indices(10; frame_stride=3, min_frames=2, max_frames=0)
    @test idxs_stride == [1, 4, 7, 10]

    idxs_min = AWHGrads.probe_frame_indices(10; frame_stride=6, min_frames=8, max_frames=0)
    @test length(idxs_min) >= 8
    @test first(idxs_min) == 1
    @test last(idxs_min) == 10

    idxs_capped = AWHGrads.probe_frame_indices(100; frame_stride=1, min_frames=2, max_frames=7)
    @test length(idxs_capped) <= 7
    @test first(idxs_capped) == 1
    @test last(idxs_capped) == 100

    logger = (
        active_idx_history = collect(1:10),
        coords_history = collect(101:110),
        volume_history = collect(201:210),
        potential_energy_history = collect(301:310),
    )
    logger_subset = AWHGrads.subset_awh_logger_frames(logger, [1, 4, 10])
    @test logger_subset.active_idx_history == [1, 4, 10]
    @test logger_subset.coords_history == [101, 104, 110]
    @test logger_subset.volume_history == [201, 204, 210]
    @test logger_subset.potential_energy_history == [301, 304, 310]
    @test logger.active_idx_history == collect(1:10)

    selection = AWHGrads.select_probe_frame_indices(10; frame_stride=1, min_frames=2, max_frames=0, discard_fraction=0.2)
    @test selection.selected == collect(1:10)
    @test selection.retained == collect(3:10)

    split_a, split_b = AWHGrads.split_half_ranges(length(selection.retained))
    @test selection.retained[split_a] == [3, 4, 5, 6]
    @test selection.retained[split_b] == [7, 8, 9, 10]

    short_selection = AWHGrads.select_probe_frame_indices(2; frame_stride=1, min_frames=2, max_frames=0, discard_fraction=0.5)
    @test short_selection.selected == [1, 2]
    @test short_selection.retained == [2]
    @test length(short_selection.retained) < 2
end

@testset "active index history source priority" begin
    stats_field = Symbol("active_\u03bb")

    mock_from_stats = (
        state = (
            stats = NamedTuple{(stats_field,)}(([1, 2, 3, 2],)),
            active_sys = (loggers = (awh_logger = (active_idx_history = [9, 9],),),),
            state_loggers = Any[],
        ),
    )
    @test AWHGrads.get_awh_active_idx_history(mock_from_stats) == [1, 2, 3, 2]

    mock_from_main_logger = (
        state = (
            stats = NamedTuple{(stats_field,)}((Int[],)),
            active_sys = (loggers = (awh_logger = (active_idx_history = [4, 5, 6],),),),
            state_loggers = Any[],
        ),
    )
    @test AWHGrads.get_awh_active_idx_history(mock_from_main_logger) == [4, 5, 6]

    mock_from_state_loggers = (
        state = (
            stats = NamedTuple{(stats_field,)}((Int[],)),
            active_sys = (loggers = (awh_logger = (active_idx_history = Int[],),),),
            state_loggers = [
                (awh_logger = (active_idx_history = [1, 2],),),
                (awh_logger = (active_idx_history = [3],),),
            ],
        ),
    )
    @test AWHGrads.get_awh_active_idx_history(mock_from_state_loggers) == [1, 2, 3]

    mock_empty = (
        state = (
            stats = NamedTuple{(stats_field,)}((Int[],)),
            active_sys = (loggers = (awh_logger = (active_idx_history = Int[],),),),
            state_loggers = Any[],
        ),
    )
    @test isempty(AWHGrads.get_awh_active_idx_history(mock_empty))
end

@testset "lambda-history ESS" begin
    rng = MersenneTwister(123)
    iid_history = rand(rng, 1:21, 1200)
    iid_ess = AWHGrads.estimate_lambda_history_ess(iid_history, Float64)
    @test iid_ess > 0.75 * length(iid_history)
    @test iid_ess <= length(iid_history)

    blocky_history = vcat(fill(1, 400), fill(21, 400), fill(1, 400), fill(21, 400), fill(1, 400))
    blocky_ess = AWHGrads.estimate_lambda_history_ess(blocky_history, Float64)
    @test blocky_ess < 0.2 * length(blocky_history)

    @test AWHGrads.estimate_lambda_history_ess(fill(7, 100), Float64) == 1.0
end

@testset "Stage A readiness uses lambda ESS" begin
    stats_field = Symbol("active_\u03bb")

    function mock_awh(history; N_eff::Float32, delta::Float32=1f-4)
        return (
            state = (
                in_initial_stage = false,
                N_eff = N_eff,
                stats = NamedTuple{(:stage_history, :max_delta_f_history, stats_field)}((
                    fill(:linear, 20),
                    fill(delta, 20),
                    history,
                )),
                active_sys = (loggers = (awh_logger = (active_idx_history = Int[],),),),
                state_loggers = Any[],
            ),
        )
    end

    high_ess_history = repeat([1, 4], 250)
    high_ess_result = AWHGrads.evaluate_stage_a_readiness(
        mock_awh(high_ess_history; N_eff=0f0),
        1f-3;
        tail_lag=10,
        min_lambda_ess=300,
        min_linear_neff=3000,
        min_round_trips=3,
        endpoint_min_fraction=0.03f0,
        high_idx=4,
    )
    @test high_ess_result.ready
    @test high_ess_result.lambda_ess_ready
    @test high_ess_result.lambda_ess >= 300
    @test high_ess_result.tau_int_est ≈ length(high_ess_history) / high_ess_result.lambda_ess
    @test !high_ess_result.neff_ready

    low_ess_history = vcat(fill(1, 250), fill(4, 250), fill(1, 250), fill(4, 250), fill(1, 250))
    low_ess_result = AWHGrads.evaluate_stage_a_readiness(
        mock_awh(low_ess_history; N_eff=10_000f0),
        1f-3;
        tail_lag=10,
        min_lambda_ess=300,
        min_linear_neff=3000,
        min_round_trips=2,
        endpoint_min_fraction=0.03f0,
        high_idx=4,
    )
    @test low_ess_result.df_ready
    @test low_ess_result.neff_ready
    @test !low_ess_result.lambda_ess_ready
    @test low_ess_result.tau_int_est ≈ length(low_ess_history) / low_ess_result.lambda_ess
    @test !low_ess_result.ready
end

@testset "leg schedule validation and defaults" begin
    fallback_schedule = Float32.(range(1.0, stop=0.0, length=21))
    fallback_leg = AWHGrads.ThermodynamicLegConfig(name=:solvent, pdb="ethanol_solv.pdb")
    fallback_resolved = AWHGrads.resolve_leg_state_schedule(fallback_leg, fallback_schedule, Float32)
    @test fallback_resolved.coulomb == fallback_schedule
    @test fallback_resolved.lj == fallback_schedule
    @test fallback_resolved.coupled_state_idx == 1
    @test fallback_resolved.decoupled_state_idx == 21

    missing_pair_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        coulomb_lambda_schedule=[1.0, 0.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(missing_pair_leg, fallback_schedule, Float32)

    length_mismatch_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        coulomb_lambda_schedule=[1.0, 0.5, 0.0],
        lj_lambda_schedule=[1.0, 0.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(length_mismatch_leg, fallback_schedule, Float32)

    out_of_range_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        coulomb_lambda_schedule=[1.0, -0.1],
        lj_lambda_schedule=[1.0, 0.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(out_of_range_leg, fallback_schedule, Float32)

    too_short_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        coulomb_lambda_schedule=[1.0],
        lj_lambda_schedule=[1.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(too_short_leg, fallback_schedule, Float32)

    cycle_cfg = AWHGrads.default_cycle_config()
    solvent_leg = only(filter(leg -> leg.name == :solvent, cycle_cfg.legs))
    vacuum_leg = only(filter(leg -> leg.name == :vacuum, cycle_cfg.legs))

    @test length(solvent_leg.coulomb_lambda_schedule) == 31
    @test length(solvent_leg.lj_lambda_schedule) == 31
    @test solvent_leg.coulomb_lambda_schedule[1] ≈ 1.0f0
    @test solvent_leg.coulomb_lambda_schedule[11] ≈ 0.0f0
    @test solvent_leg.coulomb_lambda_schedule[end] ≈ 0.0f0
    @test solvent_leg.lj_lambda_schedule[1] ≈ 1.0f0
    @test solvent_leg.lj_lambda_schedule[11] ≈ 1.0f0
    @test solvent_leg.lj_lambda_schedule[end] ≈ 0.0f0
    @test isnothing(vacuum_leg.coulomb_lambda_schedule)
    @test isnothing(vacuum_leg.lj_lambda_schedule)

    staged_resolved = AWHGrads.resolve_leg_state_schedule(solvent_leg, fallback_schedule, Float32)
    @test length(staged_resolved.coulomb) == 31
    @test length(staged_resolved.lj) == 31
    @test staged_resolved.coupled_state_idx == 1
    @test staged_resolved.decoupled_state_idx == 31
end

@testset "ensemble controls and benchmark configs" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    AWHGrads.apply_simulation_config!(sim_cfg)

    nvt_methods = AWHGrads.awh_coupling_methods(false, :nvt)
    @test length(nvt_methods) == 1
    @test first(nvt_methods) isa Molly.VelocityRescaleThermostat

    npt_methods = AWHGrads.awh_coupling_methods(false, :npt)
    @test length(npt_methods) == 2
    @test any(method -> method isa Molly.CRescaleBarostat, npt_methods)

    vacuum_methods = AWHGrads.awh_coupling_methods(true, :npt)
    @test length(vacuum_methods) == 1
    @test first(vacuum_methods) isa Molly.VelocityRescaleThermostat

    baseline_cfg = include(joinpath(@__DIR__, "..", "scripts", "benchmark_config_baseline.jl"))
    baseline_solvent_leg = only(filter(leg -> leg.name == :solvent, baseline_cfg.sim_cfg.cycle.legs))
    @test isnothing(baseline_solvent_leg.coulomb_lambda_schedule)
    @test isnothing(baseline_solvent_leg.lj_lambda_schedule)
    @test baseline_solvent_leg.include_pv

    nvt_cfg = include(joinpath(@__DIR__, "..", "scripts", "benchmark_config_staged_nvt.jl"))
    nvt_solvent_leg = only(filter(leg -> leg.name == :solvent, nvt_cfg.sim_cfg.cycle.legs))
    @test nvt_solvent_leg.ensemble == :nvt
    @test !nvt_solvent_leg.include_pv
    @test length(nvt_solvent_leg.coulomb_lambda_schedule) == 31
    @test length(nvt_solvent_leg.lj_lambda_schedule) == 31
end

@testset "phase timing helpers" begin
    meta_start = AWHGrads.phase_timing_metadata("Stage A Block", "solvent"; md_steps=500)
    @test meta_start.phase == "Stage A Block"
    @test meta_start.leg == "solvent"
    @test meta_start.md_steps == 500
    @test meta_start.md_ns == AWHGrads.steps_to_ns(500)
    @test isnothing(meta_start.wall_s)
    @test isnothing(meta_start.steps_per_s)

    meta_end = AWHGrads.phase_timing_metadata("Stage A Block", "solvent"; md_steps=500, wall_s=2.0)
    @test meta_end.wall_s == 2.0
    @test meta_end.steps_per_s == 250.0

    payload = [1, 2, 3]
    timed_payload = AWHGrads.timed_phase("Test Payload", "vacuum"; md_steps=10) do
        payload
    end
    @test timed_payload.result === payload
    @test payload == [1, 2, 3]
    @test timed_payload.timing.phase == "Test Payload"
    @test timed_payload.timing.leg == "vacuum"
    @test timed_payload.timing.md_steps == 10
    @test timed_payload.timing.md_ns == AWHGrads.steps_to_ns(10)
    @test timed_payload.timing.wall_s !== nothing
    @test timed_payload.timing.wall_s >= 0.0
    @test timed_payload.timing.steps_per_s !== nothing
end

@testset "updated optimization defaults" begin
    sim_cfg = AWHGrads.default_simulation_config()
    @test sim_cfg.awh_probe_discard_fraction == 0.2
    @test sim_cfg.awh_control.update_freq == 100
    @test sim_cfg.awh_control.coverage_threshold == 0.8
    @test sim_cfg.awh_control.significant_weight == 0.1
    @test sim_cfg.awh_control.initial_n_bias == 100

    opt_cfg = AWHGrads.default_optimization_config()
    @test opt_cfg.awh_min_lambda_ess == 300
    @test opt_cfg.awh_parity_tol_kT == Float32(0.25)
    @test opt_cfg.awh_stageB_cooldown_blocks == 2
end

@testset "AWH control plumbing" begin
    awh_control = AWHGrads.AWHControlConfig(
        seed_num_md_steps=10,
        seed_log_freq=100,
        update_freq=123,
        coverage_threshold=0.7,
        significant_weight=0.25,
        initial_n_bias=321,
        well_tempered_factor=Inf,
        coverage_type=:physical,
    )
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    sim_cfg = AWHGrads.simulation_config_with(sim_cfg; awh_control=awh_control)
    AWHGrads.apply_simulation_config!(sim_cfg)

    awh_sim, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_vac.pdb",
        sim_cfg.solute_idx;
        lambda_values=sim_cfg.lambda_schedule,
        awh_control=awh_control,
        is_vacuum=true,
    )

    @test awh_sim.state.N_bias == Float32(awh_control.initial_n_bias)
    @test awh_sim.update_freq == awh_control.update_freq
    @test awh_sim.coverage_threshold == Float32(awh_control.coverage_threshold)
    @test awh_sim.significant_weight == Float32(awh_control.significant_weight)
end
