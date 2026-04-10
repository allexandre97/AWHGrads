using Logging
using Random
using Molly
using Unitful

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
        n_states = isempty(history) ? 0 : maximum(history)
        return (
            state = (
                in_initial_stage = false,
                N_eff = N_eff,
                f = zeros(Float32, n_states),
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
    @test high_ess_result.switch_count == length(high_ess_history) - 1
    @test high_ess_result.mean_residence == 1.0f0
    @test high_ess_result.median_residence == 1.0f0
    @test high_ess_result.min_state_occupancy == 0.0f0
    @test high_ess_result.low_occupancy_states == [2, 3]

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
    @test low_ess_result.switch_count == 4
    @test low_ess_result.mean_residence == 250.0f0
    @test low_ess_result.median_residence == 250.0f0
    @test !low_ess_result.ready
end

@testset "lambda residence diagnostics" begin
    history = Int[1, 1, 2, 2, 2, 1, 3]
    @test AWHGrads.count_lambda_switches(history) == 3
    @test AWHGrads.contiguous_residence_lengths(history) == [2, 3, 1, 1]

    summary = AWHGrads.residence_length_summary(history, Float32)
    @test summary.switch_count == 3
    @test summary.mean_residence == 1.75f0
    @test summary.median_residence == 1.5f0

    empty_summary = AWHGrads.residence_length_summary(Int[], Float32)
    @test empty_summary.switch_count == 0
    @test empty_summary.mean_residence == 0.0f0
    @test empty_summary.median_residence == 0.0f0
end

@testset "Stage A readiness surfaces low-occupancy states" begin
    stats_field = Symbol("active_\u03bb")
    history = vcat(fill(1, 199), [4])
    mock_awh = (
        state = (
            in_initial_stage = false,
            N_eff = 5_000f0,
            f = zeros(Float32, 4),
            stats = NamedTuple{(:stage_history, :max_delta_f_history, stats_field)}((
                fill(:linear, 20),
                fill(1f-4, 20),
                history,
            )),
            active_sys = (loggers = (awh_logger = (active_idx_history = Int[],),),),
            state_loggers = Any[],
        ),
    )

    result = AWHGrads.evaluate_stage_a_readiness(
        mock_awh,
        1f-3;
        tail_lag=10,
        min_lambda_ess=10,
        min_linear_neff=3000,
        min_round_trips=0,
        endpoint_min_fraction=0.0f0,
        high_idx=4,
    )

    @test result.min_state_occupancy == 0.0f0
    @test result.low_occupancy_states == [2, 3, 4]
end

@testset "Stage A readiness enforces solvent tail and endpoint floors" begin
    stats_field = Symbol("active_\u03bb")

    function mock_awh(history; N_eff::Float32=5_000f0, delta::Float32=1f-4)
        n_states = isempty(history) ? 0 : maximum(history)
        return (
            state = (
                in_initial_stage = false,
                N_eff = N_eff,
                f = zeros(Float32, n_states),
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

    tail_limited_history = repeat([1, 6], 250)
    tail_limited = AWHGrads.evaluate_stage_a_readiness(
        mock_awh(tail_limited_history),
        1f-3;
        tail_lag=10,
        min_lambda_ess=300,
        min_linear_neff=3000,
        min_round_trips=3,
        endpoint_min_fraction=0.02f0,
        tail_state_idxs=[5, 6],
        tail_min_state_occupancy_floor=0.05f0,
        endpoint_high_min_fraction_abs=0.03f0,
        high_idx=6,
    )
    @test !tail_limited.tail_ready
    @test tail_limited.tail_min_state_occupancy == 0.0f0
    @test tail_limited.tail_low_occupancy_states == [5]
    @test !tail_limited.ready

    endpoint_limited_history = vcat(repeat([1, 2, 3, 4, 5], 60), [6, 1, 6, 1, 6, 1])
    endpoint_limited = AWHGrads.evaluate_stage_a_readiness(
        mock_awh(endpoint_limited_history),
        1f-3;
        tail_lag=10,
        min_lambda_ess=10,
        min_linear_neff=3000,
        min_round_trips=2,
        endpoint_min_fraction=0.0f0,
        endpoint_high_min_fraction_abs=0.03f0,
        high_idx=6,
    )
    @test endpoint_limited.endpoint_high < endpoint_limited.endpoint_high_required
    @test endpoint_limited.endpoint_high_required == 0.03f0
    @test !endpoint_limited.endpoint_ready
    @test !endpoint_limited.ready
end

@testset "Stage A readiness uses recent occupancy history and endpoint bands" begin
    stats_field = Symbol("active_\u03bb")

    function mock_awh(history; N_eff::Float32=5_000f0, delta::Float32=1f-4)
        n_states = isempty(history) ? 0 : maximum(history)
        return (
            state = (
                in_initial_stage = false,
                N_eff = N_eff,
                f = zeros(Float32, n_states),
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

    stale_then_recovered_history = vcat(fill(1, 200), repeat([1, 6], 60))
    recent_window_result = AWHGrads.evaluate_stage_a_readiness(
        mock_awh(stale_then_recovered_history),
        1f-3;
        tail_lag=10,
        min_lambda_ess=10,
        min_linear_neff=3000,
        min_round_trips=0,
        endpoint_min_fraction=0.1f0,
        history_window_length=100,
        high_idx=6,
    )
    @test recent_window_result.endpoint_low ≈ 0.5f0
    @test recent_window_result.endpoint_high ≈ 0.5f0
    @test recent_window_result.endpoint_high_required == 0.1f0
    @test recent_window_result.endpoint_ready
    @test recent_window_result.n_hist == length(stale_then_recovered_history)
    @test recent_window_result.n_hist_recent == 100

    band_history = repeat([1, 5, 6, 1], 40)
    band_result = AWHGrads.evaluate_stage_a_readiness(
        mock_awh(band_history),
        1f-3;
        tail_lag=10,
        min_lambda_ess=10,
        min_linear_neff=3000,
        min_round_trips=0,
        endpoint_min_fraction=0.05f0,
        endpoint_state_idxs=[5, 6],
        high_idx=6,
    )
    @test band_result.endpoint_high ≈ 0.5f0
    @test band_result.endpoint_high_required == 0.1f0
    @test band_result.endpoint_ready
end

@testset "leg schedule resolution and defaults" begin
    fallback_schedule = Float32.(range(1.0, stop=0.0, length=21))
    fallback_leg = AWHGrads.ThermodynamicLegConfig(name=:solvent, pdb="ethanol_solv.pdb")
    fallback_resolved = AWHGrads.resolve_leg_state_schedule(fallback_leg, fallback_schedule, Float32)
    @test fallback_resolved.lambda == fallback_schedule
    @test fallback_resolved.coupled_state_idx == 1
    @test fallback_resolved.decoupled_state_idx == 21

    solvent_schedule = AWHGrads.default_solvent_leg_lambda_schedule(Float32)
    @test length(solvent_schedule) == 31
    @test solvent_schedule[1] ≈ 1.0f0
    @test solvent_schedule[11] ≈ 0.5f0
    @test solvent_schedule[12] ≈ 0.45125f0
    @test solvent_schedule[end] ≈ 0.0f0

    dense_solvent_schedule = AWHGrads.dense_solvent_leg_lambda_schedule(Float32)
    dense_solvent_schedule_ele_scaled = AWHGrads.dense_solvent_leg_lambda_schedule(
        Float32;
        lambda_scheduler=:ele_scaled,
    )
    @test length(dense_solvent_schedule) == 21
    @test dense_solvent_schedule[1] ≈ 1.0f0
    @test dense_solvent_schedule[9] ≈ 0.5f0
    @test dense_solvent_schedule[10] ≈ 0.405f0
    @test dense_solvent_schedule[end] ≈ 0.0f0
    @test length(dense_solvent_schedule_ele_scaled) == 21
    @test dense_solvent_schedule_ele_scaled[1] ≈ 1.0f0
    @test dense_solvent_schedule_ele_scaled[2] ≈ 0.86125f0
    @test dense_solvent_schedule_ele_scaled[9] ≈ 0.5f0
    @test dense_solvent_schedule_ele_scaled[10] ≈ 0.405f0
    @test dense_solvent_schedule_ele_scaled[end] ≈ 0.0f0
    default_dense_region = count(λ -> 0.55f0 <= λ <= 0.75f0, solvent_schedule)
    dense_dense_region = count(λ -> 0.55f0 <= λ <= 0.75f0, dense_solvent_schedule)
    @test dense_dense_region > 0
    default_hotspot_region = count(λ -> 0.575f0 <= λ <= 0.725f0, solvent_schedule)
    dense_hotspot_region = count(λ -> 0.575f0 <= λ <= 0.725f0, dense_solvent_schedule)
    @test dense_hotspot_region > 0

    default_diag = AWHGrads.solvent_lambda_schedule_diagnostics(solvent_schedule, :default, Float32)
    dense_diag = AWHGrads.solvent_lambda_schedule_diagnostics(dense_solvent_schedule, :default, Float32)
    @test length(dense_diag) == length(dense_solvent_schedule)
    @test count(entry -> entry.stage == :charge, dense_diag) == 9
    @test count(entry -> entry.stage == :lj, dense_diag) == 12
    @test dense_diag[1].idx == 1
    @test dense_diag[1].global_lambda ≈ 1.0f0
    @test dense_diag[1].elec_lambda ≈ 1.0f0
    @test dense_diag[1].lj_lambda ≈ 1.0f0
    @test dense_diag[1].stage == :charge
    @test dense_diag[9].idx == 9
    @test dense_diag[9].global_lambda ≈ 0.5f0
    @test dense_diag[9].elec_lambda ≈ 0.0f0
    @test dense_diag[9].lj_lambda ≈ 1.0f0
    @test dense_diag[9].stage == :charge
    endpoint_idxs = AWHGrads.solvent_stage_a_endpoint_state_indices(
        AWHGrads.ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            lambda_scheduler=:ele_scaled,
        ),
        AWHGrads.resolve_leg_state_schedule(
            AWHGrads.ThermodynamicLegConfig(
                name=:solvent,
                pdb="ethanol_solv.pdb",
                lambda_schedule=dense_solvent_schedule_ele_scaled,
                lambda_scheduler=:ele_scaled,
            ),
            fallback_schedule,
            Float32,
        );
        lj_lambda_max=0.3025f0,
    )
    @test length(endpoint_idxs) > 0
    @test dense_diag[10].idx == 10
    @test dense_diag[10].global_lambda ≈ 0.405f0
    @test dense_diag[10].elec_lambda ≈ 0.0f0
    @test dense_diag[10].lj_lambda ≈ 0.81f0
    @test dense_diag[10].stage == :lj

    dense_diag_ele_scaled = AWHGrads.solvent_lambda_schedule_diagnostics(
        dense_solvent_schedule_ele_scaled,
        :ele_scaled,
        Float32,
    )
    @test length(dense_diag_ele_scaled) == length(dense_solvent_schedule_ele_scaled)
    @test count(entry -> entry.stage == :charge, dense_diag_ele_scaled) == 9
    @test count(entry -> entry.stage == :lj, dense_diag_ele_scaled) == 12
    @test dense_diag_ele_scaled[1].global_lambda ≈ 1.0f0
    @test dense_diag_ele_scaled[1].elec_lambda ≈ 1.0f0
    @test dense_diag_ele_scaled[2].global_lambda ≈ 0.86125f0
    @test dense_diag_ele_scaled[2].elec_lambda ≈ 0.85f0
    @test dense_diag_ele_scaled[9].global_lambda ≈ 0.5f0
    @test dense_diag_ele_scaled[9].elec_lambda ≈ 0.0f0
    @test dense_diag_ele_scaled[9].lj_lambda ≈ 1.0f0
    intended_elec_stage = Float32[
        1.0,
        0.85,
        0.7,
        0.55,
        0.4,
        0.3,
        0.2,
        0.1,
        0.0,
    ]
    @test [entry.elec_lambda for entry in dense_diag_ele_scaled if entry.stage == :charge] ≈ intended_elec_stage

    override_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        lambda_schedule=solvent_schedule,
    )
    override_resolved = AWHGrads.resolve_leg_state_schedule(override_leg, fallback_schedule, Float32)
    @test override_resolved.lambda == solvent_schedule
    @test override_resolved.coupled_state_idx == 1
    @test override_resolved.decoupled_state_idx == 31

    default_cycle = AWHGrads.default_cycle_config(FT=Float32)
    solvent_leg = only(filter(leg -> leg.name == :solvent, default_cycle.legs))
    vacuum_leg = only(filter(leg -> leg.name == :vacuum, default_cycle.legs))
    @test solvent_leg.lambda_schedule == dense_solvent_schedule_ele_scaled
    @test solvent_leg.electrostatics_method == :pme
    @test solvent_leg.lambda_scheduler == :ele_scaled
    @test solvent_leg.coulomb_softcore_model == :gapsys
    @test solvent_leg.lj_softcore_model == :gapsys
    @test vacuum_leg.electrostatics_method == :none
    @test vacuum_leg.coulomb_softcore_model == :gapsys
    @test vacuum_leg.lj_softcore_model == :gapsys

    invalid_short_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        lambda_schedule=[1.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_short_leg, fallback_schedule, Float32)

    invalid_out_of_range_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        lambda_schedule=[1.0, -0.1, 0.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_out_of_range_leg, fallback_schedule, Float32)

    invalid_nonmonotonic_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        lambda_schedule=[1.0, 0.5, 0.75, 0.0],
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_nonmonotonic_leg, fallback_schedule, Float32)

    cycle_cfg = AWHGrads.default_cycle_config()
    solvent_leg = only(filter(leg -> leg.name == :solvent, cycle_cfg.legs))
    vacuum_leg = only(filter(leg -> leg.name == :vacuum, cycle_cfg.legs))

    @test solvent_leg.lambda_schedule == dense_solvent_schedule_ele_scaled
    @test isnothing(vacuum_leg.lambda_schedule)
    @test solvent_leg.ensemble == :npt
    @test solvent_leg.include_pv
    @test vacuum_leg.is_vacuum
    @test !vacuum_leg.include_pv

    staged_resolved = AWHGrads.resolve_leg_state_schedule(solvent_leg, fallback_schedule, Float32)
    @test staged_resolved.lambda == dense_solvent_schedule_ele_scaled
    @test staged_resolved.coupled_state_idx == 1
    @test staged_resolved.decoupled_state_idx == 21

    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    resolved_cycle = AWHGrads.resolved_cycle_config(sim_cfg)
    resolved_solvent_leg = only(filter(leg -> leg.name == :solvent, resolved_cycle.legs))
    resolved_vacuum_leg = only(filter(leg -> leg.name == :vacuum, resolved_cycle.legs))
    @test resolved_solvent_leg.lambda_schedule == dense_solvent_schedule_ele_scaled
    @test isnothing(resolved_vacuum_leg.lambda_schedule)
    @test resolved_solvent_leg.electrostatics_method == :pme
    @test resolved_solvent_leg.lambda_scheduler == :ele_scaled
    @test resolved_solvent_leg.coulomb_softcore_model == :gapsys
    @test resolved_solvent_leg.lj_softcore_model == :gapsys
    @test resolved_vacuum_leg.electrostatics_method == :none
    @test resolved_vacuum_leg.coulomb_softcore_model == :gapsys
    @test resolved_vacuum_leg.lj_softcore_model == :gapsys

    invalid_electrostatics_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        electrostatics_method=:mystery,
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_electrostatics_leg, fallback_schedule, Float32)

    invalid_scheduler_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        lambda_scheduler=:unknown_mode,
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_scheduler_leg, fallback_schedule, Float32)

    invalid_coulomb_model_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        coulomb_softcore_model=:mystery,
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_coulomb_model_leg, fallback_schedule, Float32)

    invalid_lj_model_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        lj_softcore_model=:mystery,
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_lj_model_leg, fallback_schedule, Float32)

    invalid_pme_rf_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        electrostatics_method=:pme,
        coulomb_softcore_model=:gapsys_rf,
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_pme_rf_leg, fallback_schedule, Float32)

    invalid_vacuum_pme_leg = AWHGrads.ThermodynamicLegConfig(
        name=:vacuum,
        pdb="ethanol_vac.pdb",
        is_vacuum=true,
        electrostatics_method=:pme,
        coulomb_softcore_model=:gapsys,
    )
    @test_throws ArgumentError AWHGrads.resolve_leg_state_schedule(invalid_vacuum_pme_leg, fallback_schedule, Float32)
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
    baseline_vacuum_leg = only(filter(leg -> leg.name == :vacuum, baseline_cfg.sim_cfg.cycle.legs))
    @test isnothing(baseline_solvent_leg.lambda_schedule)
    @test isnothing(baseline_vacuum_leg.lambda_schedule)
    @test length(baseline_cfg.sim_cfg.lambda_schedule) == 21
    @test baseline_solvent_leg.include_pv

    staged_npt_cfg = include(joinpath(@__DIR__, "..", "scripts", "benchmark_config_staged_npt.jl"))
    staged_npt_solvent_leg = only(filter(leg -> leg.name == :solvent, staged_npt_cfg.sim_cfg.cycle.legs))
    staged_npt_vacuum_leg = only(filter(leg -> leg.name == :vacuum, staged_npt_cfg.sim_cfg.cycle.legs))
    @test length(staged_npt_solvent_leg.lambda_schedule) == 31
    @test isnothing(staged_npt_vacuum_leg.lambda_schedule)

    nvt_cfg = include(joinpath(@__DIR__, "..", "scripts", "benchmark_config_staged_nvt.jl"))
    nvt_solvent_leg = only(filter(leg -> leg.name == :solvent, nvt_cfg.sim_cfg.cycle.legs))
    nvt_vacuum_leg = only(filter(leg -> leg.name == :vacuum, nvt_cfg.sim_cfg.cycle.legs))
    @test length(nvt_solvent_leg.lambda_schedule) == 31
    @test isnothing(nvt_vacuum_leg.lambda_schedule)
    @test nvt_solvent_leg.ensemble == :nvt
    @test !nvt_solvent_leg.include_pv
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
    @test sim_cfg.awh_control.bias_update_interval_md_steps == 1000
    @test sim_cfg.awh_control.stats_log_every_updates == 1
    @test isnothing(sim_cfg.awh_control.update_freq)
    @test sim_cfg.awh_control.coverage_threshold == 1.0
    @test sim_cfg.awh_control.significant_weight == 0.1
    @test sim_cfg.awh_control.initial_n_bias == 100

    opt_cfg = AWHGrads.default_optimization_config()
    @test opt_cfg.max_inner_epochs == 10
    @test opt_cfg.kl_target == Float32(0.25)
    @test opt_cfg.eigenvalue_tol_scale == Float32(1e-3)
    @test opt_cfg.max_phi_step_solute == Float32(0.6)
    @test opt_cfg.line_search_noise_tolerance_fraction == Float32(0.1)
    @test opt_cfg.awh_min_lambda_ess == 300
    @test opt_cfg.awh_parity_tol_kT == Float32(0.25)
    @test opt_cfg.awh_parity_gate_mode == :support_aware_max
    @test opt_cfg.awh_parity_support_threshold == Float32(300)
    @test opt_cfg.awh_parity_near_pass_factor == Float32(2.0)
    @test opt_cfg.awh_endpoint_target_ratio == Float32(0.3)
    @test opt_cfg.awh_solvent_tail_lj_max == Float32(0.3025)
    @test opt_cfg.awh_solvent_tail_min_state_occupancy == Float32(0.0125)
    @test opt_cfg.awh_solvent_endpoint_min_fraction == Float32(0.0)
    @test opt_cfg.awh_stageA_history_blocks == 8
    @test opt_cfg.awh_stageB_cooldown_blocks == 2
    @test opt_cfg.awh_stageB_near_pass_cooldown_blocks == 0
    @test opt_cfg.awh_stageB_probe_growth_factor == Float32(1.5)
    @test opt_cfg.awh_stageB_probe_near_pass_scale == Float32(0.5)
    @test opt_cfg.awh_stageB_probe_growth_ns == Float32(2.0)
    @test opt_cfg.awh_stageB_support_allow_missing == 3
end

@testset "full physical coverage gates initial-stage doubling" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    awh_control = AWHGrads.AWHControlConfig(
        seed_num_md_steps=5,
        bias_update_interval_md_steps=5,
        stats_log_every_updates=1,
        update_freq=1,
        coverage_threshold=1.0,
        significant_weight=0.1,
        initial_n_bias=50,
        well_tempered_factor=Inf,
        coverage_type=:physical,
    )
    sim_cfg = AWHGrads.simulation_config_with(sim_cfg; awh_control=awh_control)
    AWHGrads.apply_simulation_config!(sim_cfg)

    function build_awh_sim()
        awh_sim, _ = AWHGrads.setup_alchemical_awh(
            "ethanol_vac.pdb",
            sim_cfg.solute_idx;
            lambda_values=Float32[1.0, 0.5, 0.0],
            awh_control=awh_control,
            is_vacuum=true,
        )
        awh_sim.state.w_seg .= awh_sim.state.rho
        awh_sim.state.w2_seg .= awh_sim.state.rho .^ 2
        awh_sim.state.n_accum = awh_sim.update_freq
        awh_sim.state.inefficiency = 1.0f0
        awh_sim.state.N_eff = 0.0f0
        awh_sim.state.in_initial_stage = true
        return awh_sim
    end

    partial_cov = build_awh_sim()
    partial_cov.state.visited_windows = Set([1, 2])
    Molly.update_awh_bias!(partial_cov, 1)
    @test partial_cov.state.N_bias == 50.0f0
    @test partial_cov.state.in_initial_stage

    full_cov = build_awh_sim()
    full_cov.state.visited_windows = Set([1, 2, 3])
    Molly.update_awh_bias!(full_cov, 1)
    @test full_cov.state.N_bias == 100.0f0
end

@testset "smaller initial_n_bias produces larger initial bias corrections" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    AWHGrads.apply_simulation_config!(sim_cfg)

    function build_awh_sim(initial_n_bias::Int)
        awh_control = AWHGrads.AWHControlConfig(
            seed_num_md_steps=5,
            bias_update_interval_md_steps=5,
            stats_log_every_updates=1,
            update_freq=1,
            coverage_threshold=1.0,
            significant_weight=0.1,
            initial_n_bias=initial_n_bias,
            well_tempered_factor=Inf,
            coverage_type=:physical,
        )
        awh_sim, _ = AWHGrads.setup_alchemical_awh(
            "ethanol_vac.pdb",
            sim_cfg.solute_idx;
            lambda_values=Float32[1.0, 0.5, 0.0],
            awh_control=awh_control,
            is_vacuum=true,
        )
        awh_sim.state.w_seg .= Float32[0.90, 0.09, 0.01]
        awh_sim.state.w2_seg .= awh_sim.state.w_seg .^ 2
        awh_sim.state.n_accum = awh_sim.update_freq
        awh_sim.state.inefficiency = 1.0f0
        awh_sim.state.N_eff = 0.0f0
        awh_sim.state.in_initial_stage = true
        return awh_sim
    end

    aggressive = build_awh_sim(100)
    conservative = build_awh_sim(500)

    delta_aggressive = Molly.update_awh_bias!(aggressive, 1)
    delta_conservative = Molly.update_awh_bias!(conservative, 1)

    @test maximum(abs.(delta_aggressive)) > maximum(abs.(delta_conservative))
end

@testset "AWH cadence resolution" begin
    default_cadence = AWHGrads.resolve_awh_iteration_cadence(AWHGrads.AWHControlConfig())
    @test default_cadence.n_md_steps == 10
    @test default_cadence.update_freq == 100
    @test default_cadence.log_freq == 100
    @test default_cadence.effective_bias_update_md_steps == 1000
    @test default_cadence.auto_update_freq

    legacy_example_cadence = AWHGrads.resolve_awh_iteration_cadence(
        AWHGrads.AWHControlConfig(seed_num_md_steps=50, bias_update_interval_md_steps=25_000)
    )
    @test legacy_example_cadence.update_freq == 500
    @test legacy_example_cadence.effective_bias_update_md_steps == 25_000

    long_relax_cadence = AWHGrads.resolve_awh_iteration_cadence(
        AWHGrads.AWHControlConfig(seed_num_md_steps=250, bias_update_interval_md_steps=25_000)
    )
    @test long_relax_cadence.update_freq == 100
    @test long_relax_cadence.log_freq == 100
    @test long_relax_cadence.effective_bias_update_md_steps == 25_000

    nondivisible_cadence = AWHGrads.resolve_awh_iteration_cadence(
        AWHGrads.AWHControlConfig(seed_num_md_steps=300, bias_update_interval_md_steps=25_000)
    )
    @test nondivisible_cadence.update_freq == 83
    @test nondivisible_cadence.effective_bias_update_md_steps == 24_900

    override_cadence = AWHGrads.resolve_awh_iteration_cadence(
        AWHGrads.AWHControlConfig(seed_num_md_steps=10, update_freq=123, stats_log_every_updates=2)
    )
    @test override_cadence.update_freq == 123
    @test override_cadence.log_freq == 246
    @test !override_cadence.auto_update_freq
end

@testset "per-leg AWH cadence overrides" begin
    global_awh = AWHGrads.AWHControlConfig(
        seed_num_md_steps=250,
        bias_update_interval_md_steps=25_000,
    )
    solvent_leg = AWHGrads.ThermodynamicLegConfig(
        name=:solvent,
        pdb="ethanol_solv.pdb",
        awh_seed_num_md_steps=1000,
        awh_bias_update_interval_md_steps=100_000,
        probe_awh_seed_num_md_steps=2000,
    )
    vacuum_leg = AWHGrads.ThermodynamicLegConfig(
        name=:vacuum,
        pdb="ethanol_vac.pdb",
        is_vacuum=true,
    )

    solvent_awh = AWHGrads.resolve_leg_awh_control(global_awh, solvent_leg)
    vacuum_awh = AWHGrads.resolve_leg_awh_control(global_awh, vacuum_leg)

    @test solvent_awh.seed_num_md_steps == 1000
    @test solvent_awh.bias_update_interval_md_steps == 100_000
    @test solvent_leg.probe_awh_seed_num_md_steps == 2000
    @test vacuum_awh.seed_num_md_steps == 250
    @test vacuum_awh.bias_update_interval_md_steps == 25_000

    solvent_cadence = AWHGrads.resolve_awh_iteration_cadence(solvent_awh)
    vacuum_cadence = AWHGrads.resolve_awh_iteration_cadence(vacuum_awh)
    @test solvent_cadence.n_md_steps == 1000
    @test solvent_cadence.update_freq == 100
    @test vacuum_cadence.n_md_steps == 250
    @test vacuum_cadence.update_freq == 100
end

@testset "per-leg alchemical path selection" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    AWHGrads.apply_simulation_config!(sim_cfg)

    default_diag = AWHGrads.solvent_lambda_schedule_diagnostics(Float32[1.0, 0.75, 0.5], :default, Float32)
    ele_scaled_diag = AWHGrads.solvent_lambda_schedule_diagnostics(Float32[1.0, 0.75, 0.5], :ele_scaled, Float32)
    @test default_diag[2].elec_lambda ≈ 0.5f0
    @test ele_scaled_diag[2].elec_lambda ≈ sqrt(0.5f0)

    awh_sim, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_solv.pdb",
        sim_cfg.solute_idx;
        lambda_values=Float32[1.0, 0.75, 0.5, 0.0],
        is_vacuum=false,
        ensemble=:npt,
        lambda_scheduler=:ele_scaled,
        coulomb_softcore_model=:gapsys_rf,
        lj_softcore_model=:gapsys,
    )

    state_inters = awh_sim.state.state_pairwise_inters[1]
    lj_idx = findfirst(x -> x isa AWHGrads.Molly.LennardJonesSoftCoreGapsys, state_inters)
    coul_idx = findfirst(x -> x isa AWHGrads.Molly.CoulombSoftCoreGapsysReactionField, state_inters)
    @test lj_idx !== nothing
    @test coul_idx !== nothing
    @test state_inters[lj_idx].scheduler isa AWHGrads.Molly.EleScaledLambdaScheduler
    @test state_inters[coul_idx].scheduler isa AWHGrads.Molly.EleScaledLambdaScheduler

    awh_sim_pme, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_solv.pdb",
        sim_cfg.solute_idx;
        lambda_values=Float32[1.0, 0.75, 0.5, 0.0],
        is_vacuum=false,
        ensemble=:npt,
        electrostatics_method=:pme,
        lambda_scheduler=:ele_scaled,
        coulomb_softcore_model=:gapsys,
        lj_softcore_model=:gapsys,
    )

    pme_pairwise = awh_sim_pme.state.state_pairwise_inters[1]
    pme_general = awh_sim_pme.state.state_general_inters[1]
    lj_pme_idx = findfirst(x -> x isa AWHGrads.Molly.LennardJonesSoftCoreGapsys, pme_pairwise)
    coul_pme_idx = findfirst(x -> x isa AWHGrads.Molly.CoulombSoftCoreGapsysEwald, pme_pairwise)
    pme_idx = findfirst(x -> x isa AWHGrads.Molly.PME, pme_general)
    @test lj_pme_idx !== nothing
    @test coul_pme_idx !== nothing
    @test pme_idx !== nothing
    @test pme_pairwise[coul_pme_idx].scheduler isa AWHGrads.Molly.EleScaledLambdaScheduler
    @test pme_general[pme_idx].scheduler isa AWHGrads.Molly.EleScaledLambdaScheduler

    awh_sim_ewald, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_solv.pdb",
        sim_cfg.solute_idx;
        lambda_values=Float32[1.0, 0.5, 0.0],
        is_vacuum=false,
        ensemble=:npt,
        electrostatics_method=:ewald,
        coulomb_softcore_model=:beutler,
        lj_softcore_model=:gapsys,
    )

    ewald_pairwise = awh_sim_ewald.state.state_pairwise_inters[1]
    ewald_general = awh_sim_ewald.state.state_general_inters[1]
    coul_ewald_idx = findfirst(x -> x isa AWHGrads.Molly.CoulombSoftCoreBeutlerEwald, ewald_pairwise)
    ewald_idx = findfirst(x -> x isa AWHGrads.Molly.Ewald, ewald_general)
    @test coul_ewald_idx !== nothing
    @test ewald_idx !== nothing
    @test ewald_pairwise[coul_ewald_idx].scheduler isa AWHGrads.Molly.DefaultLambdaScheduler
    @test ewald_general[ewald_idx].scheduler isa AWHGrads.Molly.DefaultLambdaScheduler
end

@testset "AWH control plumbing" begin
    awh_control = AWHGrads.AWHControlConfig(
        seed_num_md_steps=10,
        bias_update_interval_md_steps=1230,
        stats_log_every_updates=2,
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
    @test awh_sim.log_freq == awh_control.update_freq * awh_control.stats_log_every_updates
    @test awh_sim.coverage_threshold == typeof(awh_sim.coverage_threshold)(awh_control.coverage_threshold)
    @test awh_sim.significant_weight == typeof(awh_sim.significant_weight)(awh_control.significant_weight)
end

@testset "macro epoch startup only reuses AWH state on warm starts" begin
    FT = Float32
    sim_cfg = AWHGrads.default_simulation_config(FT=FT, AT=Array)
    AWHGrads.apply_simulation_config!(sim_cfg)

    cycle_cfg = AWHGrads.ThermodynamicCycleConfig(
        legs=[
            AWHGrads.ThermodynamicLegConfig(
                name=:vacuum,
                pdb="ethanol_vac.pdb",
                is_vacuum=true,
                include_pv=false,
                probe_time=FT(0.25)u"ns",
            ),
        ],
        include_standard_state_correction=false,
        target_dG_kcal_mol=0.0,
    )
    state_schedules_by_leg = Dict(
        :vacuum => AWHGrads.resolve_leg_state_schedule(only(cycle_cfg.legs), sim_cfg.lambda_schedule, FT),
    )
    opt_cfg = AWHGrads.default_optimization_config(FT=FT)
    pstate = AWHGrads.initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)

    T_coord = typeof(FT(1.0)u"nm")
    T_vol = typeof(FT(1.0)u"nm^3")
    T_en = typeof(FT(1.0)u"kJ * mol^-1")
    n_states = length(state_schedules_by_leg[:vacuum].lambda)
    sentinel_rho = fill(FT(0.25), n_states)
    sentinel_bias = (
        f=FT.(collect(11:(10 + n_states))),
        rho=sentinel_rho,
        log_rho=log.(sentinel_rho),
    )

    cold_runtime = AWHGrads.RuntimeState(active_bias=Dict{Symbol, Any}(:vacuum => sentinel_bias))
    awh_cold_by_leg, _ = AWHGrads.setup_macro_legs(
        cycle_cfg,
        sim_cfg,
        state_schedules_by_leg,
        cold_runtime,
        pstate.theta_active,
        pstate.idxs_by_leg,
        T_coord,
        T_vol,
        T_en,
        2,
        FT(1e-5),
    )

    @test awh_cold_by_leg[:vacuum].state.f != sentinel_bias.f
    @test awh_cold_by_leg[:vacuum].state.rho != sentinel_bias.rho
    @test awh_cold_by_leg[:vacuum].state.log_rho != sentinel_bias.log_rho

    seed_runtime = AWHGrads.RuntimeState()
    awh_seed_by_leg, _ = AWHGrads.setup_macro_legs(
        cycle_cfg,
        sim_cfg,
        state_schedules_by_leg,
        seed_runtime,
        pstate.theta_active,
        pstate.idxs_by_leg,
        T_coord,
        T_vol,
        T_en,
        1,
        FT(1e-5),
    )
    restart_state = AWHGrads.capture_restart_state(awh_seed_by_leg[:vacuum])

    warm_runtime = AWHGrads.RuntimeState(
        active_bias=Dict{Symbol, Any}(:vacuum => sentinel_bias),
        restart_cache=Dict{Symbol, Any}(:vacuum => restart_state),
    )
    awh_warm_by_leg, _ = AWHGrads.setup_macro_legs(
        cycle_cfg,
        sim_cfg,
        state_schedules_by_leg,
        warm_runtime,
        pstate.theta_active,
        pstate.idxs_by_leg,
        T_coord,
        T_vol,
        T_en,
        2,
        FT(1e-5),
    )

    @test awh_warm_by_leg[:vacuum].state.f == sentinel_bias.f
    @test awh_warm_by_leg[:vacuum].state.rho == sentinel_bias.rho
    @test awh_warm_by_leg[:vacuum].state.log_rho == sentinel_bias.log_rho
end

@testset "Stage B probe freezes bias updates" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    awh_control = AWHGrads.AWHControlConfig(
        seed_num_md_steps=5,
        bias_update_interval_md_steps=10,
        stats_log_every_updates=1,
        update_freq=2,
        coverage_threshold=0.8,
        significant_weight=0.1,
        initial_n_bias=50,
        well_tempered_factor=Inf,
        coverage_type=:physical,
    )
    sim_cfg = AWHGrads.simulation_config_with(sim_cfg; awh_control=awh_control)
    AWHGrads.apply_simulation_config!(sim_cfg)

    T_coord = typeof(sim_cfg.FT(1.0)u"nm")
    T_vol = typeof(sim_cfg.FT(1.0)u"nm^3")
    T_en = typeof(sim_cfg.FT(1.0)u"kJ * mol^-1")
    logger = Molly.AWHEnsembleLogger(T_coord, T_vol, T_en, 1)

    awh_sim, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_vac.pdb",
        sim_cfg.solute_idx;
        lambda_values=Float32[1.0, 0.5, 0.0],
        awh_control=awh_control,
        is_vacuum=true,
        logger=logger,
    )

    awh_sim.state.f .= Float32[0.0, 0.25, -0.15]
    awh_sim.state.n_accum = 1
    awh_sim.state.w_seg .= one.(awh_sim.state.w_seg)
    awh_sim.state.w2_seg .= fill(2.0f0, length(awh_sim.state.w2_seg))
    awh_sim.state.w_last .= fill(3.0f0, length(awh_sim.state.w_last))
    push!(awh_sim.state.visited_windows, 1)
    push!(awh_sim.state.cv_buffer, Float32[0.25])

    frozen_bias = AWHGrads.extract_awh_data(awh_sim)
    probe_sim, probe_bias = AWHGrads.build_stage_b_probe_sim(awh_sim, 20)

    @test probe_bias.f == frozen_bias.f
    @test probe_bias.log_rho == frozen_bias.log_rho
    @test probe_sim.update_freq == max(awh_sim.update_freq, fld(20, awh_sim.n_md_steps) + 1)
    @test probe_sim.n_md_steps == awh_sim.n_md_steps
    @test probe_sim.state.n_accum == 0
    @test all(iszero, probe_sim.state.w_seg)
    @test all(iszero, probe_sim.state.w2_seg)
    @test all(iszero, probe_sim.state.w_last)
    @test isempty(probe_sim.state.visited_windows)
    @test isempty(probe_sim.state.cv_buffer)
    @test probe_sim.state.active_sys.loggers.awh_logger.should_log
    @test all(
        state_loggers -> !hasproperty(state_loggers, :awh_logger) || state_loggers.awh_logger.should_log,
        probe_sim.state.state_loggers,
    )

    simulate!(probe_sim, 20)

    @test probe_sim.state.f == frozen_bias.f
    @test probe_sim.state.log_rho == frozen_bias.log_rho
end

@testset "Stage B probe can override lambda cadence without updating bias" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    awh_control = AWHGrads.AWHControlConfig(
        seed_num_md_steps=5,
        bias_update_interval_md_steps=10,
        stats_log_every_updates=1,
        update_freq=2,
        coverage_threshold=0.8,
        significant_weight=0.1,
        initial_n_bias=50,
        well_tempered_factor=Inf,
        coverage_type=:physical,
    )
    sim_cfg = AWHGrads.simulation_config_with(sim_cfg; awh_control=awh_control)
    AWHGrads.apply_simulation_config!(sim_cfg)

    T_coord = typeof(sim_cfg.FT(1.0)u"nm")
    T_vol = typeof(sim_cfg.FT(1.0)u"nm^3")
    T_en = typeof(sim_cfg.FT(1.0)u"kJ * mol^-1")
    logger = Molly.AWHEnsembleLogger(T_coord, T_vol, T_en, 1)

    awh_sim, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_vac.pdb",
        sim_cfg.solute_idx;
        lambda_values=Float32[1.0, 0.5, 0.0],
        awh_control=awh_control,
        is_vacuum=true,
        logger=logger,
    )

    awh_sim.state.f .= Float32[0.0, 0.25, -0.15]
    frozen_bias = AWHGrads.extract_awh_data(awh_sim)
    probe_sim, probe_bias = AWHGrads.build_stage_b_probe_sim(
        awh_sim,
        20;
        probe_num_md_steps=10,
    )

    @test probe_bias.f == frozen_bias.f
    @test probe_bias.log_rho == frozen_bias.log_rho
    @test probe_sim.n_md_steps == 10
    @test probe_sim.update_freq == max(awh_sim.update_freq, fld(20, 10) + 1)

    simulate!(probe_sim, 20)

    @test probe_sim.state.f == frozen_bias.f
    @test probe_sim.state.log_rho == frozen_bias.log_rho
end

@testset "Stage B split-parity diagnostics helpers" begin
    FT = Float32
    beta = FT(1.0)
    log_gibbs = zeros(FT, 3)
    energies = FT[
        0.0  1.0  2.0;
        0.2  1.1  1.9;
        0.1  0.9  2.2;
        0.0  1.0  2.0;
        0.2  1.1  1.9;
        0.1  0.9  2.2;
    ]

    log_denom = AWHGrads.reference_log_mixture_denominator(energies, log_gibbs, beta)
    F_mbar = AWHGrads.compute_full_mbar_profile_from_log_mixture_denom(energies, log_denom, beta)

    good = AWHGrads.compute_stage_b_split_parity(
        energies,
        log_denom,
        copy(F_mbar),
        1,
        3,
        beta,
        FT(1.0),
        FT(0.1);
        parity_support_threshold=FT(0.0),
    )
    @test good.parity_ready
    @test good.parity_gap ≈ 0.0f0 atol=1f-6
    @test good.raw_parity_gap ≈ 0.0f0 atol=1f-6
    @test good.supported_parity_gap ≈ 0.0f0 atol=1f-6
    @test good.endpoint_parity_ready
    @test good.n_supported_states == 3
    @test good.failure_mode == :passed
    @test good.parity_worst_state_idx in 1:3
    @test occursin("endpoint_worst=", good.diagnostics)

    bad_awh = copy(F_mbar)
    bad_awh[2] += FT(2.0)
    bad = AWHGrads.compute_stage_b_split_parity(
        energies,
        log_denom,
        bad_awh,
        1,
        3,
        beta,
        FT(1.0),
        FT(0.1);
        parity_support_threshold=FT(0.0),
    )
    @test !bad.parity_ready
    @test bad.parity_gap > FT(1.5)
    @test bad.failure_mode == :supported_parity
    @test bad.parity_worst_state_idx == 2
    @test occursin("worst=λ2", bad.diagnostics)
    @test occursin("effective_source=internal", bad.diagnostics)
end

@testset "Stage B split-parity promotes mixed precision inputs" begin
    beta = Float32(1.0)
    log_gibbs = Float64[0.0, 0.1, -0.1]
    energies = Float32[
        0.0  1.0  2.0;
        0.2  1.1  1.9;
        0.1  0.9  2.2;
        0.0  1.0  2.0;
        0.2  1.1  1.9;
        0.1  0.9  2.2;
    ]
    volumes = Float32[1.0, 1.1, 0.9, 1.05, 1.0, 0.95]
    p0 = Float32(0.05)

    log_denom = AWHGrads.reference_log_mixture_denominator(
        energies,
        log_gibbs,
        beta;
        volumes=volumes,
        P0_energy_per_vol=p0,
    )
    F_mbar = AWHGrads.compute_full_mbar_profile_from_log_mixture_denom(
        energies,
        log_denom,
        beta;
        volumes=volumes,
        P0_energy_per_vol=p0,
    )
    result = AWHGrads.compute_stage_b_split_parity(
        energies,
        log_denom,
        F_mbar,
        1,
        3,
        beta,
        Float32(1.0),
        Float32(0.1);
        volumes=volumes,
        P0_energy_per_vol=p0,
        parity_support_threshold=Float32(0.0),
    )
    @test eltype(log_denom) == Float64
    @test eltype(F_mbar) == Float64
    @test result.parity_gap isa Float64
    @test result.split_gap isa Float64
    @test result.parity_ready
end

@testset "Stage B support-aware gate and probe policy" begin
    FT = Float32
    beta = FT(1.0)
    log_gibbs = zeros(FT, 3)
    energies = FT[
        0.0  12.0  2.0;
        0.1  11.5  2.1;
        0.2  12.5  1.9;
        0.0  12.0  2.0;
        0.1  11.7  2.2;
        0.2  12.3  1.8;
    ]

    log_denom = AWHGrads.reference_log_mixture_denominator(energies, log_gibbs, beta)
    F_mbar = AWHGrads.compute_full_mbar_profile_from_log_mixture_denom(energies, log_denom, beta)
    ess = AWHGrads.compute_state_reweighting_ess_from_log_mixture_denom(energies, log_denom, beta)
    @test ess[2] < min(ess[1], ess[3])

    threshold = FT((ess[2] + min(ess[1], ess[3])) / 2)
    bad_awh = copy(F_mbar)
    bad_awh[2] += FT(2.0)
    support_aware = AWHGrads.compute_stage_b_split_parity(
        energies,
        log_denom,
        bad_awh,
        1,
        3,
        beta,
        FT(1.0),
        FT(0.1);
        parity_support_threshold=threshold,
        support_allow_missing=0,
    )
    @test !support_aware.ready
    @test !support_aware.parity_ready
    @test support_aware.raw_parity_gap > FT(1.5)
    @test support_aware.supported_parity_gap < FT(1e-4)
    @test support_aware.n_supported_states == 2
    @test !support_aware.support_coverage_ready
    @test support_aware.required_supported_states == 3
    @test support_aware.failure_mode == :low_support

    raw_only = AWHGrads.compute_stage_b_split_parity(
        energies,
        log_denom,
        bad_awh,
        1,
        3,
        beta,
        FT(1.0),
        FT(0.1);
        parity_support_threshold=threshold,
        support_allow_missing=1,
    )
    @test !raw_only.ready
    @test raw_only.parity_ready
    @test raw_only.support_coverage_ready
    @test raw_only.required_supported_states == 2
    @test raw_only.failure_mode == :passed_raw_only

    grow_policy = AWHGrads.stage_b_next_probe_policy(
        100,
        100,
        AWHGrads.StageBStats(failure_mode=:low_support),
        2,
        0,
        50,
        FT(0.5),
        FT(4.0),
    )
    @test grow_policy.policy == :grow_sampling
    @test grow_policy.next_probe_steps == 150
    @test grow_policy.cooldown_blocks == 2

    endpoint_policy = AWHGrads.stage_b_next_probe_policy(
        100,
        100,
        AWHGrads.StageBStats(failure_mode=:endpoint_parity),
        2,
        0,
        50,
        FT(0.5),
        FT(4.0),
    )
    @test endpoint_policy.policy == :stay_bias_error
    @test endpoint_policy.next_probe_steps == 100
    @test endpoint_policy.cooldown_blocks == 2

    repeated_endpoint_policy = AWHGrads.stage_b_next_probe_policy(
        endpoint_policy.next_probe_steps,
        100,
        AWHGrads.StageBStats(failure_mode=:endpoint_parity),
        2,
        0,
        50,
        FT(0.5),
        FT(4.0),
    )
    @test repeated_endpoint_policy.policy == :stay_bias_error
    @test repeated_endpoint_policy.next_probe_steps == 100
    @test repeated_endpoint_policy.cooldown_blocks == 2

    supported_policy = AWHGrads.stage_b_next_probe_policy(
        100,
        100,
        AWHGrads.StageBStats(failure_mode=:supported_parity),
        2,
        0,
        50,
        FT(0.5),
        FT(4.0),
    )
    @test supported_policy.policy == :stay_bias_error
    @test supported_policy.next_probe_steps == 100
    @test supported_policy.cooldown_blocks == 2

    near_pass_policy = AWHGrads.stage_b_next_probe_policy(
        100,
        100,
        AWHGrads.StageBStats(failure_mode=:supported_parity, near_pass=true),
        2,
        0,
        50,
        FT(0.5),
        FT(4.0),
    )
    @test near_pass_policy.policy == :near_pass
    @test near_pass_policy.next_probe_steps == 50
    @test near_pass_policy.cooldown_blocks == 0

    passed_policy = AWHGrads.stage_b_next_probe_policy(
        250,
        100,
        AWHGrads.StageBStats(ready=true, failure_mode=:passed),
        2,
        0,
        50,
        FT(0.5),
        FT(4.0),
    )
    @test passed_policy.policy == :passed
    @test passed_policy.next_probe_steps == 250
    @test passed_policy.cooldown_blocks == 0
end

@testset "compiler-safe logger helper" begin
    tee = AWHGrads.TeeLogger([Logging.NullLogger(), Logging.NullLogger()])

    original_global_logger = Logging.global_logger()

    try
        Logging.global_logger(tee)

        hidden_loggers = Logging.with_logger(tee) do
            AWHGrads.with_compiler_safe_logger() do
                spawned_loggers = Threads.@spawn (Logging.current_logger(), Logging.global_logger())
                (
                    Logging.current_logger(),
                    Logging.global_logger(),
                    fetch(spawned_loggers)...,
                )
            end
        end

        hidden_current, hidden_global, spawned_current, spawned_global = hidden_loggers
        @test hidden_current isa Logging.NullLogger
        @test hidden_global isa Logging.NullLogger
        @test spawned_current isa Logging.NullLogger
        @test spawned_global isa Logging.NullLogger

        restored_loggers = Logging.with_logger(tee) do
            AWHGrads.with_compiler_safe_logger() do
                nothing
            end
            (Logging.current_logger(), Logging.global_logger())
        end

        restored_current, restored_global = restored_loggers
        @test restored_current === tee
        @test restored_global === tee
    finally
        Logging.global_logger(original_global_logger)
    end
end

@testset "setup_logging flushes and truncates file output" begin
    original_global_logger = Logging.global_logger()
    log_path, temp_io = mktemp()
    close(temp_io)
    open(log_path, "w") do io
        write(io, "stale log contents\n")
    end

    log_io = nothing
    try
        log_io = AWHGrads.setup_logging(log_path; append=false)
        @info "sync-check" marker=:immediate_flush

        contents = read(log_path, String)
        @test !occursin("stale log contents", contents)
        @test occursin("Logging initialized", contents)
        @test occursin("sync-check", contents)
        @test occursin("marker = immediate_flush", contents)
    finally
        Logging.global_logger(original_global_logger)
        if log_io !== nothing
            close(log_io)
        end
        rm(log_path; force=true)
    end
end

@testset "Stage B single-probe semantics" begin
    stats_from_probe = AWHGrads.StageBStats((
        ready=false,
        split_ready=true,
        split_gap=0.4f0,
        parity_ready=false,
        parity_gap=0.3f0,
        n_frames=54,
        dG_half_1=1.0f0,
        dG_half_2=1.4f0,
    ))
    @test stats_from_probe.n_accumulated_frames == 54
    @test stats_from_probe.n_probe_segments == 1
    @test stats_from_probe.accumulation_mode == :single_probe
    @test !stats_from_probe.support_switch_ready
    @test stats_from_probe.n_evicted_frames == 0

    probe_stats = (
        ready=false,
        split_ready=true,
        split_gap=0.4f0,
        parity_ready=false,
        parity_gap=0.3f0,
        raw_parity_gap=0.7f0,
        supported_parity_gap=0.3f0,
        endpoint_parity_gap=0.1f0,
        endpoint_parity_ready=true,
        n_frames=54,
        dG_half_1=1.0f0,
        dG_half_2=1.4f0,
        diagnostics="probe-only",
        parity_worst_state_idx=3,
        parity_worst_state_residual=0.3f0,
        n_supported_states=5,
        support_threshold=300.0f0,
        failure_mode=:supported_parity,
        near_pass=true,
    )
    staged = AWHGrads.StageBStats(merge(
        probe_stats,
        (
            n_accumulated_frames=probe_stats.n_frames,
            n_probe_segments=1,
            accumulation_mode=:single_probe,
            support_switch_ready=false,
            n_evicted_frames=0,
        ),
    ))
    @test staged.n_accumulated_frames == staged.n_frames == 54
    @test staged.n_probe_segments == 1
    @test staged.accumulation_mode == :single_probe
    @test !staged.support_switch_ready
    @test staged.n_evicted_frames == 0
    @test staged.failure_mode == :supported_parity
    @test staged.near_pass

    FT = Float32
    beta = FT(1.0)
    probe_a_energies = FT[
        0.0  1.0  2.0;
        0.1  1.1  1.9;
        0.2  1.2  1.8;
        0.0  1.0  2.1;
        0.1  1.1  1.95;
        0.2  1.2  1.85;
    ]
    probe_b_energies = FT[
        0.0  2.0  1.0;
        0.1  1.9  1.1;
        0.2  1.8  1.2;
        0.0  2.1  1.0;
        0.1  1.95 1.1;
        0.2  1.85 1.2;
    ]
    probe_a_log_gibbs = zeros(FT, 3)
    probe_b_log_gibbs = FT[0.0, 1.0, -1.0]

    probe_a_log_denom = AWHGrads.reference_log_mixture_denominator(
        probe_a_energies,
        probe_a_log_gibbs,
        beta,
    )
    probe_b_log_denom = AWHGrads.reference_log_mixture_denominator(
        probe_b_energies,
        probe_b_log_gibbs,
        beta,
    )
    frozen_bias_b = AWHGrads.compute_full_mbar_profile_from_log_mixture_denom(
        probe_b_energies,
        probe_b_log_denom,
        beta,
    )

    independent_probe = AWHGrads.compute_stage_b_split_parity(
        probe_b_energies,
        probe_b_log_denom,
        copy(frozen_bias_b),
        1,
        3,
        beta,
        FT(1.0),
        FT(0.25);
        parity_support_threshold=FT(0.0),
    )
    mixed_history = AWHGrads.compute_stage_b_split_parity(
        vcat(probe_a_energies, probe_b_energies),
        vcat(probe_a_log_denom, probe_b_log_denom),
        copy(frozen_bias_b),
        1,
        3,
        beta,
        FT(1.0),
        FT(0.25);
        parity_support_threshold=FT(0.0),
    )
    @test independent_probe.ready
    @test independent_probe.parity_gap ≈ 0.0f0 atol=1f-6
    @test !mixed_history.ready
    @test mixed_history.parity_gap > FT(0.5)
end

@testset "AWH default lambda scheduler plumbing" begin
    sim_cfg = AWHGrads.default_simulation_config(FT=Float32, AT=Array)
    AWHGrads.apply_simulation_config!(sim_cfg)

    custom_lambda = Float32[1.0, 0.75, 0.25, 0.0]
    awh_sim_custom, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_vac.pdb",
        sim_cfg.solute_idx;
        lambda_values=custom_lambda,
        is_vacuum=true,
    )

    solute_idx = first(sim_cfg.solute_idx)
    observed_lambda = [
        AWHGrads.Molly.from_device(atoms)[solute_idx].λ
        for atoms in awh_sim_custom.state.partition.λ_atoms
    ]
    @test observed_lambda == custom_lambda

    solvent_schedule = AWHGrads.default_solvent_leg_lambda_schedule(Float32)
    awh_sim_staged, _ = AWHGrads.setup_alchemical_awh(
        "ethanol_solv.pdb",
        sim_cfg.solute_idx;
        lambda_values=solvent_schedule,
        is_vacuum=false,
        ensemble=:npt,
        electrostatics_method=:pme,
        coulomb_softcore_model=:gapsys,
        lj_softcore_model=:gapsys,
    )

    @test length(awh_sim_staged.state.partition.λ_atoms) == length(solvent_schedule)

    first_atoms = AWHGrads.Molly.from_device(first(awh_sim_staged.state.partition.λ_atoms))
    mid_atoms = AWHGrads.Molly.from_device(awh_sim_staged.state.partition.λ_atoms[11])
    next_atoms = AWHGrads.Molly.from_device(awh_sim_staged.state.partition.λ_atoms[12])
    last_atoms = AWHGrads.Molly.from_device(last(awh_sim_staged.state.partition.λ_atoms))
    @test first_atoms[solute_idx].λ ≈ 1.0f0
    @test mid_atoms[solute_idx].λ ≈ 0.5f0
    @test next_atoms[solute_idx].λ ≈ 0.45125f0
    @test last_atoms[solute_idx].λ ≈ 0.0f0

    state_inters = awh_sim_staged.state.state_pairwise_inters[1]
    state_general = awh_sim_staged.state.state_general_inters[1]
    lj_idx = findfirst(x -> x isa AWHGrads.Molly.LennardJonesSoftCoreGapsys, state_inters)
    coul_idx = findfirst(x -> x isa AWHGrads.Molly.CoulombSoftCoreGapsysEwald, state_inters)
    pme_idx = findfirst(x -> x isa AWHGrads.Molly.PME, state_general)
    @test lj_idx !== nothing
    @test coul_idx !== nothing
    @test pme_idx !== nothing
    @test state_inters[lj_idx].scheduler isa AWHGrads.Molly.DefaultLambdaScheduler
    @test state_inters[coul_idx].scheduler isa AWHGrads.Molly.DefaultLambdaScheduler
    @test state_general[pme_idx].scheduler isa AWHGrads.Molly.DefaultLambdaScheduler
end
