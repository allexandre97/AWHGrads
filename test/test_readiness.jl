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
    opt_cfg = AWHGrads.default_optimization_config()
    @test opt_cfg.awh_parity_tol_kT == Float32(0.25)
    @test opt_cfg.awh_stageB_cooldown_blocks == 2
end
