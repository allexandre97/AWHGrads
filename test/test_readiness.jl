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
