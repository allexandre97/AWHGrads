using Test
using Molly
using Unitful
# No need for using AWHGrads here as it is included in runtests.jl

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
