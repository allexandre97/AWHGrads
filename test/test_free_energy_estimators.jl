@testset "free energy estimators" begin
    FT = Float32
    energies = FT[
        1 2 3;
        2 3 4;
        3 4 5;
    ]
    bias = FT[0.0, 0.1, -0.1]

    profile = AWHGrads.compute_full_mbar_profile(energies, energies, bias, FT(1.0))
    @test length(profile) == 3
    @test isfinite(maximum(profile))

    gap = AWHGrads.compute_parity_gap(profile, profile; ref_idx=1)
    @test gap == 0
end
