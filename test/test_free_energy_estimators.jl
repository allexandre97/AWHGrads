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

@testset "AWH Gibbs log-weights drive MBAR denominators" begin
    FT = Float64
    bias_data = (
        f = FT[0.35, -0.10, 0.20],
        log_rho = log.(FT[0.70, 0.20, 0.10]),
    )
    log_gibbs_weights = AWHGrads.awh_log_gibbs_weights(bias_data)
    @test log_gibbs_weights ≈ bias_data.f .+ bias_data.log_rho

    energies_ref = FT[
        0.2 1.1 1.8;
        0.4 0.9 1.5;
        0.7 1.3 1.2;
    ]
    log_mixture = AWHGrads.reference_log_mixture_denominator(
        energies_ref,
        log_gibbs_weights,
        FT(1.0),
    )
    manual_log_mixture = [
        log(sum(exp.(log_gibbs_weights .- energies_ref[k, :])))
        for k in axes(energies_ref, 1)
    ]
    @test log_mixture ≈ manual_log_mixture

    profile = AWHGrads.compute_full_mbar_profile(
        energies_ref,
        energies_ref,
        log_gibbs_weights,
        FT(1.0),
    )
    profile_wrong = AWHGrads.compute_full_mbar_profile(
        energies_ref,
        energies_ref,
        bias_data.f,
        FT(1.0),
    )
    @test !isapprox(profile[2] - profile[1], profile_wrong[2] - profile_wrong[1]; atol=1e-8, rtol=1e-8)
end
