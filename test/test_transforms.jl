@testset "transforms" begin
    FT = Float32
    phi = FT.([-1, 0, 1])
    tmin = fill(FT(0.1), 3)
    tmax = fill(FT(0.5), 3)
    p0 = zeros(FT, 3)
    k = FT(1.0)

    theta = AWHGrads.map_phi_to_theta(phi, tmin, tmax, p0, k)
    @test length(theta) == 3
    @test all(theta .>= tmin)
    @test all(theta .<= tmax)

    cr = AWHGrads.get_chain_rule_multiplier(theta, tmin, tmax, k)
    @test length(cr) == 3
    @test all(cr .>= 0)
end

@testset "mixed family transforms" begin
    FT = Float32
    phi = zeros(FT, 4)
    tmin = FT[0.1, 0.0, -Inf, 0.05]
    tmax = FT[0.5, 1.5, Inf, Inf]
    p0 = FT[0.0, 0.0, -0.2, -log(exp(FT(1.0) - FT(0.05)) - 1)]
    families = [:sigma, :epsilon, :charge_chi, :charge_eta]
    k = FT(1.0)

    theta = AWHGrads.map_phi_to_theta(phi, tmin, tmax, p0, k, families)
    @test theta[1] >= tmin[1]
    @test theta[1] <= tmax[1]
    @test theta[2] >= tmin[2]
    @test theta[2] <= tmax[2]
    @test theta[3] ≈ p0[3]
    @test theta[4] ≈ FT(1.0)

    cr = AWHGrads.get_chain_rule_multiplier(phi, theta, tmin, tmax, p0, k, families)
    @test cr[1] >= 0
    @test cr[2] >= 0
    @test cr[3] == FT(1.0)
    @test FT(0.0) < cr[4] < FT(1.0)
end

@testset "constrained charge solve preserves molecular net charge" begin
    FT = Float64
    params = FT[0.1, 1.0, -0.3, 2.0]
    charge_chi = [1, 1, 3, 0]
    charge_eta = [2, 2, 4, 0]
    molecule_ids = [1, 1, 2, 2]
    molecule_targets = FT[0.0, 0.0]
    reference_charges = FT[0.0, 0.0, 0.0, 0.25]

    charges = AWHGrads.solve_constrained_charges(
        params,
        charge_chi,
        charge_eta,
        molecule_ids,
        molecule_targets,
        reference_charges,
    )

    @test sum(charges[1:2]) ≈ FT(0.0)
    @test sum(charges[3:4]) ≈ FT(0.0)
    @test charges[4] == FT(0.25)
end
