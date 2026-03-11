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
