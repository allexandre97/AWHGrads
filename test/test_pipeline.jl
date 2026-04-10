using Test
using Molly
using Unitful
# No need for using AWHGrads here as it is included in runtests.jl

@testset "mixed precision defaults" begin
    @test AWHGrads.default_nonbonded_energy_type(Float32) == Float64
    @test AWHGrads.default_nonbonded_energy_type(Float64) == Float64

    sim_cfg = AWHGrads.default_simulation_config()
    @test sim_cfg.FT == Float32
    @test sim_cfg.nonbonded_energy_type == Float64
    @test AWHGrads.effective_nonbonded_energy_type(sim_cfg) == Float64
    @test AWHGrads.default_energy_analysis_type(sim_cfg) == Float64

    _, _, T_en = AWHGrads.awh_logger_value_types(sim_cfg)
    @test T_en == typeof(Float64(1.0)u"kJ * mol^-1")
end

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

@testset "optimization helper scheduling" begin
    param_names = [
        "atom_c3_σ",
        "atom_c3_ϵ",
        "atom_oh_σ",
        "atom_oh_ϵ",
        "atom_ow_σ",
        "atom_ow_ϵ",
    ]

    solute_indices = [1, 2, 3, 4]
    solvent_indices = [5, 6]

    solute_trainable_map = Dict(idx => pos for (pos, idx) in enumerate(solute_indices))
    solute_blocks = AWHGrads.optimization_active_blocks(
        param_names,
        solute_indices,
        solute_trainable_map,
        solute_indices,
        solvent_indices,
        false,
    )

    @test [block.name for block in solute_blocks] == ["Solute"]
    @test [block.kind for block in solute_blocks] == [:solute]
    @test [block.global_indices for block in solute_blocks] == [[1, 2, 3, 4]]
    @test [block.trainable_indices for block in solute_blocks] == [[1, 2, 3, 4]]

    all_indices = collect(1:length(param_names))
    all_trainable_map = Dict(idx => idx for idx in all_indices)
    alternating_blocks = AWHGrads.optimization_active_blocks(
        param_names,
        all_indices,
        all_trainable_map,
        solute_indices,
        solvent_indices,
        true,
    )

    @test [block.name for block in alternating_blocks] == ["Solute", "Solvent"]
    @test [block.kind for block in alternating_blocks] == [:solute, :solvent]
    @test [block.global_indices for block in alternating_blocks] == [[1, 2, 3, 4], [5, 6]]
end

@testset "line search residual tolerance" begin
    FT = Float32
    @test AWHGrads.line_search_noise_tolerance(FT(-5.0), FT(0.1)) == FT(0.5)
    @test AWHGrads.line_search_residual_acceptable(FT(10.0), FT(10.9), FT(0.1))
    @test !AWHGrads.line_search_residual_acceptable(FT(10.0), FT(11.2), FT(0.1))
    @test AWHGrads.default_optimization_config().max_inner_epochs == 10
end

@testset "Molly dispersion correction unit stripping" begin
    inter = Molly.LJDispersionCorrection(1.25f0u"kJ*nm^3/mol")
    stripped = ustrip(inter)

    @test stripped isa Molly.LJDispersionCorrection
    @test stripped.factor == 1.25f0
end

@testset "Molly System reconstruction bypasses launch_config forwarding" begin
    custom_launch = Molly.CUDALaunchConfig(force_block_y=4)
    sys = Molly.System(
        "ethanol_vac.pdb",
        AWHGrads.ff;
        array_type=Array,
        nonbonded_method=:none,
        launch_config=custom_launch,
    )
    sys_nounits = ustrip(Molly.from_device(sys))

    rebuilt = AWHGrads._rebuild_system_like(
        sys_nounits,
        sys_nounits.atoms,
        sys_nounits.coords,
        sys_nounits.boundary,
        sys_nounits.pairwise_inters,
        sys_nounits.specific_inter_lists,
        sys_nounits.general_inters,
    )

    @test sys_nounits.launch_config == custom_launch
    @test rebuilt.launch_config == Molly.CUDALaunchConfig()
    @test rebuilt.coords == sys_nounits.coords
    @test rebuilt.general_inters == sys_nounits.general_inters
    @test rebuilt.masses == sys_nounits.masses
    @test rebuilt.total_mass == sys_nounits.total_mass
end
