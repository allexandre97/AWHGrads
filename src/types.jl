Base.@kwdef struct SimulationConfig
    device_id::Int = 1
    FT::DataType = Float32
    AT::Any = CuArray
    Δt = Float32(1)u"fs"
    T0 = Float32(310)u"K"
    P0 = Float32(1)u"bar"
    lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
    awh_budget_time = Float32(20)u"ns"
    awh_block_time = Float32(1.0)u"ns"
    awh_probe_time_solv = Float32(0.75)u"ns"
    awh_probe_time_vac = Float32(0.25)u"ns"
    md_time_production = Float32(0.1)u"ns"
    production_log_interval::Int = 100
    solute_idx = 1:9
    pdb_solv::String = "ethanol_solv.pdb"
    pdb_vac::String = "ethanol_vac.pdb"
    dG_exp_kcal_mol::Float64 = -5.01
end

Base.@kwdef struct OptimizationConfig
    awh_convergence_tol = Float32(1e-3)
    rewarm_fraction = Float32(0.05)
    max_macro_epochs::Int = 30
    huber_delta = Float32(2.0)

    kl_target = Float32(0.1)
    eigenvalue_tol_scale = Float32(1e-2)
    min_phi_step = Float32(5e-4)
    max_phi_step_solute = Float32(0.35)
    max_phi_step_solvent = Float32(0.035)
    tiny_alpha_cutoff = Float32(0.015625)
    max_tiny_alpha_hits::Int = 2
    restart_rmsd_tol_nm = Float32(1e-5)
    optimize_solvent::Bool = false

    ess_threshold_scale = Float32(0.22)
    awh_min_linear_neff::Int = 3000
    awh_split_tol_kT = Float32(0.5)
    awh_parity_tol_kT = Float32(0.1)
    awh_tail_lag::Int = 10
    awh_min_round_trips::Int = 3
    awh_endpoint_min_fraction = Float32(0.03)
    awh_stageA_stable_blocks::Int = 2

    k_sigmoid = Float32(1.0)
end

Base.@kwdef mutable struct RuntimeState
    active_bias_solv::Any = nothing
    active_bias_vac::Any = nothing
    restart_cache_solv::Any = nothing
    restart_cache_vac::Any = nothing
    phi_active::Any = nothing
    theta_active::Any = nothing
end

Base.@kwdef mutable struct LegArtifacts
    awh_prod::Any = nothing
    logger_prod::Any = nothing
    neighbors::Any = nothing
    u_ref::Any = nothing
end

Base.@kwdef struct StageAStats
    ready::Bool = false
    df_ready::Bool = false
    df_mean::Any = Inf
    linear_neff::Any = 0.0
    neff_ready::Bool = false
    round_trips::Int = 0
    round_trip_ready::Bool = false
    endpoint_low::Any = 0.0
    endpoint_high::Any = 0.0
    endpoint_ready::Bool = false
    n_hist::Int = 0
end

StageAStats(nt::NamedTuple) = StageAStats(
    ready = nt.ready,
    df_ready = nt.df_ready,
    df_mean = nt.df_mean,
    linear_neff = nt.linear_neff,
    neff_ready = nt.neff_ready,
    round_trips = nt.round_trips,
    round_trip_ready = nt.round_trip_ready,
    endpoint_low = nt.endpoint_low,
    endpoint_high = nt.endpoint_high,
    endpoint_ready = nt.endpoint_ready,
    n_hist = nt.n_hist,
)

Base.@kwdef struct StageBStats
    ready::Bool = false
    split_ready::Bool = false
    split_gap::Any = Inf
    parity_ready::Bool = false
    parity_gap::Any = Inf
    n_frames::Int = 0
    dG_half_1::Any = NaN
    dG_half_2::Any = NaN
end

StageBStats(nt::NamedTuple) = StageBStats(
    ready = nt.ready,
    split_ready = nt.split_ready,
    split_gap = nt.split_gap,
    parity_ready = nt.parity_ready,
    parity_gap = nt.parity_gap,
    n_frames = nt.n_frames,
    dG_half_1 = nt.dG_half_1,
    dG_half_2 = nt.dG_half_2,
)
