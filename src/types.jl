"""
    ForceFieldConfig

Describes how force-field XML files are resolved. Relative `xml_files` entries
are looked up under `force_field_dir`; when that is `nothing`, Molly's bundled
force-field directory is used.
"""
Base.@kwdef struct ForceFieldConfig
    force_field_dir::Union{Nothing, String} = nothing
    xml_files::Vector{String} = ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"]
end

"""
    AWHControlConfig

Collects the low-level AWH and soft-core parameters passed into each
`AWHSimulation`. These settings control how quickly the bias is updated and how
the linear stage is initialized.
"""
Base.@kwdef struct AWHControlConfig
    lj_softcore_alpha::Float64 = 0.85
    coul_softcore_alpha::Float64 = 0.3
    reuse_neighbors::Bool = true
    # Lambda-sampling interval in MD steps.
    seed_num_md_steps::Int = 10
    seed_log_freq::Int = 100
    update_freq::Int = 100
    coverage_threshold::Float64 = 0.8
    significant_weight::Float64 = 0.1
    initial_n_bias::Int = 100
    well_tempered_factor::Float64 = Inf
    coverage_type::Symbol = :physical
end

"""
    ThermodynamicLegConfig

Configuration for one leg of the thermodynamic cycle. Each leg contributes
`coefficient * ΔG_leg` to the cycle free energy and can optionally include a
`pV` correction during reweighting.
"""
Base.@kwdef struct ThermodynamicLegConfig
    name::Symbol
    pdb::String
    coefficient::Float64 = 1.0
    is_vacuum::Bool = false
    include_pv::Bool = false
    probe_time = Float32(0.5)u"ns"
    lambda_schedule::Union{Nothing, Vector{<:Real}} = nothing
    ensemble::Symbol = :npt
end

"""
    ResolvedLegStateSchedule

Concrete per-leg alchemical state schedule used at runtime after applying
fallbacks and validation.
"""
struct ResolvedLegStateSchedule{FT <: AbstractFloat}
    lambda::Vector{FT}
    coupled_state_idx::Int
    decoupled_state_idx::Int
end

"""
    ThermodynamicCycleConfig

Defines the legs that make up the free-energy cycle and the experimental target
the optimization is trying to match.
"""
Base.@kwdef struct ThermodynamicCycleConfig
    legs::Vector{ThermodynamicLegConfig} = ThermodynamicLegConfig[]
    include_standard_state_correction::Bool = true
    target_dG_kcal_mol::Float64 = -5.01
end

"""
    ParameterBoundsConfig

Hard bounds for the optimized Lennard-Jones parameters. The optimization is
performed in an unconstrained `ϕ` space and mapped back into these intervals.
"""
Base.@kwdef struct ParameterBoundsConfig
    sigma_hydrogen_min::Float64 = 0.1
    sigma_hydrogen_max::Float64 = 0.4
    sigma_heavy_min::Float64 = 0.2
    sigma_heavy_max::Float64 = 0.5
    epsilon_hydrogen_min::Float64 = 0.0
    epsilon_hydrogen_max::Float64 = 0.5
    epsilon_heavy_min::Float64 = 0.0
    epsilon_heavy_max::Float64 = 1.5
    sigma_floor::Float64 = 0.15
    epsilon_floor::Float64 = 1e-4
    reference_clamp_eps::Float64 = 1e-4
end

"""
    SimulationConfig

Top-level runtime configuration for simulation, AWH sampling, and thermodynamic
cycle setup. The defaults keep backward compatibility with the original
two-state ethanol hydration example while also supporting arbitrary cycle
definitions through `cycle`.
"""
Base.@kwdef struct SimulationConfig
    # Backend and numeric types used throughout the run.
    device_id::Int = 1
    FT::DataType = Float32
    AT::Any = CuArray

    # Integrator timestep and thermodynamic state.
    Δt = Float32(1)u"fs"
    T0 = Float32(310)u"K"
    P0 = Float32(1)u"bar"
    lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))

    # Macro-cycle timing.
    awh_budget_time = Float32(20)u"ns"
    awh_block_time = Float32(1.0)u"ns"
    md_time_production = Float32(0.1)u"ns"
    production_log_interval::Int = 100

    # Solute atoms whose nonbonded parameters are coupled to λ.
    solute_idx = 1:9

    # Backward-compatible defaults for a 2-leg hydration cycle.
    pdb_solv::String = "ethanol_solv.pdb"
    pdb_vac::String = "ethanol_vac.pdb"
    awh_probe_time_solv = Float32(1.5)u"ns"
    awh_probe_time_vac = Float32(0.25)u"ns"
    # Probe-frame controls for Stage B reweighting/evaluation only.
    awh_probe_reweight_stride_solv::Int = 5
    awh_probe_reweight_stride_vac::Int = 5
    awh_probe_reweight_min_frames_solv::Int = 1000
    awh_probe_reweight_min_frames_vac::Int = 500
    awh_probe_reweight_max_frames_solv::Int = 2000
    awh_probe_reweight_max_frames_vac::Int = 1200
    awh_probe_discard_fraction::Float64 = 0.2
    dG_exp_kcal_mol::Float64 = -5.01

    # New user-facing configuration for non-hardcoded systems/cycles.
    force_field::ForceFieldConfig = ForceFieldConfig()
    cycle::Union{Nothing, ThermodynamicCycleConfig} = nothing
    parameter_reference_leg::Union{Nothing, Symbol} = nothing
    parameter_bounds::ParameterBoundsConfig = ParameterBoundsConfig()
    awh_control::AWHControlConfig = AWHControlConfig()
end

"""
    OptimizationConfig

Controls the readiness gates and the natural-gradient-like optimization step
used after each production phase.
"""
Base.@kwdef struct OptimizationConfig
    # Macro-epoch and line-search controls.
    awh_convergence_tol = Float32(1e-3)
    rewarm_fraction = Float32(0.05)
    max_macro_epochs::Int = 30
    huber_delta = Float32(2.0)

    # Trust-region / natural-gradient step sizing.
    kl_target = Float32(0.1)
    eigenvalue_tol_scale = Float32(1e-2)
    min_phi_step = Float32(5e-4)
    max_phi_step_solute = Float32(0.35)
    max_phi_step_solvent = Float32(0.035)
    tiny_alpha_cutoff = Float32(0.015625)
    max_tiny_alpha_hits::Int = 2
    restart_rmsd_tol_nm = Float32(1e-5)
    optimize_solvent::Bool = false

    # Stage A / Stage B readiness thresholds.
    ess_threshold_scale = Float32(0.22)
    awh_min_linear_neff::Int = 3000
    awh_min_lambda_ess::Int = 300
    awh_split_tol_kT = Float32(0.5)
    awh_parity_tol_kT = Float32(0.25)
    awh_tail_lag::Int = 10
    awh_min_round_trips::Int = 3
    awh_endpoint_target_ratio = Float32(0.3)
    awh_stageA_stable_blocks::Int = 2
    awh_stageB_cooldown_blocks::Int = 2

    k_sigmoid = Float32(1.0)
end

"""
    RuntimeState

Mutable state carried across macro epochs. It stores the current bias estimate,
warm-start restart snapshots, and the latest parameter vectors.
"""
Base.@kwdef mutable struct RuntimeState
    active_bias::Dict{Symbol, Any} = Dict{Symbol, Any}()
    restart_cache::Dict{Symbol, Any} = Dict{Symbol, Any}()

    # Backward-compatible aliases for the default two-leg cycle.
    active_bias_solv::Any = nothing
    active_bias_vac::Any = nothing
    restart_cache_solv::Any = nothing
    restart_cache_vac::Any = nothing

    phi_active::Any = nothing
    theta_active::Any = nothing
end

"""
    LegArtifacts

Production data cached for one thermodynamic leg and reused during the
optimization phase.
"""
Base.@kwdef mutable struct LegArtifacts
    name::Symbol = :unknown
    coefficient::Any = 0.0
    include_pv::Bool = false
    p0_energy_per_vol::Any = 0.0
    n_states::Int = 0
    coupled_state_idx::Int = 1
    decoupled_state_idx::Int = 1

    awh_prod::Any = nothing
    logger_prod::Any = nothing
    neighbors::Any = nothing
    u_ref::Any = nothing
    sys_base::Any = nothing
    active_bias::Any = nothing
    idxs::Any = nothing
end

"""
    StageAStats

Summary of the cheap readiness checks computed after each AWH block. These are
used to decide when a leg is mature enough to run a Stage B probe.
"""
Base.@kwdef struct StageAStats
    ready::Bool = false
    df_ready::Bool = false
    df_mean::Any = Inf
    lambda_ess::Any = 1.0
    tau_int_est::Any = 0.0
    lambda_ess_ready::Bool = false
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
    lambda_ess = nt.lambda_ess,
    tau_int_est = nt.tau_int_est,
    lambda_ess_ready = nt.lambda_ess_ready,
    linear_neff = nt.linear_neff,
    neff_ready = nt.neff_ready,
    round_trips = nt.round_trips,
    round_trip_ready = nt.round_trip_ready,
    endpoint_low = nt.endpoint_low,
    endpoint_high = nt.endpoint_high,
    endpoint_ready = nt.endpoint_ready,
    n_hist = nt.n_hist,
)

"""
    StageBStats

Summary of the more expensive probe-based readiness checks that validate the
split-half consistency and AWH/MBAR parity of a leg.
"""
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
