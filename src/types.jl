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
    lj_softcore_alpha::Float64 = 1.5
    coul_softcore_alpha::Float64 = 0.3
    reuse_neighbors::Bool = true
    # Lambda-sampling interval in MD steps.
    seed_num_md_steps::Int = 10
    # Target MD-step cadence for AWH bias updates when `update_freq` is left automatic.
    bias_update_interval_md_steps::Int = 1000
    # Log Molly's internal AWH stats every N bias updates.
    stats_log_every_updates::Int = 1
    # Optional low-level Molly override measured in lambda samples per bias update.
    update_freq::Union{Nothing, Int} = nothing
    coverage_threshold::Float64 = 1.0
    significant_weight::Float64 = 0.1
    initial_n_bias::Int = 10
    well_tempered_factor::Float64 = Inf
    coverage_type::Symbol = :physical
end

"""
    StageAReadinessPolicyConfig

Configuration for the cheap Stage A readiness checks attached to a single
thermodynamic leg. Policies stay leg-local, but the criteria are now selected
explicitly rather than inferred from hard-coded solvent/vacuum branches.
"""
Base.@kwdef struct StageAReadinessPolicyConfig
    preset::Symbol = :generic_alchemical
    endpoint_state_idxs::Union{Nothing, Vector{Int}} = nothing
    hotspot_state_idxs::Union{Nothing, Vector{Int}} = nothing
    hotspot_lj_lambda_max::Union{Nothing, Float64} = nothing
    hotspot_min_state_occupancy_floor::Union{Nothing, Float64} = nothing
    endpoint_high_min_fraction_abs::Union{Nothing, Float64} = nothing
end

struct ResolvedStageAReadinessPolicy{FT <: AbstractFloat}
    preset::Symbol
    endpoint_state_idxs::Vector{Int}
    hotspot_state_idxs::Vector{Int}
    hotspot_min_state_occupancy_floor::FT
    endpoint_high_min_fraction_abs::FT
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
    probe_time = Float32(3.0)u"ns"
    lambda_schedule::Union{Nothing, Vector{<:Real}} = nothing
    ensemble::Symbol = :npt
    awh_seed_num_md_steps::Union{Nothing, Int} = nothing
    awh_bias_update_interval_md_steps::Union{Nothing, Int} = nothing
    awh_initial_n_bias::Union{Nothing, Int} = nothing
    probe_awh_seed_num_md_steps::Union{Nothing, Int} = nothing
    electrostatics_method::Union{Nothing, Symbol} = nothing
    lambda_scheduler::Union{Nothing, Symbol} = nothing
    coulomb_softcore_model::Union{Nothing, Symbol} = nothing
    lj_softcore_model::Union{Nothing, Symbol} = nothing
    readiness_policy::StageAReadinessPolicyConfig = StageAReadinessPolicyConfig()
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

abstract type AbstractTrainingTarget end

"""
    CycleFreeEnergyTarget

Objective term that matches the coefficient-weighted thermodynamic-cycle free
energy assembled from the configured legs. When `target_dG_kcal_mol` is left as
`nothing`, the cycle-level target stored on `ThermodynamicCycleConfig` is used.
"""
Base.@kwdef struct CycleFreeEnergyTarget <: AbstractTrainingTarget
    name::Symbol = :cycle_free_energy
    target_dG_kcal_mol::Union{Nothing, Float64} = nothing
    tolerance_kcal_mol::Union{Nothing, Float64} = nothing
    weight::Float64 = 1.0
end

"""
    StateObservableTarget

Objective term that matches a scalar observable evaluated on one replayed
physical state of one thermodynamic leg. The observable itself is any callable
that accepts a replayed `Molly.System` and returns a scalar.
"""
Base.@kwdef struct StateObservableTarget{O, S, T <: Real} <: AbstractTrainingTarget
    name::Symbol
    leg::Symbol
    state::S = :coupled
    observable::O
    target_value::T
    tolerance::Union{Nothing, Float64} = nothing
    weight::Float64 = 1.0
    unit_label::String = ""
end

struct ResolvedCycleFreeEnergyTarget <: AbstractTrainingTarget
    name::Symbol
    target_dG_kcal_mol::Float64
    tolerance_kcal_mol::Union{Nothing, Float64}
    weight::Float64
end

struct ResolvedStateObservableTarget{O, T <: Real} <: AbstractTrainingTarget
    name::Symbol
    leg::Symbol
    state_idx::Int
    state_label::Symbol
    observable::O
    target_value::T
    tolerance::Union{Nothing, Float64}
    weight::Float64
    unit_label::String
end

"""
    ParameterPoolConfig

Define one trainable nonbonded-parameter pool. Pools are selected by explicit
atom metadata rules rather than by hard-coded molecular identities, allowing
the same optimization machinery to cover hydration, binding, and other
alchemical free-energy workflows.
"""
Base.@kwdef struct ParameterPoolConfig
    name::Symbol
    atom_indices::Union{Nothing, AbstractVector{Int}} = nothing
    atom_types::Vector{String} = String[]
    atom_type_patterns::Vector{String} = String[]
    atom_names::Vector{String} = String[]
    residue_names::Vector{String} = String[]
    residue_numbers::Vector{Int} = Int[]
    molecule_ids::Vector{Int} = Int[]
    trainable_families::Vector{Symbol} = Symbol[:sigma, :epsilon]
    exclude_atom_types::Vector{String} = String[]
    max_phi_step::Union{Nothing, Float64} = nothing
    max_sigma_drift::Union{Nothing, Float64} = nothing
    max_epsilon_drift::Union{Nothing, Float64} = nothing
    reference_penalty_strength::Float64 = 0.0
end

"""
    ChargeTrainingConfig

Controls optional charge-equilibration training on top of the existing
Lennard-Jones parameter optimization.
"""
Base.@kwdef struct ChargeTrainingConfig
    enabled::Bool = false
    typing_basis::Symbol = :atom_class
    net_charge_constraint::Symbol = :molecule
    hardness_floor::Float64 = 1e-3
    reference_hardness::Float64 = 1.0
end

"""
    ParameterBoundsConfig

Hard bounds for the optimized Lennard-Jones parameters. The optimization is
performed in an unconstrained `ϕ` space and mapped back into these intervals.
"""
Base.@kwdef struct ParameterBoundsConfig
    sigma_hydrogen_min::Float64 = 0.05
    sigma_hydrogen_max::Float64 = 0.4
    sigma_heavy_min::Float64 = 0.2
    sigma_heavy_max::Float64 = 0.5
    epsilon_hydrogen_min::Float64 = 0.0
    epsilon_hydrogen_max::Float64 = 0.5
    epsilon_heavy_min::Float64 = 0.0
    epsilon_heavy_max::Float64 = 1.5
    sigma_floor::Float64 = 0.1
    epsilon_floor::Float64 = 1e-6
    reference_clamp_eps::Float64 = 1e-4
end

"""
    EnsembleEvalConfig

Controls the offline ensemble-evaluation pass used during Stage B analysis and
the optimization phase. These knobs trade off wall-clock throughput against the
temporary working-set size of the CPU replay.
"""
Base.@kwdef struct EnsembleEvalConfig
    threads::Int = Threads.nthreads()
    lambda_tile::Int = 4
    schedule::Symbol = :dynamic
    cache_unitless_frames::Bool = true
    cache_unitless_templates::Bool = true
    progress::Bool = false
    progress_interval_seconds::Float64 = 30.0
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
    nonbonded_energy_type::Any = default_nonbonded_energy_type(Float32)

    # Integrator timestep and thermodynamic state.
    Δt = Float32(1)u"fs"
    T0 = Float32(293.15)u"K"
    P0 = Float32(1)u"bar"
    lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))

    # Macro-cycle timing.
    awh_budget_time = Float32(20)u"ns"
    awh_block_time = Float32(1.0)u"ns"
    md_time_production = Float32(0.5)u"ns"
    md_time_production_solv = Float32(1.0)u"ns"
    md_time_production_vac = nothing
    production_log_interval::Int = 100
    production_reweight_stride_solv::Int = 5
    production_reweight_stride_vac::Int = 5
    production_reweight_min_frames_solv::Int = 1000
    production_reweight_min_frames_vac::Int = 500
    production_reweight_max_frames_solv::Int = 3000
    production_reweight_max_frames_vac::Int = 1500
    production_discard_fraction::Float64 = 0.0
    production_segments_solv::Int = 1
    production_segments_vac::Int = 1

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
    targets::Union{Nothing, AbstractVector} = nothing
    parameter_reference_leg::Union{Nothing, Symbol} = nothing
    parameter_pools::Vector{ParameterPoolConfig} = ParameterPoolConfig[]
    charge_training::ChargeTrainingConfig = ChargeTrainingConfig()
    parameter_bounds::ParameterBoundsConfig = ParameterBoundsConfig()
    awh_control::AWHControlConfig = AWHControlConfig()
    ensemble_eval::EnsembleEvalConfig = EnsembleEvalConfig()
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
    max_inner_epochs::Int = 10
    huber_delta = Float32(2.0)
    default_target_relative_tolerance = Float32(0.05)
    cycle_target_absolute_tolerance_kcal_mol = Float32(0.1)
    observable_target_absolute_tolerance = Float32(1e-3)

    # Trust-region / natural-gradient step sizing.
    kl_target = Float32(0.25)
    eigenvalue_tol_scale = Float32(1e-3)
    min_phi_step = Float32(5e-4)
    max_phi_step_solute = Float32(0.6)
    max_phi_step_solvent = Float32(0.035)
    line_search_noise_tolerance_fraction = Float32(0.1)
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
    awh_parity_gate_mode::Symbol = :support_aware_max
    awh_parity_support_threshold = Float32(300)
    awh_stageB_support_allow_missing::Int = 3
    awh_parity_near_pass_factor = Float32(2.0)
    awh_tail_lag::Int = 10
    awh_min_round_trips::Int = 3
    awh_endpoint_target_ratio = Float32(0.3)
    awh_solvent_tail_lj_max = Float32(0.3025)
    awh_solvent_tail_min_state_occupancy = Float32(0.0125)
    # Optional absolute lower bound for solvent endpoint-band occupancy.
    awh_solvent_endpoint_min_fraction = Float32(0.0)
    # Number of recent Stage A blocks used for occupancy-based readiness metrics.
    awh_stageA_history_blocks::Int = 8
    awh_stageA_stable_blocks::Int = 2
    awh_stageB_cooldown_blocks::Int = 2
    awh_stageB_near_pass_cooldown_blocks::Int = 0
    awh_stageB_probe_growth_factor = Float32(1.5)
    awh_stageB_probe_near_pass_scale = Float32(0.5)
    awh_stageB_probe_near_pass_mode_solvent::Symbol = :keep
    awh_stageB_probe_near_pass_mode_vacuum::Symbol = :shrink
    awh_stageB_probe_max_factor = Float32(4.0)
    awh_stageB_probe_growth_ns = Float32(2.0)
    awh_stageB_split_extension_enabled::Bool = true
    awh_stageB_split_extension_max_segments::Int = 3

    awh_stageB_soften_failures_threshold::Int = 3
    awh_stageB_soften_factor = Float32(0.5)

    awh_stageA_streak_growth_factor = Float32(1.5)
    awh_stageB_cooldown_growth_factor = Float32(1.5)
    awh_stageA_max_streak::Int = 10
    awh_stageB_max_cooldown::Int = 10

    optimization_confidence_mode::Symbol = :split_half
    optimization_confidence_min_frames::Int = 200
    optimization_confidence_min_scale = Float32(0.25)
    optimization_confidence_scale_strength = Float32(1.0)
    optimization_confidence_residual_requirement_strength = Float32(0.25)

    readiness_log_mode::Symbol = :concise
    awh_stageB_repeat_suppression::Bool = true
    optimization_log_mode::Symbol = :default

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
    ResolvedParameterPool

Internal runtime metadata for one trainable parameter pool after resolving the
selector rules on the parameter reference leg.
"""
Base.@kwdef struct ResolvedParameterPool
    name::Symbol
    global_indices::Vector{Int} = Int[]
    sigma_global_indices::Vector{Int} = Int[]
    epsilon_global_indices::Vector{Int} = Int[]
    charge_chi_global_indices::Vector{Int} = Int[]
    charge_eta_global_indices::Vector{Int} = Int[]
    max_phi_step::Float64 = Inf
    max_sigma_drift::Union{Nothing, Float64} = nothing
    max_epsilon_drift::Union{Nothing, Float64} = nothing
    reference_penalty_strength::Float64 = 0.0
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
    eval_cache::Any = nothing
    raw_frame_count::Int = 0
    selected_frame_indices::Vector{Int} = Int[]
    n_production_segments::Int = 1
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
    switch_count::Int = 0
    mean_residence::Any = 0.0
    median_residence::Any = 0.0
    round_trips::Int = 0
    round_trip_ready::Bool = false
    endpoint_low::Any = 0.0
    # For staged solvent legs this is the summed occupancy of the endpoint band.
    endpoint_high::Any = 0.0
    endpoint_high_required::Any = 0.0
    endpoint_ready::Bool = false
    hotspot_occupancy::Any = 0.0
    hotspot_min_state_occupancy::Any = 0.0
    hotspot_ready::Bool = true
    hotspot_low_occupancy_states::Vector{Int} = Int[]
    tail_occupancy::Any = 0.0
    tail_min_state_occupancy::Any = 0.0
    tail_ready::Bool = true
    tail_low_occupancy_states::Vector{Int} = Int[]
    min_state_occupancy::Any = 0.0
    low_occupancy_states::Vector{Int} = Int[]
    n_hist::Int = 0
    n_hist_recent::Int = 0
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
    switch_count = hasproperty(nt, :switch_count) ? nt.switch_count : 0,
    mean_residence = hasproperty(nt, :mean_residence) ? nt.mean_residence : 0.0,
    median_residence = hasproperty(nt, :median_residence) ? nt.median_residence : 0.0,
    round_trips = nt.round_trips,
    round_trip_ready = nt.round_trip_ready,
    endpoint_low = nt.endpoint_low,
    endpoint_high = nt.endpoint_high,
    endpoint_high_required = hasproperty(nt, :endpoint_high_required) ? nt.endpoint_high_required : nt.endpoint_high,
    endpoint_ready = nt.endpoint_ready,
    hotspot_occupancy = hasproperty(nt, :hotspot_occupancy) ? nt.hotspot_occupancy : (hasproperty(nt, :tail_occupancy) ? nt.tail_occupancy : 0.0),
    hotspot_min_state_occupancy = hasproperty(nt, :hotspot_min_state_occupancy) ? nt.hotspot_min_state_occupancy : (hasproperty(nt, :tail_min_state_occupancy) ? nt.tail_min_state_occupancy : 0.0),
    hotspot_ready = hasproperty(nt, :hotspot_ready) ? nt.hotspot_ready : (hasproperty(nt, :tail_ready) ? nt.tail_ready : true),
    hotspot_low_occupancy_states = hasproperty(nt, :hotspot_low_occupancy_states) ? nt.hotspot_low_occupancy_states : (hasproperty(nt, :tail_low_occupancy_states) ? nt.tail_low_occupancy_states : Int[]),
    tail_occupancy = hasproperty(nt, :tail_occupancy) ? nt.tail_occupancy : (hasproperty(nt, :hotspot_occupancy) ? nt.hotspot_occupancy : 0.0),
    tail_min_state_occupancy = hasproperty(nt, :tail_min_state_occupancy) ? nt.tail_min_state_occupancy : (hasproperty(nt, :hotspot_min_state_occupancy) ? nt.hotspot_min_state_occupancy : 0.0),
    tail_ready = hasproperty(nt, :tail_ready) ? nt.tail_ready : (hasproperty(nt, :hotspot_ready) ? nt.hotspot_ready : true),
    tail_low_occupancy_states = hasproperty(nt, :tail_low_occupancy_states) ? nt.tail_low_occupancy_states : (hasproperty(nt, :hotspot_low_occupancy_states) ? nt.hotspot_low_occupancy_states : Int[]),
    min_state_occupancy = hasproperty(nt, :min_state_occupancy) ? nt.min_state_occupancy : 0.0,
    low_occupancy_states = hasproperty(nt, :low_occupancy_states) ? nt.low_occupancy_states : Int[],
    n_hist = nt.n_hist,
    n_hist_recent = hasproperty(nt, :n_hist_recent) ? nt.n_hist_recent : nt.n_hist,
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
    raw_parity_gap::Any = Inf
    supported_parity_gap::Any = Inf
    endpoint_parity_gap::Any = Inf
    endpoint_parity_ready::Bool = false
    support_coverage_ready::Bool = false
    n_total_states::Int = 0
    n_frames::Int = 0
    dG_half_1::Any = NaN
    dG_half_2::Any = NaN
    n_accumulated_frames::Int = 0
    n_probe_segments::Int = 0
    diagnostics::String = ""
    parity_worst_state_idx::Int = 0
    parity_worst_state_residual::Any = NaN
    n_supported_states::Int = 0
    required_supported_states::Int = 0
    support_threshold::Any = 0.0
    failure_mode::Symbol = :not_checked
    near_pass::Bool = false
    accumulation_mode::Symbol = :single_probe
    support_switch_ready::Bool = false
    n_evicted_frames::Int = 0
end

StageBStats(nt::NamedTuple) = StageBStats(
    ready = nt.ready,
    split_ready = nt.split_ready,
    split_gap = nt.split_gap,
    parity_ready = nt.parity_ready,
    parity_gap = nt.parity_gap,
    raw_parity_gap = hasproperty(nt, :raw_parity_gap) ? nt.raw_parity_gap : nt.parity_gap,
    supported_parity_gap = hasproperty(nt, :supported_parity_gap) ? nt.supported_parity_gap : nt.parity_gap,
    endpoint_parity_gap = hasproperty(nt, :endpoint_parity_gap) ? nt.endpoint_parity_gap : nt.parity_gap,
    endpoint_parity_ready = hasproperty(nt, :endpoint_parity_ready) ? nt.endpoint_parity_ready : nt.parity_ready,
    support_coverage_ready = hasproperty(nt, :support_coverage_ready) ? nt.support_coverage_ready : false,
    n_total_states = hasproperty(nt, :n_total_states) ? nt.n_total_states : 0,
    n_frames = nt.n_frames,
    dG_half_1 = nt.dG_half_1,
    dG_half_2 = nt.dG_half_2,
    n_accumulated_frames = hasproperty(nt, :n_accumulated_frames) ? nt.n_accumulated_frames : nt.n_frames,
    n_probe_segments = hasproperty(nt, :n_probe_segments) ? nt.n_probe_segments : (nt.n_frames > 0 ? 1 : 0),
    diagnostics = hasproperty(nt, :diagnostics) ? nt.diagnostics : "",
    parity_worst_state_idx = hasproperty(nt, :parity_worst_state_idx) ? nt.parity_worst_state_idx : 0,
    parity_worst_state_residual = hasproperty(nt, :parity_worst_state_residual) ? nt.parity_worst_state_residual : NaN,
    n_supported_states = hasproperty(nt, :n_supported_states) ? nt.n_supported_states : 0,
    required_supported_states = hasproperty(nt, :required_supported_states) ? nt.required_supported_states : 0,
    support_threshold = hasproperty(nt, :support_threshold) ? nt.support_threshold : 0.0,
    failure_mode = hasproperty(nt, :failure_mode) ? nt.failure_mode : :not_checked,
    near_pass = hasproperty(nt, :near_pass) ? nt.near_pass : false,
    accumulation_mode = hasproperty(nt, :accumulation_mode) ? nt.accumulation_mode : :single_probe,
    support_switch_ready = hasproperty(nt, :support_switch_ready) ? nt.support_switch_ready : false,
    n_evicted_frames = hasproperty(nt, :n_evicted_frames) ? nt.n_evicted_frames : 0,
)
