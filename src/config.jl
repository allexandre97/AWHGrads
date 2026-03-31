"""
    default_solvent_leg_lambda_schedule(FT=Float32)

Return the staged 31-state global λ schedule for the solvent leg. Under
Molly's `DefaultLambdaScheduler` and the current `InsertRole` convention, this
encodes charge-first, LJ-second decoupling while still using only atom λ
values.
"""
function _charge_stage_global_lambda(
    elec_stage::AbstractVector{FT},
    lambda_scheduler::Symbol,
) where {FT <: AbstractFloat}
    if lambda_scheduler in (:default, :namd)
        return (elec_stage .+ one(FT)) ./ FT(2)
    elseif lambda_scheduler == :ele_scaled
        return ((elec_stage .^ 2) .+ one(FT)) ./ FT(2)
    end
    throw(ArgumentError("Staged solvent schedules only support :default, :namd, and :ele_scaled schedulers; got `$lambda_scheduler`."))
end

function default_solvent_leg_lambda_schedule(::Type{FT}=Float32; lambda_scheduler::Symbol=:default) where {FT <: AbstractFloat}
    charge_stage = collect(FT.(range(one(FT), stop=zero(FT), length=11)))
    lj_stage = reverse(FT.(collect(range(zero(FT), stop=one(FT), length=21)) .^ 2))[2:end]
    return vcat(_charge_stage_global_lambda(charge_stage, lambda_scheduler), lj_stage ./ FT(2))
end

"""
    dense_solvent_leg_lambda_schedule(FT=Float32)

Return a staged 35-state solvent-leg global λ schedule that preserves the
charge-first, LJ-second semantics while concentrating charge-stage windows in
the problematic partially discharged electrostatic regime without over-densifying
the easy ends of the path. The charge-stage atom λ values are derived from the
chosen scheduler so the effective electrostatic λ ladder matches the intended
charge-stage windows.
"""
function dense_solvent_leg_lambda_schedule(::Type{FT}=Float32; lambda_scheduler::Symbol=:default) where {FT <: AbstractFloat}
    elec_stage = FT[
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.0,
    ]
    lj_stage = reverse(FT.(collect(range(zero(FT), stop=one(FT), length=21)) .^ 2))[2:end]
    return vcat(_charge_stage_global_lambda(elec_stage, lambda_scheduler), lj_stage ./ FT(2))
end

"""
    default_cycle_config(; target_dG_kcal_mol=-5.01, FT=Float32)

Return the built-in two-leg ethanol hydration cycle used by the example
scripts.
"""
function default_cycle_config(; target_dG_kcal_mol::Real=-5.01, FT::DataType=Float32)
    solvent_lambda_schedule = dense_solvent_leg_lambda_schedule(FT; lambda_scheduler=:ele_scaled)
    legs = [
        ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            coefficient=1.0,
            is_vacuum=false,
            include_pv=true,
            probe_time=FT(1.5)u"ns",
            lambda_schedule=solvent_lambda_schedule,
            ensemble=:npt,
            probe_awh_seed_num_md_steps=1000,
            lambda_scheduler=:ele_scaled,
            coulomb_softcore_model=:gapsys_rf,
            lj_softcore_model=:beutler,
        ),
        ThermodynamicLegConfig(
            name=:vacuum,
            pdb="ethanol_vac.pdb",
            coefficient=-1.0,
            is_vacuum=true,
            include_pv=false,
            probe_time=FT(0.25)u"ns",
        ),
    ]
    return ThermodynamicCycleConfig(
        legs=legs,
        include_standard_state_correction=true,
        target_dG_kcal_mol=Float64(target_dG_kcal_mol),
    )
end

"""
    default_simulation_config(; FT=Float32, AT=CuArray, device_id=1)

Construct a `SimulationConfig` populated with the package defaults.
"""
function default_simulation_config(; FT::DataType=Float32, AT=CuArray, device_id::Int=1)
    return SimulationConfig(
        device_id=device_id,
        FT=FT,
        AT=AT,
        Δt=FT(1)u"fs",
        T0=FT(310)u"K",
        P0=FT(1)u"bar",
        lambda_schedule=FT.(range(1.0, stop=0.0, length=21)),
        awh_budget_time=FT(20)u"ns",
        awh_block_time=FT(0.5)u"ns",
        md_time_production=FT(0.5)u"ns",
        production_log_interval=100,
        solute_idx=1:9,
        pdb_solv="ethanol_solv.pdb",
        pdb_vac="ethanol_vac.pdb",
        awh_probe_time_solv=FT(1.5)u"ns",
        awh_probe_time_vac=FT(0.25)u"ns",
        awh_probe_reweight_stride_solv=5,
        awh_probe_reweight_stride_vac=5,
        awh_probe_reweight_min_frames_solv=1000,
        awh_probe_reweight_min_frames_vac=500,
        awh_probe_reweight_max_frames_solv=2000,
        awh_probe_reweight_max_frames_vac=1200,
        awh_probe_discard_fraction=0.2,
        dG_exp_kcal_mol=-5.01,
        force_field=ForceFieldConfig(),
        cycle=nothing,
        parameter_reference_leg=nothing,
        parameter_bounds=ParameterBoundsConfig(),
        awh_control=AWHControlConfig(),
    )
end

"""
    default_optimization_config(; FT=Float32)

Construct an `OptimizationConfig` tuned for the default hydration example.
"""
function default_optimization_config(; FT::DataType=Float32)
    return OptimizationConfig(
        awh_convergence_tol=FT(1e-3),
        rewarm_fraction=FT(0.05),
        max_macro_epochs=30,
        max_inner_epochs=10,
        huber_delta=FT(2.0),
        kl_target=FT(0.25),
        eigenvalue_tol_scale=FT(1e-3),
        min_phi_step=FT(5e-4),
        max_phi_step_solute=FT(0.6),
        max_phi_step_solvent=FT(0.035),
        line_search_noise_tolerance_fraction=FT(0.1),
        tiny_alpha_cutoff=FT(0.015625),
        max_tiny_alpha_hits=2,
        restart_rmsd_tol_nm=FT(1e-5),
        optimize_solvent=false,
        ess_threshold_scale=FT(0.22),
        awh_min_linear_neff=3000,
        awh_min_lambda_ess=300,
        awh_split_tol_kT=FT(0.5),
        awh_parity_tol_kT=FT(0.25),
        awh_parity_gate_mode=:support_aware_max,
        awh_parity_support_threshold=FT(300),
        awh_stageB_support_allow_missing=3,
        awh_parity_near_pass_factor=FT(2.0),
        awh_tail_lag=10,
        awh_min_round_trips=3,
        awh_endpoint_target_ratio=FT(0.3),
        awh_stageA_stable_blocks=2,
        awh_stageB_cooldown_blocks=2,
        awh_stageB_near_pass_cooldown_blocks=0,
        awh_stageB_probe_growth_factor=FT(1.5),
        awh_stageB_probe_near_pass_scale=FT(0.5),
        awh_stageB_probe_max_factor=FT(4.0),
        awh_stageA_streak_growth_factor=FT(1.5),
        awh_stageB_cooldown_growth_factor=FT(1.5),
        awh_stageA_max_streak=10,
        awh_stageB_max_cooldown=10,
        k_sigmoid=FT(1.0),
    )
end

"""
    resolved_cycle_config(sim_cfg)

Return the explicit cycle stored in `sim_cfg.cycle` when present, otherwise
rebuild the legacy two-leg cycle from the backward-compatible fields on
`SimulationConfig`.
"""
function resolved_cycle_config(sim_cfg::SimulationConfig)
    if !isnothing(sim_cfg.cycle) && !isempty(sim_cfg.cycle.legs)
        return sim_cfg.cycle
    end

    solvent_lambda_schedule = dense_solvent_leg_lambda_schedule(sim_cfg.FT; lambda_scheduler=:ele_scaled)
    return ThermodynamicCycleConfig(
        legs=[
            ThermodynamicLegConfig(
                name=:solvent,
                pdb=sim_cfg.pdb_solv,
                coefficient=1.0,
                is_vacuum=false,
                include_pv=true,
            probe_time=sim_cfg.awh_probe_time_solv,
            lambda_schedule=solvent_lambda_schedule,
            ensemble=:npt,
            probe_awh_seed_num_md_steps=1000,
            lambda_scheduler=:ele_scaled,
            coulomb_softcore_model=:gapsys_rf,
            lj_softcore_model=:beutler,
        ),
        ThermodynamicLegConfig(
            name=:vacuum,
            pdb=sim_cfg.pdb_vac,
            coefficient=-1.0,
                is_vacuum=true,
                include_pv=false,
                probe_time=sim_cfg.awh_probe_time_vac,
            ),
        ],
        include_standard_state_correction=true,
        target_dG_kcal_mol=sim_cfg.dG_exp_kcal_mol,
    )
end

function resolve_force_field_path(xml_file::String, ff_dir::AbstractString)
    if isabspath(xml_file)
        return xml_file
    end

    repo_root = normpath(joinpath(@__DIR__, ".."))
    default_ff_dir = joinpath(dirname(pathof(Molly)), "..", "data", "force_fields")
    candidates = unique([
        joinpath(ff_dir, xml_file),
        joinpath(default_ff_dir, xml_file),
        joinpath(repo_root, xml_file),
        joinpath(repo_root, "data", "force_fields", xml_file),
        joinpath(repo_root, "..", "AWH", xml_file),
    ])
    for candidate in candidates
        if isfile(candidate)
            return candidate
        end
    end
    return first(candidates)
end

function _validate_leg_ensemble(leg::ThermodynamicLegConfig)
    if !(leg.ensemble in (:npt, :nvt))
        throw(ArgumentError("Leg $(leg.name) has unsupported ensemble=$(leg.ensemble). Supported values are :npt and :nvt."))
    end
    if !leg.is_vacuum && leg.ensemble == :nvt && leg.include_pv
        throw(ArgumentError("Leg $(leg.name) cannot set include_pv=true when ensemble=:nvt."))
    end
    return nothing
end

function _validate_leg_alchemical_path(leg::ThermodynamicLegConfig)
    if !isnothing(leg.lambda_scheduler) && !(leg.lambda_scheduler in (:default, :namd, :quarters, :ele_scaled))
        throw(ArgumentError("Leg $(leg.name) has unsupported lambda_scheduler=$(leg.lambda_scheduler). Supported values are :default, :namd, :quarters, and :ele_scaled."))
    end
    if !isnothing(leg.coulomb_softcore_model) && !(leg.coulomb_softcore_model in (:beutler, :gapsys, :beutler_rf, :gapsys_rf))
        throw(ArgumentError("Leg $(leg.name) has unsupported coulomb_softcore_model=$(leg.coulomb_softcore_model). Supported values are :beutler, :gapsys, :beutler_rf, and :gapsys_rf."))
    end
    if !isnothing(leg.lj_softcore_model) && !(leg.lj_softcore_model in (:beutler, :gapsys))
        throw(ArgumentError("Leg $(leg.name) has unsupported lj_softcore_model=$(leg.lj_softcore_model). Supported values are :beutler and :gapsys."))
    end
    return nothing
end

"""
    resolve_leg_state_schedule(leg, default_lambda_schedule, FT=Float32)

Resolve a leg's runtime global λ schedule, preferring the leg-specific
override when present and otherwise falling back to `default_lambda_schedule`.
"""
function resolve_leg_state_schedule(
    leg::ThermodynamicLegConfig,
    default_lambda_schedule,
    ::Type{FT}=Float32,
) where {FT <: AbstractFloat}
    _validate_leg_ensemble(leg)
    _validate_leg_alchemical_path(leg)
    schedule_source = isnothing(leg.lambda_schedule) ? default_lambda_schedule : leg.lambda_schedule
    validate_lambda_schedule(schedule_source)
    lambda_values = FT.(collect(schedule_source))

    atol = sqrt(eps(FT))
    coupled_candidates = [
        idx for idx in eachindex(lambda_values)
        if isapprox(lambda_values[idx], one(FT); atol=atol, rtol=atol)
    ]
    decoupled_candidates = [
        idx for idx in eachindex(lambda_values)
        if isapprox(lambda_values[idx], zero(FT); atol=atol, rtol=atol)
    ]

    if length(coupled_candidates) != 1
        throw(ArgumentError("Leg $(leg.name) must contain exactly one fully coupled state with λ=1; found $(length(coupled_candidates))."))
    end
    if length(decoupled_candidates) != 1
        throw(ArgumentError("Leg $(leg.name) must contain exactly one fully decoupled state with λ=0; found $(length(decoupled_candidates))."))
    end

    return ResolvedLegStateSchedule{FT}(
        lambda_values,
        only(coupled_candidates),
        only(decoupled_candidates),
    )
end

"""
    validate_cycle_config(cycle_cfg)

Validate a thermodynamic cycle and return it unchanged when it is well-formed.
"""
function validate_cycle_config(
    cycle_cfg::ThermodynamicCycleConfig;
    default_lambda_schedule=nothing,
    FT::DataType=Float32,
)
    if isempty(cycle_cfg.legs)
        throw(ArgumentError("Thermodynamic cycle must define at least one leg."))
    end

    seen = Set{Symbol}()
    for leg in cycle_cfg.legs
        if leg.name in seen
            throw(ArgumentError("Thermodynamic cycle contains duplicate leg name: $(leg.name)."))
        end
        push!(seen, leg.name)
        _validate_leg_ensemble(leg)
        if !isnothing(default_lambda_schedule)
            resolve_leg_state_schedule(leg, default_lambda_schedule, FT)
        end
    end

    return cycle_cfg
end

"""
    resolve_parameter_reference_leg(sim_cfg, cycle_cfg)

Choose the leg whose atom types define the reference parameter ordering used
throughout the optimization.
"""
function resolve_parameter_reference_leg(sim_cfg::SimulationConfig, cycle_cfg::ThermodynamicCycleConfig)
    validate_cycle_config(cycle_cfg)
    if isnothing(sim_cfg.parameter_reference_leg)
        return first(cycle_cfg.legs)
    end

    for leg in cycle_cfg.legs
        if leg.name == sim_cfg.parameter_reference_leg
            return leg
        end
    end

    throw(ArgumentError("parameter_reference_leg=$(sim_cfg.parameter_reference_leg) was not found in configured cycle legs."))
end

"""
    resolve_force_field_paths(sim_cfg)

Resolve the force-field XML paths referenced by `sim_cfg.force_field`.
"""
function resolve_force_field_paths(sim_cfg::SimulationConfig)
    ff_cfg = sim_cfg.force_field
    default_ff_dir = joinpath(dirname(pathof(Molly)), "..", "data", "force_fields")
    ff_dir = isnothing(ff_cfg.force_field_dir) ? default_ff_dir : ff_cfg.force_field_dir

    paths = String[]
    for xml_file in ff_cfg.xml_files
        push!(paths, resolve_force_field_path(xml_file, ff_dir))
    end

    return paths
end

"""
    validate_lambda_schedule(lambda_schedule)

Ensure the λ schedule is finite, monotone, and spans at least two windows.
"""
function validate_lambda_schedule(lambda_schedule)
    vals = collect(lambda_schedule)
    if length(vals) < 2
        throw(ArgumentError("lambda_schedule must contain at least two windows."))
    end

    vals_f64 = Float64.(vals)
    if any(v -> !isfinite(v), vals_f64)
        throw(ArgumentError("lambda_schedule contains non-finite values."))
    end

    if any(v -> v < 0.0 || v > 1.0, vals_f64)
        throw(ArgumentError("lambda_schedule values must lie in [0, 1]."))
    end

    if !(issorted(vals_f64) || issorted(vals_f64; rev=true))
        throw(ArgumentError("lambda_schedule must be monotonic (ascending or descending)."))
    end

    return nothing
end

"""
    apply_simulation_config!(cfg)

Promote `cfg` into the module-level globals used by the rest of the package and
rebuild the force field with the requested precision/backend.
"""
function apply_simulation_config!(cfg::SimulationConfig)
    validate_lambda_schedule(cfg.lambda_schedule)
    discard_fraction = Float64(cfg.awh_probe_discard_fraction)
    if !(0.0 <= discard_fraction < 1.0)
        throw(ArgumentError("awh_probe_discard_fraction must lie in [0, 1)."))
    end

    global FT = cfg.FT
    global AT = cfg.AT
    global Δt = cfg.Δt
    global T0 = cfg.T0
    global P0 = cfg.P0
    global lambda_schedule = cfg.lambda_schedule
    global num_lambda_states = length(cfg.lambda_schedule)
    global target_rho = FT(1.0 / num_lambda_states)

    # Recreate the force field after precision changes so all parameters live in
    # the same numeric type as the simulation state.
    ff_paths = resolve_force_field_paths(cfg)
    global ff = MolecularForceField(FT, ff_paths...; units=true)

    if cfg.AT isa DataType && cfg.AT <: CuArray
        device!(cfg.device_id)
    end

    return nothing
end

"""
    simulation_config_with(cfg; kwargs...)

Create a copy of `cfg` with selected fields overridden.
"""
function simulation_config_with(cfg::SimulationConfig; kwargs...)
    fields = [name => getfield(cfg, name) for name in fieldnames(SimulationConfig)]
    for (k, v) in kwargs
        idx = findfirst(p -> first(p) == k, fields)
        if isnothing(idx)
            throw(ArgumentError("Unknown SimulationConfig field override: $(k)."))
        end
        fields[idx] = k => v
    end
    return SimulationConfig(; fields...)
end

"""
    optimization_config_with(cfg; kwargs...)

Create a copy of `cfg` with selected fields overridden.
"""
function optimization_config_with(cfg::OptimizationConfig; kwargs...)
    fields = [name => getfield(cfg, name) for name in fieldnames(OptimizationConfig)]
    for (k, v) in kwargs
        idx = findfirst(p -> first(p) == k, fields)
        if isnothing(idx)
            throw(ArgumentError("Unknown OptimizationConfig field override: $(k)."))
        end
        fields[idx] = k => v
    end
    return OptimizationConfig(; fields...)
end
