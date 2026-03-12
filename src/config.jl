"""
    default_cycle_config(; target_dG_kcal_mol=-5.01)

Return the built-in two-leg ethanol hydration cycle used by the example
scripts.
"""
function default_cycle_config(; target_dG_kcal_mol::Real=-5.01)
    legs = [
        ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            coefficient=1.0,
            is_vacuum=false,
            include_pv=true,
            probe_time=Float32(1.5)u"ns",
        ),
        ThermodynamicLegConfig(
            name=:vacuum,
            pdb="ethanol_vac.pdb",
            coefficient=-1.0,
            is_vacuum=true,
            include_pv=false,
            probe_time=Float32(0.25)u"ns",
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
        huber_delta=FT(2.0),
        kl_target=FT(0.1),
        eigenvalue_tol_scale=FT(1e-2),
        min_phi_step=FT(5e-4),
        max_phi_step_solute=FT(0.35),
        max_phi_step_solvent=FT(0.035),
        tiny_alpha_cutoff=FT(0.015625),
        max_tiny_alpha_hits=2,
        restart_rmsd_tol_nm=FT(1e-5),
        optimize_solvent=false,
        ess_threshold_scale=FT(0.22),
        awh_min_linear_neff=3000,
        awh_min_lambda_ess=300,
        awh_split_tol_kT=FT(0.5),
        awh_parity_tol_kT=FT(0.25),
        awh_tail_lag=10,
        awh_min_round_trips=3,
        awh_endpoint_min_fraction=FT(0.03),
        awh_stageA_stable_blocks=2,
        awh_stageB_cooldown_blocks=2,
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

    return ThermodynamicCycleConfig(
        legs=[
            ThermodynamicLegConfig(
                name=:solvent,
                pdb=sim_cfg.pdb_solv,
                coefficient=1.0,
                is_vacuum=false,
                include_pv=true,
                probe_time=sim_cfg.awh_probe_time_solv,
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

"""
    validate_cycle_config(cycle_cfg)

Validate a thermodynamic cycle and return it unchanged when it is well-formed.
"""
function validate_cycle_config(cycle_cfg::ThermodynamicCycleConfig)
    if isempty(cycle_cfg.legs)
        throw(ArgumentError("Thermodynamic cycle must define at least one leg."))
    end

    seen = Set{Symbol}()
    for leg in cycle_cfg.legs
        if leg.name in seen
            throw(ArgumentError("Thermodynamic cycle contains duplicate leg name: $(leg.name)."))
        end
        push!(seen, leg.name)
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
        if isabspath(xml_file)
            push!(paths, xml_file)
        else
            push!(paths, joinpath(ff_dir, xml_file))
        end
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
