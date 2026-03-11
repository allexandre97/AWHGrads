function default_cycle_config(; target_dG_kcal_mol::Real=-5.01)
    legs = [
        ThermodynamicLegConfig(
            name=:solvent,
            pdb="ethanol_solv.pdb",
            coefficient=1.0,
            is_vacuum=false,
            include_pv=true,
            probe_time=Float32(0.75)u"ns",
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
        awh_probe_time_solv=FT(0.75)u"ns",
        awh_probe_time_vac=FT(0.25)u"ns",
        dG_exp_kcal_mol=-5.01,
        force_field=ForceFieldConfig(),
        cycle=nothing,
        parameter_reference_leg=nothing,
        parameter_bounds=ParameterBoundsConfig(),
    )
end

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
        awh_split_tol_kT=FT(0.5),
        awh_parity_tol_kT=FT(0.1),
        awh_tail_lag=10,
        awh_min_round_trips=3,
        awh_endpoint_min_fraction=FT(0.03),
        awh_stageA_stable_blocks=2,
        k_sigmoid=FT(1.0),
    )
end

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

function apply_simulation_config!(cfg::SimulationConfig)
    global FT = cfg.FT
    global AT = cfg.AT
    global Δt = cfg.Δt
    global T0 = cfg.T0
    global P0 = cfg.P0
    global lambda_schedule = cfg.lambda_schedule
    global num_lambda_states = length(cfg.lambda_schedule)
    global target_rho = FT(1.0 / num_lambda_states)

    ff_paths = resolve_force_field_paths(cfg)
    global ff = MolecularForceField(FT, ff_paths...; units=true)

    if cfg.AT isa DataType && cfg.AT <: CuArray
        device!(cfg.device_id)
    end

    return nothing
end


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
