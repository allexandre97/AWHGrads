"""
    awh_state_control_kwargs(awh_control; first_state)

Translate high-level AWH controls into the keyword arguments expected by
`AWHState`.
"""
function awh_state_control_kwargs(awh_control::AWHControlConfig; first_state::Int)
    return (
        reuse_neighbors=awh_control.reuse_neighbors,
        first_state=first_state,
        n_bias=awh_control.initial_n_bias,
    )
end

"""
    resolve_awh_iteration_cadence(awh_control)

Resolve the effective Molly iteration counts implied by the user-facing AWH
cadence settings.
"""
function resolve_awh_iteration_cadence(awh_control::AWHControlConfig)
    n_md_steps = awh_control.seed_num_md_steps
    n_md_steps > 0 || throw(ArgumentError("`seed_num_md_steps` must be positive, got $n_md_steps."))

    target_bias_md_steps = awh_control.bias_update_interval_md_steps
    target_bias_md_steps > 0 || throw(ArgumentError("`bias_update_interval_md_steps` must be positive, got $target_bias_md_steps."))

    stats_log_every_updates = awh_control.stats_log_every_updates
    stats_log_every_updates > 0 || throw(ArgumentError("`stats_log_every_updates` must be positive, got $stats_log_every_updates."))

    resolved_update_freq = if isnothing(awh_control.update_freq)
        # Round down so auto mode never yields a slower-than-requested bias cadence.
        max(1, fld(target_bias_md_steps, n_md_steps))
    else
        awh_control.update_freq > 0 || throw(ArgumentError("`update_freq` override must be positive, got $(awh_control.update_freq)."))
        awh_control.update_freq
    end

    resolved_log_freq = resolved_update_freq * stats_log_every_updates

    return (
        n_md_steps=n_md_steps,
        update_freq=resolved_update_freq,
        log_freq=resolved_log_freq,
        effective_bias_update_md_steps=n_md_steps * resolved_update_freq,
        stats_log_every_updates=stats_log_every_updates,
        auto_update_freq=isnothing(awh_control.update_freq),
    )
end

"""
    resolve_leg_awh_control(awh_control, leg)

Copy the global AWH controls and apply any leg-specific cadence overrides.
"""
function resolve_leg_awh_control(
    awh_control::AWHControlConfig,
    leg::ThermodynamicLegConfig,
)
    return AWHControlConfig(
        lj_softcore_alpha=awh_control.lj_softcore_alpha,
        coul_softcore_alpha=awh_control.coul_softcore_alpha,
        reuse_neighbors=awh_control.reuse_neighbors,
        seed_num_md_steps=isnothing(leg.awh_seed_num_md_steps) ? awh_control.seed_num_md_steps : leg.awh_seed_num_md_steps,
        bias_update_interval_md_steps=isnothing(leg.awh_bias_update_interval_md_steps) ? awh_control.bias_update_interval_md_steps : leg.awh_bias_update_interval_md_steps,
        stats_log_every_updates=awh_control.stats_log_every_updates,
        update_freq=awh_control.update_freq,
        coverage_threshold=awh_control.coverage_threshold,
        significant_weight=awh_control.significant_weight,
        initial_n_bias=isnothing(leg.awh_initial_n_bias) ? awh_control.initial_n_bias : leg.awh_initial_n_bias,
        well_tempered_factor=awh_control.well_tempered_factor,
        coverage_type=awh_control.coverage_type,
    )
end

"""
    awh_simulation_control_kwargs(awh_control)

Translate high-level AWH controls into the keyword arguments expected by
`AWHSimulation`.
"""
function awh_simulation_control_kwargs(awh_control::AWHControlConfig)
    cadence = resolve_awh_iteration_cadence(awh_control)
    return (
        num_md_steps=cadence.n_md_steps,
        update_freq=cadence.update_freq,
        log_freq=cadence.log_freq,
        well_tempered_factor=awh_control.well_tempered_factor,
        coverage_threshold=awh_control.coverage_threshold,
        significant_weight=awh_control.significant_weight,
        coverage_type=awh_control.coverage_type,
    )
end

normalize_lambda_scheduler_name(value::Union{Nothing, Symbol}) = isnothing(value) ? :default : value
normalize_softcore_model_name(value::Union{Nothing, Symbol}) = isnothing(value) ? :gapsys : value
normalize_reaction_field_coulomb_model_name(value::Union{Nothing, Symbol}) = isnothing(value) ? :gapsys_rf : value
uses_reaction_field_coulomb_model(value::Union{Nothing, Symbol}) = normalize_reaction_field_coulomb_model_name(value) in (:beutler_rf, :gapsys_rf)

function normalize_electrostatics_method_name(
    value::Union{Nothing, Symbol},
    is_vacuum::Bool=false,
    coulomb_softcore_model::Union{Nothing, Symbol}=nothing,
)
    if !isnothing(value)
        return value
    end
    if is_vacuum
        return :none
    end
    return uses_reaction_field_coulomb_model(coulomb_softcore_model) ? :cutoff : :none
end

function normalize_reaction_field_coulomb_model_name(value::Union{Nothing, Symbol}, ::Val{:cutoff})
    model_name = normalize_reaction_field_coulomb_model_name(value)
    if model_name == :beutler
        return :beutler_rf
    elseif model_name == :gapsys
        return :gapsys_rf
    end
    return model_name
end

function normalize_ewald_coulomb_model_name(value::Union{Nothing, Symbol})
    model_name = normalize_softcore_model_name(value)
    if model_name == :beutler_rf
        return :beutler
    elseif model_name == :gapsys_rf
        return :gapsys
    end
    return model_name
end

function resolve_lambda_scheduler(value::Union{Nothing, Symbol})
    scheduler_name = normalize_lambda_scheduler_name(value)
    if scheduler_name == :default
        return Molly.DefaultLambdaScheduler()
    elseif scheduler_name == :namd
        return Molly.NAMDLambdaScheduler()
    elseif scheduler_name == :quarters
        return Molly.QuartersLambdaScheduler()
    elseif scheduler_name == :ele_scaled
        return Molly.EleScaledLambdaScheduler()
    end
    throw(ArgumentError("Unsupported lambda scheduler `$scheduler_name`."))
end

function build_lj_softcore_interaction(
    lj_0::LennardJones,
    awh_control::AWHControlConfig,
    scheduler,
    model::Union{Nothing, Symbol},
    is_vacuum::Bool=false,
)
    model_name = normalize_softcore_model_name(model)
    # Vacuum systems on CPU need a finite cutoff for the neighbor finder to work correctly with exclusions.
    # For others, we MUST respect the reference interaction's cutoff to match AWH's neighbor-list truncation.
    cutoff = is_vacuum ? DistanceCutoff(5.0u"nm") : lj_0.cutoff
    if model_name == :beutler
        return LennardJonesSoftCoreBeutler(
            cutoff=cutoff,
            α=FT(awh_control.lj_softcore_alpha),
            use_neighbors=lj_0.use_neighbors,
            shortcut=lj_0.shortcut,
            σ_mixing=lj_0.σ_mixing,
            ϵ_mixing=lj_0.ϵ_mixing,
            scheduler=scheduler,
            weight_special=lj_0.weight_special,
        )
    elseif model_name == :gapsys
        return LennardJonesSoftCoreGapsys(
            cutoff=cutoff,
            α=FT(awh_control.lj_softcore_alpha),
            use_neighbors=lj_0.use_neighbors,
            shortcut=lj_0.shortcut,
            σ_mixing=lj_0.σ_mixing,
            ϵ_mixing=lj_0.ϵ_mixing,
            scheduler=scheduler,
            weight_special=lj_0.weight_special,
        )
    end
    throw(ArgumentError("Unsupported LJ soft-core model `$model_name`."))
end

function build_coulomb_softcore_interaction(
    cl_0::Coulomb,
    awh_control::AWHControlConfig,
    scheduler,
    model::Union{Nothing, Symbol},
    is_vacuum::Bool=false,
)
    model_name = normalize_softcore_model_name(model)
    if model_name == :beutler_rf
        model_name = :beutler
    elseif model_name == :gapsys_rf
        model_name = :gapsys
    end
    cutoff = is_vacuum ? DistanceCutoff(5.0u"nm") : cl_0.cutoff
    if model_name == :beutler
        return CoulombSoftCoreBeutler(
            cutoff=cutoff,
            α=FT(awh_control.coul_softcore_alpha),
            use_neighbors=cl_0.use_neighbors,
            scheduler=scheduler,
            weight_special=cl_0.weight_special,
            coulomb_const=cl_0.coulomb_const,
        )
    elseif model_name == :gapsys
        return CoulombSoftCoreGapsys(
            cutoff=cutoff,
            α=FT(awh_control.coul_softcore_alpha),
            σQ=FT(1.0)u"nm",
            use_neighbors=cl_0.use_neighbors,
            scheduler=scheduler,
            weight_special=cl_0.weight_special,
            coulomb_const=cl_0.coulomb_const,
        )
    end
    throw(ArgumentError("Unsupported Coulomb soft-core model `$model_name`."))
end

function build_coulomb_softcore_interaction(
    cl_0::CoulombReactionField,
    awh_control::AWHControlConfig,
    scheduler,
    model::Union{Nothing, Symbol},
    is_vacuum::Bool=false,
)
    model_name = normalize_reaction_field_coulomb_model_name(model, Val(:cutoff))
    cutoff = is_vacuum ? DistanceCutoff(5.0u"nm") : cl_0.dist_cutoff
    if model_name == :beutler_rf
        return CoulombSoftCoreBeutlerReactionField(
            dist_cutoff=cutoff isa NoCutoff ? 100.0u"nm" : cutoff, # RFC needs a distance, but we won't use it if neighbors are off
            solvent_dielectric=cl_0.solvent_dielectric,
            α=FT(awh_control.coul_softcore_alpha),
            use_neighbors=cl_0.use_neighbors,
            σ_mixing=Molly.LorentzMixing(),
            ϵ_mixing=Molly.GeometricMixing(),
            scheduler=scheduler,
            weight_special=cl_0.weight_special,
            coulomb_const=cl_0.coulomb_const,
        )
    elseif model_name == :gapsys_rf
        return CoulombSoftCoreGapsysReactionField(
            dist_cutoff=cutoff isa NoCutoff ? 100.0u"nm" : cutoff,
            solvent_dielectric=cl_0.solvent_dielectric,
            α=FT(awh_control.coul_softcore_alpha),
            σQ=FT(1.0)u"nm",
            use_neighbors=cl_0.use_neighbors,
            scheduler=scheduler,
            weight_special=cl_0.weight_special,
            coulomb_const=cl_0.coulomb_const,
        )
    end
    throw(ArgumentError("Reaction-field Coulomb base requires an RF soft-core model, got `$model_name`."))
end

function build_coulomb_softcore_interaction(
    cl_0::CoulombEwald,
    awh_control::AWHControlConfig,
    scheduler,
    model::Union{Nothing, Symbol},
    is_vacuum::Bool=false,
)
    model_name = normalize_ewald_coulomb_model_name(model)
    cutoff = is_vacuum ? DistanceCutoff(5.0u"nm") : cl_0.dist_cutoff
    if model_name == :beutler
        return CoulombSoftCoreBeutlerEwald(
            dist_cutoff=cutoff isa NoCutoff ? 100.0u"nm" : cutoff,
            error_tol=cl_0.error_tol,
            α=FT(awh_control.coul_softcore_alpha),
            use_neighbors=cl_0.use_neighbors,
            σ_mixing=Molly.LorentzMixing(),
            ϵ_mixing=Molly.GeometricMixing(),
            scheduler=scheduler,
            weight_special=cl_0.weight_special,
            coulomb_const=cl_0.coulomb_const,
            approximate_erfc=cl_0.approximate_erfc,
        )
    elseif model_name == :gapsys
        return CoulombSoftCoreGapsysEwald(
            dist_cutoff=cutoff isa NoCutoff ? 100.0u"nm" : cutoff,
            error_tol=cl_0.error_tol,
            α=FT(awh_control.coul_softcore_alpha),
            σQ=FT(1.0)u"nm",
            use_neighbors=cl_0.use_neighbors,
            scheduler=scheduler,
            weight_special=cl_0.weight_special,
            coulomb_const=cl_0.coulomb_const,
            approximate_erfc=cl_0.approximate_erfc,
        )
    end
    throw(ArgumentError("Ewald/PME Coulomb base requires a plain soft-core model, got `$model_name`."))
end

"""
    awh_coupling_methods(is_vacuum, ensemble)

Build the thermostat/barostat tuple for one leg based on its physical ensemble.
"""
function awh_coupling_methods(is_vacuum::Bool, ensemble::Symbol)
    thermostat = VelocityRescaleThermostat(T0, FT(0.1)u"ps")
    if is_vacuum || ensemble == :nvt
        return (thermostat,)
    elseif ensemble == :npt
        barostat = CRescaleBarostat(P0, FT(1)u"ps"; n_steps=250)
        return (thermostat, barostat)
    end
    throw(ArgumentError("Unsupported ensemble=$ensemble. Supported values are :npt and :nvt."))
end

function resolve_base_nonbonded_method(
    electrostatics_method::Union{Nothing, Symbol},
    is_vacuum::Bool,
    coulomb_softcore_model::Union{Nothing, Symbol},
)
    method_name = normalize_electrostatics_method_name(
        electrostatics_method,
        is_vacuum,
        coulomb_softcore_model,
    )
    if method_name in (:none, :cutoff, :ewald, :pme)
        return method_name
    end
    throw(ArgumentError("Unsupported electrostatics method `$method_name`."))
end

rebuild_general_interaction(inter, scheduler) = inter

function rebuild_general_interaction(inter::Molly.Ewald, scheduler)
    return Molly.Ewald(
        inter.dist_cutoff,
        inter.error_tol,
        deepcopy(inter.excluded_pairs),
        scheduler,
    )
end

function rebuild_general_interaction(inter::Molly.PME, scheduler)
    return Molly.PME(
        inter.dist_cutoff,
        inter.error_tol,
        inter.order,
        inter.ϵr,
        deepcopy(inter.excluded_pairs),
        inter.α,
        inter.mesh_dims,
        inter.grid_indices,
        inter.grid_fractions,
        inter.bsplines_θ,
        inter.bsplines_dθ,
        inter.bsplines_moduli_x,
        inter.bsplines_moduli_y,
        inter.bsplines_moduli_z,
        inter.charge_grid,
        inter.charge_grid_buffer,
        inter.excluded_buffer_Fs,
        inter.excluded_buffer_Es,
        inter.recip_conv_buffer,
        inter.virial_buffer,
        nothing,
        nothing,
        inter.fft_plan,
        inter.bfft_plan,
        scheduler,
        inter.grad_safe,
    )
end

function rebuild_state_general_interactions(general_inters::Tuple, scheduler)
    return Tuple(rebuild_general_interaction(inter, scheduler) for inter in general_inters)
end

"""
    awh_global_lambda_schedule(lambda_values, FT=Float32)

Resolve and validate the global atom λ ladder used by Molly's default
alchemical scheduler.
"""
function awh_global_lambda_schedule(lambda_values, ::Type{FT}=Float32) where {FT <: AbstractFloat}
    global_values = FT.(collect(lambda_values))
    validate_lambda_schedule(global_values)
    return global_values
end

"""
    solvent_lambda_schedule_diagnostics(lambda_values, lambda_scheduler=:default, FT=Float32)

Return a compact per-state summary of the solvent-leg λ schedule under Molly's
insert-role scheduler so logs can be interpreted in terms of effective
electrostatic and LJ scaling, not only state indices.
"""
function solvent_lambda_schedule_diagnostics(
    lambda_values,
    lambda_scheduler::Union{Nothing, Symbol}=:default,
    ::Type{FT}=Float32,
) where {FT <: AbstractFloat}
    scheduler = resolve_lambda_scheduler(lambda_scheduler)
    global_values = awh_global_lambda_schedule(lambda_values, FT)

    return [
        (
            idx=idx,
            global_lambda=λ,
            elec_lambda=FT(Molly.scale_elec(scheduler, λ, Molly.InsertRole)),
            lj_lambda=FT(Molly.scale_sterics(scheduler, λ, Molly.InsertRole)),
            stage=(λ >= FT(0.5) ? :charge : :lj),
        )
        for (idx, λ) in enumerate(global_values)
    ]
end

"""
    setup_alchemical_awh(pdb_file, solute_indices; kwargs...)

Build a λ-expanded AWH simulation for one thermodynamic leg. The function
optionally injects optimized parameters, restores a warm-start restart state,
reuses a previously learned bias estimate, and lets Molly's default lambda
scheduler map the global atom λ values onto the soft-core Coulomb/LJ scaling.
"""
function setup_alchemical_awh(
    pdb_file,
    solute_indices;
    lambda_values=lambda_schedule,
    awh_control=AWHControlConfig(),
    is_vacuum=false,
    ensemble::Symbol=:npt,
    logger=nothing,
    injected_bias=nothing,
    optimized_params=nothing,
    param_idxs=nothing,
    restart_state=nothing,
    restart_active_idx::Int=1,
    warm_start::Bool=false,
    electrostatics_method::Union{Nothing, Symbol}=nothing,
    lambda_scheduler::Union{Nothing, Symbol}=nothing,
    coulomb_softcore_model::Union{Nothing, Symbol}=nothing,
    lj_softcore_model::Union{Nothing, Symbol}=nothing,
    sigma_seed=nothing,
    epsilon_seed=nothing,
    array_type::Type{<:AbstractArray} = AT,
    nonbonded_energy_type=nonbonded_energy_type,
)
    # Start from the reference PDB and optionally attach an AWH logger.
    base_nonbonded_method = resolve_base_nonbonded_method(
        electrostatics_method,
        is_vacuum,
        coulomb_softcore_model,
    )
    nf_type = nothing
    sys_raw = System(
        pdb_file, ff; array_type=array_type, nonbonded_method=base_nonbonded_method,
        neighbor_finder_type=nf_type,
        nonbonded_energy_type=nonbonded_energy_type,
        loggers=isnothing(logger) ? NamedTuple() : (awh_logger=logger,)  
    )

    atoms_raw = Molly.from_device(sys_raw.atoms)
    seeded_atoms = Atom[]
    
    # Safely sized seeds strictly above the optimization boundaries to prevent 1/r^12 singularity
    sigma_seed_local = isnothing(sigma_seed) ? FT(0.15)u"nm" : sigma_seed
    epsilon_seed_local = isnothing(epsilon_seed) ? FT(1e-4)u"kJ/mol" : epsilon_seed
    
    for (i, a) in enumerate(atoms_raw)
        new_sigma = (ustrip(a.σ) <= FT(1e-6) || ustrip(a.σ) == one(FT)) ? sigma_seed_local : a.σ
        new_eps   = ustrip(a.ϵ) <= FT(1e-6) ? epsilon_seed_local : a.ϵ
        
        # Optimized parameters are stored by atom type; the index maps project
        # them back onto per-atom σ/ϵ values for Molly.
        if !isnothing(optimized_params) && !isnothing(param_idxs)
            idx_σ_map = param_idxs[1][2]
            idx_ϵ_map = param_idxs[1][3]

            if idx_σ_map[i] > 0
                new_sigma = FT(optimized_params[idx_σ_map[i]]) * u"nm"
            end
            if idx_ϵ_map[i] > 0
                new_eps = FT(optimized_params[idx_ϵ_map[i]]) * u"kJ/mol"
            end
        end
        
        push!(seeded_atoms, Atom(a.index, a.atom_type, a.mass, a.charge, new_sigma, new_eps, a.λ, a.alch_role))
    end
    
    sys_base = System(sys_raw; atoms=Molly.to_device([seeded_atoms...], array_type))

    coupling_methods = awh_coupling_methods(is_vacuum, ensemble)
    integrator = VelocityVerlet(Δt, coupling_methods, 100)

    atom_lambda_schedule = awh_global_lambda_schedule(lambda_values, FT)
    solute_index_set = Set(solute_indices)

    if warm_start && !isnothing(restart_state)
        sys_base = System(
            sys_base;
            coords = copy(restart_state.coords),
            boundary = deepcopy(restart_state.boundary),
            velocities = copy(restart_state.velocities)
        )
    else
        minim = SteepestDescentMinimizer(step_size=FT(0.01)u"nm", max_steps=1000)
        simulate!(sys_base, minim)
        random_velocities!(sys_base, T0)
    end

    p_inters = sys_base.pairwise_inters 
    idx_lj   = findfirst(x -> x isa LennardJones, p_inters)  
    idx_coul = findfirst(x -> x isa Coulomb || x isa CoulombReactionField || x isa CoulombEwald, p_inters)
    
    lj_0 = p_inters[idx_lj]  
    cl_0 = p_inters[idx_coul]  
    scheduler = resolve_lambda_scheduler(lambda_scheduler)
    lj_sc = build_lj_softcore_interaction(lj_0, awh_control, scheduler, lj_softcore_model, is_vacuum)
    coul_sc = build_coulomb_softcore_interaction(cl_0, awh_control, scheduler, coulomb_softcore_model, is_vacuum)

    thermo_states = ThermoState[]  

    # Construct one thermodynamic state per λ window by only changing the
    # solute atoms' global λ values and alchemical role.
    for state_idx in eachindex(atom_lambda_schedule)
        λ_atom = atom_lambda_schedule[state_idx]
        acopy = Atom[]  
        for (i, a) in enumerate(seeded_atoms)
            if a.index in solute_index_set
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(λ_atom), Molly.InsertRole))
            else
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, a.λ, a.alch_role))  
            end
        end

        sys_w = System(
            deepcopy(sys_base);
            atoms = Molly.to_device([acopy...], array_type),  
            pairwise_inters = (coul_sc, lj_sc),
            general_inters = rebuild_state_general_interactions(sys_base.general_inters, scheduler),
            loggers = isnothing(logger) ? NamedTuple() : (awh_logger=logger,)
        )
        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))  
    end

    first_state = (warm_start && !isnothing(restart_state)) ? clamp(restart_active_idx, 1, length(atom_lambda_schedule)) : 1
    awh_state = AWHState(thermo_states; awh_state_control_kwargs(awh_control; first_state=first_state)...)
    
    if !isnothing(injected_bias)  
        # Warm-start bias reuse intentionally forces Molly back into its initial
        # linear stage so the bias can relax to the new parameterization.
        awh_state.f .= injected_bias.f  
        awh_state.rho .= injected_bias.rho  
        awh_state.log_rho .= injected_bias.log_rho  
        awh_state.in_initial_stage = true
        awh_state.N_bias = FT(awh_control.initial_n_bias)
        awh_state.N_eff = zero(FT)
        empty!(awh_state.visited_windows)
    end

    awh_sim = AWHSimulation(
        awh_state;
        awh_simulation_control_kwargs(awh_control)...,
    )
    
    return awh_sim, sys_base  
end

"""
    capture_restart_state(awh_sim)

Extract the minimal simulation state needed to warm-start the next macro epoch.
"""
function capture_restart_state(awh_sim::AWHSimulation)
    sys = awh_sim.state.active_sys
    return (
        coords = copy(sys.coords),
        boundary = deepcopy(sys.boundary),
        velocities = copy(sys.velocities),
        active_idx = awh_sim.state.active_idx
    )
end

"""
    coords_rmsd_nm(coords_a, coords_b)

Compute the RMSD between two coordinate arrays in nanometers.
"""
function coords_rmsd_nm(coords_a, coords_b)
    coords_a_cpu = Array(coords_a)
    coords_b_cpu = Array(coords_b)
    if length(coords_a_cpu) != length(coords_b_cpu)
        return FT(NaN)
    end
    n_atoms = length(coords_a_cpu)
    if n_atoms == 0
        return zero(FT)
    end
    sq_dist_sum = zero(FT)
    for i in 1:n_atoms
        δ = ustrip.(coords_a_cpu[i] .- coords_b_cpu[i])
        sq_dist_sum += FT(sum(abs2, δ))
    end
    return sqrt(sq_dist_sum / FT(n_atoms))
end
