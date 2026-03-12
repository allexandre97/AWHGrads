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
    awh_simulation_control_kwargs(awh_control)

Translate high-level AWH controls into the keyword arguments expected by
`AWHSimulation`.
"""
function awh_simulation_control_kwargs(awh_control::AWHControlConfig)
    return (
        update_freq=awh_control.update_freq,
        well_tempered_factor=awh_control.well_tempered_factor,
        coverage_threshold=awh_control.coverage_threshold,
        significant_weight=awh_control.significant_weight,
        coverage_type=awh_control.coverage_type,
    )
end

"""
    setup_alchemical_awh(pdb_file, solute_indices; kwargs...)

Build a λ-expanded AWH simulation for one thermodynamic leg. The function
optionally injects optimized parameters, restores a warm-start restart state,
and reuses a previously learned bias estimate.
"""
function setup_alchemical_awh(
    pdb_file,
    solute_indices;
    lambda_values=lambda_schedule,
    awh_control=AWHControlConfig(),
    is_vacuum=false,
    logger=nothing,
    injected_bias=nothing,
    optimized_params=nothing,
    param_idxs=nothing,
    restart_state=nothing,
    restart_active_idx::Int=1,
    warm_start::Bool=false,
    sigma_seed=nothing,
    epsilon_seed=nothing,
)
    # Start from the reference PDB and optionally attach an AWH logger.
    sys_raw = System(
        pdb_file, ff; array_type=AT, nonbonded_method=:none, 
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
    
    sys_base = System(sys_raw; atoms=Molly.to_device([seeded_atoms...], AT))

    thermostat = VelocityRescaleThermostat(T0, FT(0.1)u"ps")  
    
    if is_vacuum
        integrator = VelocityVerlet(Δt, (thermostat,), 100)  
    else
        barostat = CRescaleBarostat(P0, FT(1)u"ps"; n_steps=250)  
        integrator = VelocityVerlet(Δt, (thermostat, barostat), 100)  
    end

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
    idx_coul = findfirst(x -> x isa Coulomb, p_inters)  
    
    lj_0 = p_inters[idx_lj]  
    cl_0 = p_inters[idx_coul]  

    thermo_states = ThermoState[]  

    lj_sc = LennardJonesSoftCoreBeutler(
        cutoff = lj_0.cutoff, α = FT(awh_control.lj_softcore_alpha),
        use_neighbors = lj_0.use_neighbors, scheduler = Molly.DefaultLambdaScheduler()
    )

    coul_sc = CoulombSoftCoreBeutler(
        cutoff = cl_0.cutoff, α = FT(awh_control.coul_softcore_alpha), coulomb_const = cl_0.coulomb_const,
        use_neighbors = cl_0.use_neighbors, scheduler = Molly.DefaultLambdaScheduler()
    )

    # Construct one thermodynamic state per λ window by only changing the
    # alchemical role of the solute atoms.
    for λ in lambda_values
        acopy = Atom[]  
        for (i, a) in enumerate(seeded_atoms)
            if a.index ∈ solute_indices 
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(λ), Molly.InsertRole))  
            else
                push!(acopy, Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, a.λ, a.alch_role))  
            end
        end

        sys_w = System(
            deepcopy(sys_base);
            atoms = Molly.to_device([acopy...], AT),  
            pairwise_inters = (coul_sc, lj_sc),
            loggers = isnothing(logger) ? NamedTuple() : (awh_logger=logger,)
        )
        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))  
    end

    first_state = (warm_start && !isnothing(restart_state)) ? clamp(restart_active_idx, 1, length(lambda_values)) : 1
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
        num_md_steps=awh_control.seed_num_md_steps,
        log_freq=awh_control.seed_log_freq,
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
