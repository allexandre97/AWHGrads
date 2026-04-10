"""
    sync_runtime_aliases!(runtime)

Populate the legacy `*_solv`/`*_vac` aliases from the general per-leg runtime
maps. This keeps older calling code working while the package uses arbitrary leg
names internally.
"""
function sync_runtime_aliases!(runtime::RuntimeState)
    runtime.active_bias_solv = get(runtime.active_bias, :solvent, nothing)
    runtime.active_bias_vac = get(runtime.active_bias, :vacuum, nothing)
    runtime.restart_cache_solv = get(runtime.restart_cache, :solvent, nothing)
    runtime.restart_cache_vac = get(runtime.restart_cache, :vacuum, nothing)
    return runtime
end

"""
    macro_epoch_awh_start_state(runtime, leg_name, macro_epoch)

Resolve the carry-over AWH state for one leg at the start of a macro epoch.
AWH bias is only reused when there is a matching restart snapshot; a cold start
from the PDB always starts from a fresh AWH state.
"""
function macro_epoch_awh_start_state(runtime::RuntimeState, leg_name::Symbol, macro_epoch::Int)
    if macro_epoch <= 1
        return (warm_start=false, restart_state=nothing, injected_bias=nothing)
    end

    restart_state = get(runtime.restart_cache, leg_name, nothing)
    warm_start = !isnothing(restart_state)
    injected_bias = warm_start ? get(runtime.active_bias, leg_name, nothing) : nothing
    return (warm_start=warm_start, restart_state=restart_state, injected_bias=injected_bias)
end

function stage_b_retry_controls(
    base_stageA_stable_blocks::Int,
    base_stageB_cooldown_blocks::Int,
    failures::Int,
    stageA_streak_growth_factor::FT,
    stageB_cooldown_growth_factor::FT,
    stageA_max_streak::Int,
    stageB_max_cooldown::Int,
) where {FT <: AbstractFloat}
    failure_count = max(0, failures)
    base_streak = max(1, base_stageA_stable_blocks)
    base_cooldown = max(0, base_stageB_cooldown_blocks)
    streak_scale = max(one(FT), stageA_streak_growth_factor) ^ failure_count
    cooldown_scale = max(one(FT), stageB_cooldown_growth_factor) ^ failure_count
    target_streak = clamp(
        round(Int, base_streak * streak_scale),
        base_streak,
        max(base_streak, stageA_max_streak),
    )
    cooldown_blocks = clamp(
        round(Int, base_cooldown * cooldown_scale),
        base_cooldown,
        max(base_cooldown, stageB_max_cooldown),
    )
    return (target_streak=target_streak, cooldown_blocks=cooldown_blocks)
end

function scaled_probe_frame_cap(base_max_frames::Int, base_probe_steps::Int, current_probe_steps::Int)
    if base_max_frames <= 0
        return 0
    end
    base_steps = max(1, base_probe_steps)
    current_steps = max(1, current_probe_steps)
    scale = current_steps / base_steps
    return max(base_max_frames, Int(ceil(base_max_frames * scale)))
end

function solvent_stage_a_tail_state_indices(
    leg::ThermodynamicLegConfig,
    state_schedule::ResolvedLegStateSchedule{FT};
    lj_lambda_max::FT,
) where {FT <: AbstractFloat}
    if leg.is_vacuum
        return Int[]
    end

    scheduler_name = isnothing(leg.lambda_scheduler) ? :default : leg.lambda_scheduler
    if scheduler_name ∉ (:default, :namd, :ele_scaled)
        return Int[]
    end

    diagnostics = solvent_lambda_schedule_diagnostics(state_schedule.lambda, scheduler_name, FT)
    lj_tol = sqrt(eps(FT))
    tail_idxs = Int[
        entry.idx for entry in diagnostics
        if entry.stage == :lj && entry.lj_lambda <= lj_lambda_max + lj_tol
    ]
    return isempty(tail_idxs) ? Int[state_schedule.decoupled_state_idx] : tail_idxs
end

function solvent_stage_a_endpoint_state_indices(
    leg::ThermodynamicLegConfig,
    state_schedule::ResolvedLegStateSchedule{FT};
    lj_lambda_max::FT,
) where {FT <: AbstractFloat}
    if leg.is_vacuum
        return Int[state_schedule.coupled_state_idx, state_schedule.decoupled_state_idx]
    end
    endpoint_idxs = solvent_stage_a_tail_state_indices(
        leg,
        state_schedule;
        lj_lambda_max=lj_lambda_max,
    )
    if state_schedule.decoupled_state_idx ∉ endpoint_idxs
        push!(endpoint_idxs, state_schedule.decoupled_state_idx)
        sort!(endpoint_idxs)
    end
    return endpoint_idxs
end

function stage_a_history_window_sample_count(
    md_steps_block::Int,
    awh_control::AWHControlConfig,
    history_blocks::Int,
)
    history_blocks <= 0 && return 0
    cadence = resolve_awh_iteration_cadence(awh_control)
    stats_log_md_steps = cadence.n_md_steps * cadence.log_freq
    samples_per_block = max(1, Int(cld(md_steps_block, stats_log_md_steps)))
    return max(1, history_blocks * samples_per_block)
end

function format_state_idx_span(idxs::Vector{Int})
    isempty(idxs) && return "-"
    if length(idxs) == 1
        return "λ$(only(idxs))"
    end
    if all(diff(idxs) .== 1)
        return "λ$(first(idxs)):λ$(last(idxs))"
    end
    return join(["λ$(idx)" for idx in idxs], ",")
end


time_to_steps_floor(duration) = Int(floor(uconvert(unit(Δt), duration) / Δt))
time_to_steps_round(duration) = max(1, Int(round(uconvert(unit(Δt), duration) / Δt)))

"""
    compute_standard_state_correction(cycle_cfg, FT)

Return the dimensionless standard-state correction applied to the cycle free
energy when requested.
"""
function compute_standard_state_correction(cycle_cfg::ThermodynamicCycleConfig, ::Type{FT}) where {FT <: AbstractFloat}
    if !cycle_cfg.include_standard_state_correction
        return zero(FT)
    end
    V_gas = ustrip(u"nm^3", Unitful.k * T0 / P0)
    V_std = ustrip(u"nm^3", 1.0u"L" / (1.0u"mol" * Unitful.Na))
    return FT(log(V_gas / V_std))
end


"""
    initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)

Derive the reference parameter vector, trainable subset, transform bounds, and
per-leg injection index maps used throughout the optimization pipeline.
"""
function initialize_parameter_state(
    sim_cfg::SimulationConfig,
    opt_cfg::OptimizationConfig,
    cycle_cfg::ThermodynamicCycleConfig,
)
    FT = sim_cfg.FT
    bounds_cfg = sim_cfg.parameter_bounds
    solute_idx = sim_cfg.solute_idx

    ref_leg = resolve_parameter_reference_leg(sim_cfg, cycle_cfg)
    ref_nonbonded_method = resolve_base_nonbonded_method(
        ref_leg.electrostatics_method,
        ref_leg.is_vacuum,
        ref_leg.coulomb_softcore_model,
    )
    sys_ref = System(ref_leg.pdb, ff; array_type=(ref_leg.is_vacuum ? Array : AT), nonbonded_method=ref_nonbonded_method, nonbonded_energy_type=sim_cfg.nonbonded_energy_type)
    atoms_cpu = Molly.from_device(sys_ref.atoms)

    theta_ref = Vector{FT}()
    param_names = String[]
    param_kind = Symbol[]
    param_is_hydrogen = Bool[]
    processed_atom_types = Dict{String, Int}()
    solute_param_indices = Int[]
    solvent_param_indices = Int[]

    sigma_floor = FT(bounds_cfg.sigma_floor)
    epsilon_floor = FT(bounds_cfg.epsilon_floor)
    clamp_eps = FT(bounds_cfg.reference_clamp_eps)

    # Each unique atom type contributes one σ and one ϵ parameter. The ordering
    # established here becomes the canonical parameter ordering everywhere else.
    for idx in eachindex(atoms_cpu)
        atom = atoms_cpu[idx]
        atype = String(sys_ref.atoms_data[idx].atom_type)
        if haskey(processed_atom_types, atype)
            continue
        end

        is_hydrogen = startswith(lowercase(atype), "h")

        sigma_raw = FT(ustrip(atom.σ))
        if sigma_raw <= FT(1e-6) || sigma_raw == one(FT)
            sigma_raw = sigma_floor
        end
        push!(theta_ref, sigma_raw)
        push!(param_names, "atom_$(atype)_σ")
        push!(param_kind, :sigma)
        push!(param_is_hydrogen, is_hydrogen)
        sigma_idx = length(theta_ref)

        epsilon_raw = FT(ustrip(atom.ϵ))
        if epsilon_raw <= FT(1e-6)
            epsilon_raw = epsilon_floor
        end
        push!(theta_ref, epsilon_raw)
        push!(param_names, "atom_$(atype)_ϵ")
        push!(param_kind, :epsilon)
        push!(param_is_hydrogen, is_hydrogen)
        epsilon_idx = length(theta_ref)

        processed_atom_types[atype] = sigma_idx
        if idx ∈ solute_idx
            push!(solute_param_indices, sigma_idx, epsilon_idx)
        else
            push!(solvent_param_indices, sigma_idx, epsilon_idx)
        end
    end

    trainable_param_indices = opt_cfg.optimize_solvent ? collect(eachindex(theta_ref)) : copy(solute_param_indices)
    trainable_param_names = [param_names[idx] for idx in trainable_param_indices]
    trainable_position_map = Dict{Int, Int}()
    for (i_local, i_global) in enumerate(trainable_param_indices)
        trainable_position_map[i_global] = i_local
    end

    theta_min = zeros(FT, length(theta_ref))
    theta_max = zeros(FT, length(theta_ref))
    phi_0 = zeros(FT, length(theta_ref))

    for i in eachindex(theta_ref)
        if param_kind[i] == :sigma
            if param_is_hydrogen[i]
                theta_min[i] = FT(bounds_cfg.sigma_hydrogen_min)
                theta_max[i] = FT(bounds_cfg.sigma_hydrogen_max)
            else
                theta_min[i] = FT(bounds_cfg.sigma_heavy_min)
                theta_max[i] = FT(bounds_cfg.sigma_heavy_max)
            end
        else
            if param_is_hydrogen[i]
                theta_min[i] = FT(bounds_cfg.epsilon_hydrogen_min)
                theta_max[i] = FT(bounds_cfg.epsilon_hydrogen_max)
            else
                theta_min[i] = FT(bounds_cfg.epsilon_heavy_min)
                theta_max[i] = FT(bounds_cfg.epsilon_heavy_max)
            end
        end

        val_ref = clamp(theta_ref[i], theta_min[i] + clamp_eps, theta_max[i] - clamp_eps)
        val = (theta_max[i] - val_ref) / (val_ref - theta_min[i])
        phi_0[i] = (FT(1.0) / opt_cfg.k_sigmoid) * log(val)
    end

    phi_active = zeros(FT, length(theta_ref))
    theta_active = map_phi_to_theta(phi_active, theta_min, theta_max, phi_0, opt_cfg.k_sigmoid)

    # Rebuild injection maps for every leg so later reweighting can reuse the
    # same global parameter vector across different systems.
    idxs_by_leg = Dict{Symbol, Any}()
    for leg in cycle_cfg.legs
        leg_nonbonded_method = resolve_base_nonbonded_method(
            leg.electrostatics_method,
            leg.is_vacuum,
            leg.coulomb_softcore_model,
        )
        sys_leg = leg.name == ref_leg.name ? sys_ref : System(leg.pdb, ff; array_type=(leg.is_vacuum ? Array : AT), nonbonded_method=leg_nonbonded_method, nonbonded_energy_type=sim_cfg.nonbonded_energy_type)
        idxs_by_leg[leg.name] = build_index_maps(sys_leg, processed_atom_types)
    end

    return (
        phi_active=phi_active,
        theta_active=theta_active,
        param_names=param_names,
        trainable_param_names=trainable_param_names,
        trainable_param_indices=trainable_param_indices,
        trainable_position_map=trainable_position_map,
        solute_param_indices=solute_param_indices,
        solvent_param_indices=solvent_param_indices,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_0=phi_0,
        idxs_by_leg=idxs_by_leg,
    )
end


"""
    setup_macro_legs(cycle_cfg, sim_cfg, runtime, theta_active, idxs_by_leg,
                     T_coord, T_vol, T_en, macro_epoch, restart_rmsd_tol_nm)

Create or warm-start the AWH simulations for the current macro epoch.
"""
function setup_macro_legs(
    cycle_cfg::ThermodynamicCycleConfig,
    sim_cfg::SimulationConfig,
    state_schedules_by_leg::Dict{Symbol, <:ResolvedLegStateSchedule},
    runtime::RuntimeState,
    theta_active::Vector{FT},
    idxs_by_leg::Dict{Symbol, Any},
    T_coord,
    T_vol,
    T_en,
    macro_epoch::Int,
    restart_rmsd_tol_nm::FT,
) where {FT <: AbstractFloat}
    awh_by_leg = Dict{Symbol, Any}()
    sys_by_leg = Dict{Symbol, Any}()
    bounds_cfg = sim_cfg.parameter_bounds
    sigma_seed = FT(bounds_cfg.sigma_floor)u"nm"
    epsilon_seed = FT(bounds_cfg.epsilon_floor)u"kJ/mol"

    for leg in cycle_cfg.legs
        start_state = macro_epoch_awh_start_state(runtime, leg.name, macro_epoch)
        warm_start = start_state.warm_start
        restart_state = start_state.restart_state
        injected_bias = start_state.injected_bias
        state_schedule = state_schedules_by_leg[leg.name]
        leg_awh_control = resolve_leg_awh_control(sim_cfg.awh_control, leg)

        logger = AWHEnsembleLogger(T_coord, T_vol, T_en, sim_cfg.production_log_interval)
        awh_leg, sys_leg = setup_alchemical_awh(
            leg.pdb,
            sim_cfg.solute_idx;
            lambda_values=state_schedule.lambda,
            awh_control=leg_awh_control,
            is_vacuum=leg.is_vacuum,
            ensemble=leg.ensemble,
            logger=logger,
            injected_bias=injected_bias,
            optimized_params=theta_active,
            param_idxs=idxs_by_leg[leg.name],
            restart_state=restart_state,
            restart_active_idx=warm_start ? restart_state.active_idx : 1,
            warm_start=warm_start,
            electrostatics_method=leg.electrostatics_method,
            lambda_scheduler=leg.lambda_scheduler,
            coulomb_softcore_model=leg.coulomb_softcore_model,
            lj_softcore_model=leg.lj_softcore_model,
            sigma_seed=sigma_seed,
            epsilon_seed=epsilon_seed,
            array_type=leg.is_vacuum ? Array : sim_cfg.AT,
            nonbonded_energy_type=sim_cfg.nonbonded_energy_type,
        )
        awh_by_leg[leg.name] = awh_leg
        sys_by_leg[leg.name] = sys_leg

        cadence = resolve_awh_iteration_cadence(leg_awh_control)
        @info "AWH cadence ($(leg.name)): λ_sample_every=$(cadence.n_md_steps) md_steps | bias_update_every=$(cadence.update_freq) samples ($(cadence.effective_bias_update_md_steps) md_steps) | stats_log_every=$(cadence.stats_log_every_updates) updates | update_mode=$(cadence.auto_update_freq ? :auto : :override)"
        electrostatics_method = normalize_electrostatics_method_name(
            leg.electrostatics_method,
            leg.is_vacuum,
            leg.coulomb_softcore_model,
        )
        scheduler_name = normalize_lambda_scheduler_name(leg.lambda_scheduler)
        coulomb_model = electrostatics_method == :cutoff ?
            normalize_reaction_field_coulomb_model_name(leg.coulomb_softcore_model, Val(:cutoff)) :
            normalize_ewald_coulomb_model_name(leg.coulomb_softcore_model)
        lj_model = normalize_softcore_model_name(leg.lj_softcore_model)
        @info "AWH path ($(leg.name)): electrostatics=$electrostatics_method | scheduler=$scheduler_name | coulomb_softcore=$coulomb_model | lj_softcore=$lj_model"
        if !leg.is_vacuum
            λ_diag = solvent_lambda_schedule_diagnostics(state_schedule.lambda, leg.lambda_scheduler, FT)
            n_charge = count(entry -> entry.stage == :charge, λ_diag)
            n_lj = length(λ_diag) - n_charge
            @info "AWH λ schedule ($(leg.name)): n_states=$(length(λ_diag)) | charge_windows=$n_charge | lj_windows=$n_lj"
            for entry in λ_diag
                @info "  λ$(entry.idx): global=$(round(entry.global_lambda, digits=5)) | elec=$(round(entry.elec_lambda, digits=5)) | lj=$(round(entry.lj_lambda, digits=5)) | stage=$(entry.stage)"
            end
        end

        leg_label = uppercasefirst(String(leg.name))
        bias_reused = !isnothing(injected_bias)
        if warm_start
            restart_rmsd = coords_rmsd_nm(restart_state.coords, awh_leg.state.active_sys.coords)
            idx_match = restart_state.active_idx == awh_leg.state.active_idx
            if leg.is_vacuum
                @info "Macro Start ($leg_label): warm_start=true | bias_reused=$bias_reused | λ_prev=$(restart_state.active_idx) | λ_new=$(awh_leg.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6))"
            else
                prev_vol = FT(ustrip(volume(restart_state.boundary)))
                new_vol = FT(ustrip(volume(awh_leg.state.active_sys.boundary)))
                @info "Macro Start ($leg_label): warm_start=true | bias_reused=$bias_reused | λ_prev=$(restart_state.active_idx) | λ_new=$(awh_leg.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6)) | volume_ratio=$(round(new_vol / prev_vol, digits=6))"
            end
            if restart_rmsd > restart_rmsd_tol_nm
                @info "  [!] $leg_label restart RMSD above tolerance ($(restart_rmsd_tol_nm) nm)."
            end
        else
            @info "Macro Start ($leg_label): warm_start=false | bias_reused=false | awh_state_reset=true (cold-start from PDB)."
        end
    end

    return awh_by_leg, sys_by_leg
end


"""
    run_readiness_loop!(cycle_cfg, awh_by_leg, sys_by_leg, idxs_by_leg, runtime,
                        theta_active, param_names, ... )

Advance each leg through repeated Stage A blocks until all legs either pass the
Stage B probe and freeze, or one of them exhausts its AWH budget.
"""
function run_readiness_loop!(
    cycle_cfg::ThermodynamicCycleConfig,
    state_schedules_by_leg::Dict{Symbol, <:ResolvedLegStateSchedule},
    awh_by_leg::Dict{Symbol, Any},
    sys_by_leg::Dict{Symbol, Any},
    idxs_by_leg::Dict{Symbol, Any},
    runtime::RuntimeState,
    theta_active::Vector{FT},
    param_names::Vector{String},
    md_steps_budget::Int,
    md_steps_block::Int,
    md_steps_rewarm::Int,
    probe_steps_by_leg::Dict{Symbol, Int},
    probe_stride_by_leg::Dict{Symbol, Int},
    probe_min_frames_by_leg::Dict{Symbol, Int},
    probe_max_frames_by_leg::Dict{Symbol, Int},
    awh_budget_time,
    awh_convergence_tol::FT,
    awh_min_linear_neff::Int,
    awh_min_lambda_ess::Int,
    awh_split_tol_kT::FT,
    awh_parity_tol_kT::FT,
    awh_parity_gate_mode::Symbol,
    awh_parity_support_threshold::FT,
    awh_stageB_support_allow_missing::Int,
    awh_parity_near_pass_factor::FT,
    awh_tail_lag::Int,
    awh_min_round_trips::Int,
    awh_endpoint_target_ratio::FT,
    awh_solvent_tail_lj_max::FT,
    awh_solvent_tail_min_state_occupancy::FT,
    awh_solvent_endpoint_min_fraction::FT,
    awh_stageA_history_blocks::Int,
    awh_stageA_stable_blocks::Int,
    awh_stageB_cooldown_blocks::Int,
    awh_stageB_near_pass_cooldown_blocks::Int,
    awh_stageB_probe_growth_ns::FT,
    awh_stageB_probe_near_pass_scale::FT,
    awh_stageB_probe_max_factor::FT,
    awh_min_initial_df_threshold::FT,
    awh_min_initial_state_occupancy::FT,
    awh_stageB_soften_failures_threshold::Int,
    awh_stageB_soften_factor::FT,
    awh_stageA_streak_growth_factor::FT,
    awh_stageB_cooldown_growth_factor::FT,
    awh_stageA_max_streak::Int,
    awh_stageB_max_cooldown::Int,
    awh_control::AWHControlConfig,
    awh_probe_discard_fraction::Float64,
    beta_val::BT,
    p0_energy_per_vol::PT,
) where {FT <: AbstractFloat, BT <: AbstractFloat, PT <: AbstractFloat}
    ET = promote_energy_analysis_type(
        beta_val,
        p0_energy_per_vol,
        awh_split_tol_kT,
        awh_parity_tol_kT,
        awh_parity_support_threshold,
    )
    stageA_default = StageAStats(df_mean=FT(Inf), lambda_ess=one(FT), tau_int_est=zero(FT), linear_neff=zero(FT))
    stageB_default = StageBStats(split_gap=ET(Inf), parity_gap=ET(Inf), dG_half_1=ET(NaN), dG_half_2=ET(NaN))

    stageA_stats_by_leg = Dict{Symbol, StageAStats}(leg.name => stageA_default for leg in cycle_cfg.legs)
    stageB_stats_by_leg = Dict{Symbol, StageBStats}(leg.name => stageB_default for leg in cycle_cfg.legs)
    stageA_streak = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    stageB_cooldown = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    stageB_consecutive_failures = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    spent_steps = Dict{Symbol, Int}(leg.name => 0 for leg in cycle_cfg.legs)
    leg_status = Dict{Symbol, Symbol}(leg.name => :active for leg in cycle_cfg.legs)
    split_gap_by_leg = Dict{Symbol, ET}(leg.name => ET(Inf) for leg in cycle_cfg.legs)
    parity_gap_by_leg = Dict{Symbol, ET}(leg.name => ET(Inf) for leg in cycle_cfg.legs)
    near_pass_cooldown_blocks = max(0, awh_stageB_near_pass_cooldown_blocks)
    base_probe_steps_by_leg = copy(probe_steps_by_leg)
    current_probe_steps_by_leg = copy(probe_steps_by_leg)
    stageA_tail_state_idxs_by_leg = Dict{Symbol, Vector{Int}}()
    stageA_endpoint_state_idxs_by_leg = Dict{Symbol, Vector{Int}}()
    stageA_tail_min_occ_floor_by_leg = Dict{Symbol, FT}()
    stageA_endpoint_high_min_fraction_by_leg = Dict{Symbol, FT}()
    stageA_history_window_samples_by_leg = Dict{Symbol, Int}()
    for leg in cycle_cfg.legs
        leg_awh_control = resolve_leg_awh_control(awh_control, leg)
        stageA_history_window_samples_by_leg[leg.name] = stage_a_history_window_sample_count(
            md_steps_block,
            leg_awh_control,
            awh_stageA_history_blocks,
        )
        if leg.is_vacuum
            stageA_tail_state_idxs_by_leg[leg.name] = Int[]
            stageA_endpoint_state_idxs_by_leg[leg.name] = solvent_stage_a_endpoint_state_indices(
                leg,
                state_schedules_by_leg[leg.name];
                lj_lambda_max=zero(FT),
            )
            stageA_tail_min_occ_floor_by_leg[leg.name] = zero(FT)
            stageA_endpoint_high_min_fraction_by_leg[leg.name] = zero(FT)
            continue
        end
        stageA_tail_state_idxs_by_leg[leg.name] = solvent_stage_a_tail_state_indices(
            leg,
            state_schedules_by_leg[leg.name];
            lj_lambda_max=awh_solvent_tail_lj_max,
        )
        stageA_endpoint_state_idxs_by_leg[leg.name] = solvent_stage_a_endpoint_state_indices(
            leg,
            state_schedules_by_leg[leg.name];
            lj_lambda_max=awh_solvent_tail_lj_max,
        )
        stageA_tail_min_occ_floor_by_leg[leg.name] = max(zero(FT), awh_solvent_tail_min_state_occupancy)
        stageA_endpoint_high_min_fraction_by_leg[leg.name] = max(zero(FT), awh_solvent_endpoint_min_fraction)
    end

    for leg in cycle_cfg.legs
        runtime.active_bias[leg.name] = extract_awh_data(awh_by_leg[leg.name])
    end
    sync_runtime_aliases!(runtime)

    for leg in cycle_cfg.legs
        awh_leg = awh_by_leg[leg.name]
        if !awh_leg.state.in_initial_stage
            continue
        end
        rewarm_steps = min(md_steps_rewarm, md_steps_budget)
        println("Running $(uppercasefirst(String(leg.name))) AWH Leg (Initial Rewarm)...")
        timed_phase("Initial Rewarm", String(leg.name); md_steps=rewarm_steps) do
            simulate!(awh_leg, rewarm_steps)
        end
        spent_steps[leg.name] += rewarm_steps
        runtime.active_bias[leg.name] = extract_awh_data(awh_leg)
    end
    sync_runtime_aliases!(runtime)

    awh_ready = false
    awh_readiness_reason = :not_checked

    while true
        for leg in cycle_cfg.legs
            name = leg.name
            leg_awh_control = resolve_leg_awh_control(awh_control, leg)
            if leg_status[name] != :active
                continue
            end

            awh_leg = awh_by_leg[name]::AWHSimulation
            state_schedule = state_schedules_by_leg[name]
            remaining_steps = md_steps_budget - spent_steps[name]
            if remaining_steps <= 0
                leg_status[name] = :budget_exhausted
                continue
            end

            block_steps = min(md_steps_block, remaining_steps)
            println("Running $(uppercasefirst(String(name))) AWH Leg (Stage A Block)...")
            timed_phase("Stage A Block", String(name); md_steps=block_steps) do
                simulate!(awh_leg, block_steps)
            end
            spent_steps[name] += block_steps
            runtime.active_bias[name] = extract_awh_data(awh_leg)

            n_states = length(state_schedule.lambda)
            dynamic_endpoint_fraction = FT(awh_endpoint_target_ratio / n_states)

            stageA_stats_by_leg[name] = StageAStats(evaluate_stage_a_readiness(
                awh_leg,
                awh_convergence_tol;
                tail_lag=awh_tail_lag,
                min_lambda_ess=awh_min_lambda_ess,
                min_linear_neff=awh_min_linear_neff,
                min_round_trips=awh_min_round_trips,
                endpoint_min_fraction=dynamic_endpoint_fraction,
                history_window_length=stageA_history_window_samples_by_leg[name],
                tail_state_idxs=stageA_tail_state_idxs_by_leg[name],
                endpoint_state_idxs=stageA_endpoint_state_idxs_by_leg[name],
                tail_min_state_occupancy_floor=stageA_tail_min_occ_floor_by_leg[name],
                endpoint_high_min_fraction_abs=stageA_endpoint_high_min_fraction_by_leg[name],
                low_idx=state_schedule.coupled_state_idx,
                high_idx=state_schedule.decoupled_state_idx,
            ))

            if !awh_leg.state.in_initial_stage
                df_ok = stageA_stats_by_leg[name].df_mean <= awh_min_initial_df_threshold
                occ_ok = stageA_stats_by_leg[name].min_state_occupancy >= awh_min_initial_state_occupancy
                if !(df_ok && occ_ok)
                    @info "[!] Strict gating: Reverting Stage A ($(name)) to initial stage (df=$(round(stageA_stats_by_leg[name].df_mean, digits=4)), min_occ=$(round(stageA_stats_by_leg[name].min_state_occupancy, digits=4)))."
                    awh_leg.state.in_initial_stage = true
                    awh_leg.state.N_bias = FT(leg_awh_control.initial_n_bias)
                    awh_leg.state.n_accum = 0
                    fill!(awh_leg.state.w_seg, zero(FT))
                    fill!(awh_leg.state.w2_seg, zero(FT))
                    empty!(awh_leg.state.visited_windows)
                end
            end

            if stageA_stats_by_leg[name].ready
                stageA_streak[name] += 1
            else
                stageA_streak[name] = 0
            end

            failures = stageB_consecutive_failures[name]
            retry_controls = stage_b_retry_controls(
                awh_stageA_stable_blocks,
                awh_stageB_cooldown_blocks,
                failures,
                awh_stageA_streak_growth_factor,
                awh_stageB_cooldown_growth_factor,
                awh_stageA_max_streak,
                awh_stageB_max_cooldown,
            )
            current_target_streak = retry_controls.target_streak

            if stageA_streak[name] >= current_target_streak
                if stageB_cooldown[name] > 0
                    @info "Stage B ($(name)) cooldown active: remaining_checks=$(stageB_cooldown[name]); skipping probe this block."
                    stageB_cooldown[name] -= 1
                else
                    probe_steps = current_probe_steps_by_leg[name]
                    effective_probe_max_frames = scaled_probe_frame_cap(
                        probe_max_frames_by_leg[name],
                        base_probe_steps_by_leg[name],
                        probe_steps,
                    )
                    @info "Stage A ($(name)) reached stable streak ($(stageA_streak[name])/$(current_target_streak)); entering Stage B probe | probe_steps=$probe_steps | probe_ns=$(round(steps_to_ns(probe_steps), digits=4))"
                    probe_stats = run_stage_b_probe(
                        awh_leg,
                        sys_by_leg[name],
                        theta_active,
                        param_names,
                        idxs_by_leg[name],
                        state_schedule.coupled_state_idx,
                        state_schedule.decoupled_state_idx,
                        beta_val,
                        awh_split_tol_kT,
                        awh_parity_tol_kT;
                        md_steps_probe=probe_steps,
                        leg_name=String(name),
                        probe_num_md_steps=leg.probe_awh_seed_num_md_steps,
                        include_pv=leg.include_pv,
                        P0_energy_per_vol=leg.include_pv ? p0_energy_per_vol : zero(p0_energy_per_vol),
                        probe_frame_stride=probe_stride_by_leg[name],
                        probe_min_frames=probe_min_frames_by_leg[name],
                        probe_max_frames=effective_probe_max_frames,
                        awh_probe_discard_fraction=awh_probe_discard_fraction,
                        parity_gate_mode=awh_parity_gate_mode,
                        parity_support_threshold=awh_parity_support_threshold,
                        support_allow_missing=awh_stageB_support_allow_missing,
                        parity_near_pass_factor=awh_parity_near_pass_factor,
                        probe_tail_state_idxs=stageA_tail_state_idxs_by_leg[name],
                        probe_tail_min_state_occupancy_floor=stageA_tail_min_occ_floor_by_leg[name],
                    )
                    stageB_stats_by_leg[name] = StageBStats(merge(
                        probe_stats,
                        (
                            n_accumulated_frames=probe_stats.n_frames,
                            n_probe_segments=probe_stats.n_frames > 0 ? 1 : 0,
                            accumulation_mode=:single_probe,
                            support_switch_ready=false,
                            n_evicted_frames=0,
                        ),
                    ))

                    split_gap_by_leg[name] = stageB_stats_by_leg[name].split_gap
                    parity_gap_by_leg[name] = stageB_stats_by_leg[name].parity_gap
                    if stageB_stats_by_leg[name].ready
                        stageB_consecutive_failures[name] = 0
                        current_probe_steps_by_leg[name] = base_probe_steps_by_leg[name]
                        stageB_cooldown[name] = 0
                        # Once a leg passes Stage B it is frozen for the rest of
                        # the macro epoch; the bias and restart snapshot are kept
                        # for later production/resimulation.
                        leg_status[name] = :ready_frozen
                        @info "$(uppercasefirst(String(name))) leg frozen after passing Stage B (split_gap=$(round(split_gap_by_leg[name], digits=4)) kT, parity_gap=$(round(parity_gap_by_leg[name], digits=4)) kT)."
                    else
                        stageA_streak[name] = 0
                        stageB_consecutive_failures[name] += 1
                        
                        if stageB_consecutive_failures[name] >= awh_stageB_soften_failures_threshold
                            new_neff = awh_leg.state.N_eff * awh_stageB_soften_factor
                            @info "[!] Stage B ($(name)) softening AWH bias: N_eff reduced from $(round(awh_leg.state.N_eff, digits=1)) to $(round(new_neff, digits=1))."
                            awh_leg.state.N_eff = new_neff
                            stageB_consecutive_failures[name] = 0
                        end

                        retry_controls = stage_b_retry_controls(
                            awh_stageA_stable_blocks,
                            awh_stageB_cooldown_blocks,
                            stageB_consecutive_failures[name],
                            awh_stageA_streak_growth_factor,
                            awh_stageB_cooldown_growth_factor,
                            awh_stageA_max_streak,
                            awh_stageB_max_cooldown,
                        )
                        probe_policy = stage_b_next_probe_policy(
                            current_probe_steps_by_leg[name],
                            base_probe_steps_by_leg[name],
                            stageB_stats_by_leg[name],
                            retry_controls.cooldown_blocks,
                            near_pass_cooldown_blocks,
                            time_to_steps_floor(awh_stageB_probe_growth_ns * 1.0u"ns"),
                            awh_stageB_probe_near_pass_scale,
                            awh_stageB_probe_max_factor,
                        )
                        current_probe_steps_by_leg[name] = probe_policy.next_probe_steps
                        stageB_cooldown[name] = probe_policy.cooldown_blocks
                        if probe_policy.policy == :grow_sampling
                            @info "Stage B ($(name)) failed ($(stageB_stats_by_leg[name].failure_mode)); increasing next probe to $(probe_policy.next_probe_steps) steps ($(round(steps_to_ns(probe_policy.next_probe_steps), digits=4)) ns) and scheduling cooldown=$(probe_policy.cooldown_blocks)."
                        elseif probe_policy.policy == :stay_bias_error
                            @info "Stage B ($(name)) failed ($(stageB_stats_by_leg[name].failure_mode)); keeping probe at $(probe_policy.next_probe_steps) steps and scheduling cooldown=$(probe_policy.cooldown_blocks) for Stage A learning."
                        elseif probe_policy.policy == :near_pass
                            @info "Stage B ($(name)) near pass; retrying with probe=$(probe_policy.next_probe_steps) steps ($(round(steps_to_ns(probe_policy.next_probe_steps), digits=4)) ns) after cooldown=$(probe_policy.cooldown_blocks)."
                        elseif stageB_cooldown[name] > 0
                            @info "Stage B ($(name)) failed; scheduling cooldown for $(stageB_cooldown[name]) stable-check opportunities."
                        end
                    end
                end
            end

            if leg_status[name] == :active && spent_steps[name] >= md_steps_budget
                leg_status[name] = :budget_exhausted
            end
        end

        sync_runtime_aliases!(runtime)

        status_msg = join(["$(leg.name)=$(leg_status[leg.name])" for leg in cycle_cfg.legs], " ")
        spent_msg = join(
            [
                "$(leg.name)=$(round(steps_to_ns(spent_steps[leg.name]), digits=2))/$(round(ustrip(u"ns", awh_budget_time), digits=2))"
                for leg in cycle_cfg.legs
            ],
            " | ",
        )
        @info "AWH Block Status: $status_msg | spent_ns: $spent_msg"

        for leg in cycle_cfg.legs
            name = leg.name
            statsA = stageA_stats_by_leg[name]
            low_occ_msg = isempty(statsA.low_occupancy_states) ? "-" : join(statsA.low_occupancy_states, ",")
            endpoint_state_idxs = stageA_endpoint_state_idxs_by_leg[name]
            endpoint_band_msg = format_state_idx_span(endpoint_state_idxs)
            tail_msg = ""
            if !isempty(stageA_tail_state_idxs_by_leg[name])
                tail_low_msg = isempty(statsA.tail_low_occupancy_states) ? "-" : join(statsA.tail_low_occupancy_states, ",")
                tail_msg = " | tail=(Σ=$(round(statsA.tail_occupancy, digits=3)), min=$(round(statsA.tail_min_state_occupancy, digits=4)), ok=$(statsA.tail_ready)) | tail_low=$tail_low_msg"
            end
            @info "  Stage A ($(name)): df=$(round(statsA.df_mean, digits=6)) (ok=$(statsA.df_ready)) | ess=$(round(statsA.lambda_ess, digits=1)) (ok=$(statsA.lambda_ess_ready)) | lin_neff=$(round(statsA.linear_neff, digits=1)) (ok=$(statsA.neff_ready)) | tau_int_est=$(round(statsA.tau_int_est, digits=2)) | switches=$(statsA.switch_count) | dwell_samples=(mean=$(round(statsA.mean_residence, digits=2)), med=$(round(statsA.median_residence, digits=2))) | rt=$(statsA.round_trips) (ok=$(statsA.round_trip_ready)) | endpt_recent=(low=$(round(statsA.endpoint_low, digits=3)), band=$(round(statsA.endpoint_high, digits=3)); band=$(endpoint_band_msg), req>=$(round(statsA.endpoint_high_required, digits=3))) (ok=$(statsA.endpoint_ready))$tail_msg | occ_min=$(round(statsA.min_state_occupancy, digits=4)) | low_occ=$low_occ_msg | n_hist_recent=$(statsA.n_hist_recent) | n_hist_total=$(statsA.n_hist) | streak=$(stageA_streak[name]) | cooldown=$(stageB_cooldown[name]) | failures=$(stageB_consecutive_failures[name])"

            statsB = stageB_stats_by_leg[name]
            if statsB.n_frames > 0
                n_states = length(state_schedules_by_leg[name].lambda)
                @info "  Stage B ($(name)): split_gap=$(round(statsB.split_gap, digits=4)) kT (ok=$(statsB.split_ready)) | parity_gap=$(round(statsB.parity_gap, digits=4)) kT (ok=$(statsB.parity_ready)) | raw_parity=$(round(statsB.raw_parity_gap, digits=4)) | supported_parity=$(round(statsB.supported_parity_gap, digits=4)) | endpoint_parity=$(round(statsB.endpoint_parity_gap, digits=4)) (ok=$(statsB.endpoint_parity_ready)) | support_coverage=$(statsB.n_supported_states)/$(n_states) (required>=$(statsB.required_supported_states), ok=$(statsB.support_coverage_ready)) @ ess>=$(round(statsB.support_threshold, digits=1)) | failure=$(statsB.failure_mode) | near_pass=$(statsB.near_pass) | mode=$(statsB.accumulation_mode) | frames=$(statsB.n_frames) | accumulated_frames=$(statsB.n_accumulated_frames) | probe_segments=$(statsB.n_probe_segments)"
                if !isempty(statsB.diagnostics)
                    @info "    Stage B Diagnostics ($(name)): $(statsB.diagnostics)"
                end
            end
        end

        all_ready = all(leg_status[leg.name] == :ready_frozen for leg in cycle_cfg.legs)
        if all_ready
            awh_ready = true
            awh_readiness_reason = :ready
            break
        end

        any_budget_exhausted = any(leg_status[leg.name] == :budget_exhausted for leg in cycle_cfg.legs)
        if any_budget_exhausted
            awh_ready = false
            awh_readiness_reason = :awh_not_ready_budget
            break
        end
    end

    return (
        awh_ready=awh_ready,
        awh_readiness_reason=awh_readiness_reason,
        stageA_stats_by_leg=stageA_stats_by_leg,
        stageB_stats_by_leg=stageB_stats_by_leg,
        split_gap_by_leg=split_gap_by_leg,
        parity_gap_by_leg=parity_gap_by_leg,
        leg_status=leg_status,
    )
end


"""
    collect_production_artifacts!(cycle_cfg, awh_by_leg, sys_by_leg, idxs_by_leg,
                                  runtime, theta_active, param_names, md_steps_prod,
                                  p0_energy_per_vol)

Run the final production segment for each ready leg and package the results for
the optimization phase.
"""
function collect_production_artifacts!(
    cycle_cfg::ThermodynamicCycleConfig,
    state_schedules_by_leg::Dict{Symbol, <:ResolvedLegStateSchedule},
    awh_by_leg::Dict{Symbol, Any},
    sys_by_leg::Dict{Symbol, Any},
    idxs_by_leg::Dict{Symbol, Any},
    runtime::RuntimeState,
    theta_active::Vector{FT},
    param_names::Vector{String},
    md_steps_prod::Int,
    p0_energy_per_vol::PT,
) where {FT <: AbstractFloat, PT <: AbstractFloat}
    for leg in cycle_cfg.legs
        runtime.active_bias[leg.name] = extract_awh_data(awh_by_leg[leg.name])
    end
    sync_runtime_aliases!(runtime)

    println("Running Production Runs (Extended Ensemble) [Readiness-Passed Dataset]...")
    for leg in cycle_cfg.legs
        clear_awh_logger_histories!(awh_by_leg[leg.name])
    end

    artifacts = LegArtifacts[]
    for leg in cycle_cfg.legs
        name = leg.name
        # Production frames must be sampled under the exact same frozen bias that
        # the offline MBAR step later uses in its mixture denominator.
        awh_prod, bias_data = build_frozen_bias_awh_sim(awh_by_leg[name], md_steps_prod)
        runtime.active_bias[name] = bias_data
        simulate!(awh_prod, md_steps_prod)

        # Persist the end-of-production state so the next macro epoch can start
        # close to the current basin.
        runtime.restart_cache[name] = capture_restart_state(awh_prod)

        logger_prod = get_production_logger(awh_prod, String(name))
        neighbors = precompute_neighbors(logger_prod, awh_prod.state.active_sys)
        u_ref, _ = evaluate_ensemble(
            logger_prod,
            neighbors,
            awh_prod,
            sys_by_leg[name],
            theta_active,
            param_names,
            idxs_by_leg[name]...;
            compute_gradients=false,
        )

        push!(artifacts, LegArtifacts(
            name=name,
            coefficient=FT(leg.coefficient),
            include_pv=leg.include_pv,
            p0_energy_per_vol=leg.include_pv ? p0_energy_per_vol : zero(p0_energy_per_vol),
            n_states=length(state_schedules_by_leg[name].lambda),
            coupled_state_idx=state_schedules_by_leg[name].coupled_state_idx,
            decoupled_state_idx=state_schedules_by_leg[name].decoupled_state_idx,
            awh_prod=awh_prod,
            logger_prod=logger_prod,
            neighbors=neighbors,
            u_ref=u_ref,
            sys_base=sys_by_leg[name],
            active_bias=bias_data,
            idxs=idxs_by_leg[name],
        ))
    end

    sync_runtime_aliases!(runtime)
    return artifacts
end


"""
    run_pipeline(; sim_cfg=default_simulation_config(),
                   opt_cfg=default_optimization_config())

Run the full outer loop:

1. Apply the requested simulation configuration.
2. Repeatedly sample each cycle leg until readiness passes.
3. Collect production artifacts and update the active parameter vector.

The returned `RuntimeState` contains the final parameters, reusable bias state,
and restart snapshots from the last macro epoch.
"""
function run_pipeline(; sim_cfg::SimulationConfig=default_simulation_config(), opt_cfg::OptimizationConfig=default_optimization_config())
    apply_simulation_config!(sim_cfg)
    FT = sim_cfg.FT
    cycle_cfg = validate_cycle_config(
        resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=FT,
    )
    state_schedules_by_leg = Dict{Symbol, ResolvedLegStateSchedule{FT}}(
        leg.name => resolve_leg_state_schedule(leg, sim_cfg.lambda_schedule, FT)
        for leg in cycle_cfg.legs
    )
    runtime = RuntimeState()

    thermo_AT = default_energy_analysis_type(sim_cfg)
    dG_std_corr = compute_standard_state_correction(cycle_cfg, thermo_AT)

    md_steps_budget = time_to_steps_floor(sim_cfg.awh_budget_time)
    md_steps_block = time_to_steps_round(sim_cfg.awh_block_time)
    md_steps_prod = time_to_steps_floor(sim_cfg.md_time_production)
    md_steps_rewarm = min(
        max(1, Int(round(md_steps_budget * opt_cfg.rewarm_fraction))),
        md_steps_budget,
    )

    probe_steps_by_leg = Dict{Symbol, Int}()
    probe_stride_by_leg = Dict{Symbol, Int}()
    probe_min_frames_by_leg = Dict{Symbol, Int}()
    probe_max_frames_by_leg = Dict{Symbol, Int}()
    for leg in cycle_cfg.legs
        probe_steps_by_leg[leg.name] = time_to_steps_round(leg.probe_time)
        if leg.is_vacuum
            probe_stride_by_leg[leg.name] = max(1, sim_cfg.awh_probe_reweight_stride_vac)
            probe_min_frames_by_leg[leg.name] = max(2, sim_cfg.awh_probe_reweight_min_frames_vac)
            probe_max_frames_by_leg[leg.name] = max(0, sim_cfg.awh_probe_reweight_max_frames_vac)
        else
            probe_stride_by_leg[leg.name] = max(1, sim_cfg.awh_probe_reweight_stride_solv)
            probe_min_frames_by_leg[leg.name] = max(2, sim_cfg.awh_probe_reweight_min_frames_solv)
            probe_max_frames_by_leg[leg.name] = max(0, sim_cfg.awh_probe_reweight_max_frames_solv)
        end
    end

    T_coord, T_vol, T_en = awh_logger_value_types(sim_cfg)

    pstate = initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)
    phi_active = pstate.phi_active
    theta_active = pstate.theta_active
    param_names = pstate.param_names
    trainable_param_names = pstate.trainable_param_names
    trainable_param_indices = pstate.trainable_param_indices
    trainable_position_map = pstate.trainable_position_map
    solute_param_indices = pstate.solute_param_indices
    solvent_param_indices = pstate.solvent_param_indices
    theta_min = pstate.theta_min
    theta_max = pstate.theta_max
    phi_0 = pstate.phi_0
    idxs_by_leg = pstate.idxs_by_leg

    for macro_epoch in 1:opt_cfg.max_macro_epochs
        println("\n>>> STARTING MACRO EPOCH $macro_epoch <<<")

        awh_by_leg, sys_by_leg = setup_macro_legs(
            cycle_cfg,
            sim_cfg,
            state_schedules_by_leg,
            runtime,
            theta_active,
            idxs_by_leg,
            T_coord,
            T_vol,
            T_en,
            macro_epoch,
            opt_cfg.restart_rmsd_tol_nm,
        )

        first_leg = first(cycle_cfg.legs)
        e_unit = sys_by_leg[first_leg.name].energy_units
        beta_val = thermo_AT(1.0 / ustrip(uconvert(e_unit, Unitful.R * T0)))
        p0_energy_per_vol = thermo_AT(ustrip(uconvert(e_unit, P0 * thermo_AT(1.0)u"nm^3" * Unitful.Na)))

        dG_exp_physical = thermo_AT(cycle_cfg.target_dG_kcal_mol) * thermo_AT(4.184)
        dG_exp = dG_exp_physical * beta_val

        readiness_result = run_readiness_loop!(
            cycle_cfg,
            state_schedules_by_leg,
            awh_by_leg,
            sys_by_leg,
            idxs_by_leg,
            runtime,
            theta_active,
            param_names,
            md_steps_budget,
            md_steps_block,
            md_steps_rewarm,
            probe_steps_by_leg,
            probe_stride_by_leg,
            probe_min_frames_by_leg,
            probe_max_frames_by_leg,
            sim_cfg.awh_budget_time,
            FT(opt_cfg.awh_convergence_tol),
            opt_cfg.awh_min_linear_neff,
            opt_cfg.awh_min_lambda_ess,
            FT(opt_cfg.awh_split_tol_kT),
            FT(opt_cfg.awh_parity_tol_kT),
            opt_cfg.awh_parity_gate_mode,
            FT(opt_cfg.awh_parity_support_threshold),
            opt_cfg.awh_stageB_support_allow_missing,
            FT(opt_cfg.awh_parity_near_pass_factor),
            opt_cfg.awh_tail_lag,
            opt_cfg.awh_min_round_trips,
            FT(opt_cfg.awh_endpoint_target_ratio),
            FT(opt_cfg.awh_solvent_tail_lj_max),
            FT(opt_cfg.awh_solvent_tail_min_state_occupancy),
            FT(opt_cfg.awh_solvent_endpoint_min_fraction),
            opt_cfg.awh_stageA_history_blocks,
            opt_cfg.awh_stageA_stable_blocks,
            opt_cfg.awh_stageB_cooldown_blocks,
            opt_cfg.awh_stageB_near_pass_cooldown_blocks,
            FT(opt_cfg.awh_stageB_probe_growth_ns),
            FT(opt_cfg.awh_stageB_probe_near_pass_scale),
            FT(opt_cfg.awh_stageB_probe_max_factor),
            FT(opt_cfg.awh_min_initial_df_threshold),
            FT(opt_cfg.awh_min_initial_state_occupancy),
            opt_cfg.awh_stageB_soften_failures_threshold,
            FT(opt_cfg.awh_stageB_soften_factor),
            FT(opt_cfg.awh_stageA_streak_growth_factor),
            FT(opt_cfg.awh_stageB_cooldown_growth_factor),
            opt_cfg.awh_stageA_max_streak,
            opt_cfg.awh_stageB_max_cooldown,
            sim_cfg.awh_control,
            Float64(sim_cfg.awh_probe_discard_fraction),
            beta_val,
            p0_energy_per_vol,
        )

        if !readiness_result.awh_ready
            stageA_summary = join(
                [
                    "$(leg.name)=df:$(round(readiness_result.stageA_stats_by_leg[leg.name].df_mean, digits=6)) ess:$(round(readiness_result.stageA_stats_by_leg[leg.name].lambda_ess, digits=1)) rt:$(readiness_result.stageA_stats_by_leg[leg.name].round_trips)"
                    for leg in cycle_cfg.legs
                ],
                " | ",
            )
            parity_summary = join(
                [
                    "$(leg.name)=$(round(readiness_result.parity_gap_by_leg[leg.name], digits=4))"
                    for leg in cycle_cfg.legs
                ],
                ", ",
            )
            status_summary = join(
                ["$(leg.name)=$(readiness_result.leg_status[leg.name])" for leg in cycle_cfg.legs],
                " ",
            )
            @info "Macro $macro_epoch skipped: strict AWH readiness failed ($(readiness_result.awh_readiness_reason)). Status=[$status_summary] | StageA=[$stageA_summary] | Parity gaps=[$parity_summary] kT"
            continue
        end

        leg_artifacts = collect_production_artifacts!(
            cycle_cfg,
            state_schedules_by_leg,
            awh_by_leg,
            sys_by_leg,
            idxs_by_leg,
            runtime,
            theta_active,
            param_names,
            md_steps_prod,
            p0_energy_per_vol,
        )

        GC.gc()

        opt_result = run_optimization_phase!(
            phi_active,
            theta_active,
            leg_artifacts,
            param_names,
            trainable_param_names,
            trainable_param_indices,
            trainable_position_map,
            solute_param_indices,
            solvent_param_indices,
            theta_min,
            theta_max,
            phi_0,
            beta_val,
            dG_std_corr,
            dG_exp,
            opt_cfg,
        )

        split_gap_max = maximum(values(readiness_result.split_gap_by_leg))
        parity_summary = join(
            [
                "$(leg.name)=$(round(readiness_result.parity_gap_by_leg[leg.name], digits=4))"
                for leg in cycle_cfg.legs
            ],
            ", ",
        )

        @info "Macro $macro_epoch Residual Summary: Start = $(round(opt_result.macro_start_residual, digits=3)) | Best = $(round(opt_result.best_macro_residual, digits=3)) (Epoch $(opt_result.best_macro_epoch)) | End = $(round(opt_result.macro_end_residual, digits=3)) | Exit = $(opt_result.phase2_exit_reason) | AWH split-gap(max) = $(round(split_gap_max, digits=4)) kT | Parity gaps = [$parity_summary] kT"
    end

    runtime.phi_active = phi_active
    runtime.theta_active = theta_active
    sync_runtime_aliases!(runtime)
    return runtime
end
