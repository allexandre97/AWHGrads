##
using Revise
using Molly
using CUDA
using Unitful
using StatsBase
using LinearAlgebra

##

##
# --- Simulation Constants ---
device!(1)
FT = Float32
AT = CuArray
Δt = FT(1)u"fs"
T0 = FT(310)u"K"
P0 = FT(1)u"bar"

lambda_schedule = FT.(range(1.0, stop=0.0, length=21))
num_lambda_states = length(lambda_schedule)
target_rho = FT(1.0 / num_lambda_states)

# Evaluate Standard State Correction (Gas to 1M Liquid)
V_gas = ustrip(u"nm^3", Unitful.k * T0 / P0)
V_std = ustrip(u"nm^3", 1.0u"L" / (1.0u"mol" * Unitful.Na))
dG_std_corr = FT(log(V_gas / V_std))

# --- Force Field Setup ---
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")  
ff_dir   = joinpath(data_dir, "force_fields")  
ff = MolecularForceField(FT, joinpath.(ff_dir, ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"])...; units=true)  


##
include("src/gradients_core.jl")
include("src/setup.jl")
include("src/logging_utils.jl")
include("src/ensemble_eval.jl")
include("src/readiness.jl")
include("src/transforms.jl")
include("src/index_maps.jl")

##
# ==============================================================================
# --- MAIN PIPELINE ORCHESTRATION ---
# ==============================================================================

awh_budget_time = FT(20)u"ns"
awh_block_time = FT(1.0)u"ns"
awh_probe_time_solv = FT(0.75)u"ns"
awh_probe_time_vac = FT(0.25)u"ns"
md_time_production  = FT(0.1)u"ns"

md_steps_budget = Int(floor(uconvert(unit(Δt), awh_budget_time) / Δt))
md_steps_block = max(1, Int(round(uconvert(unit(Δt), awh_block_time) / Δt)))
md_steps_probe_solv = max(1, Int(round(uconvert(unit(Δt), awh_probe_time_solv) / Δt)))
md_steps_probe_vac = max(1, Int(round(uconvert(unit(Δt), awh_probe_time_vac) / Δt)))
md_steps_prod = Int(floor(uconvert(unit(Δt), md_time_production) / Δt))
solute_idx = 1:9  

production_log_interval = 100  
awh_convergence_tol = FT(1e-3)  
rewarm_fraction = FT(0.05)
md_steps_rewarm = max(1, Int(round(md_steps_budget * rewarm_fraction)))
md_steps_rewarm = min(md_steps_rewarm, md_steps_budget)

T_coord = typeof(FT(1.0)u"nm")  
T_vol   = typeof(FT(1.0)u"nm^3")  
T_en    = typeof(FT(1.0)u"kJ * mol^-1")  

max_macro_epochs = 30
huber_delta = FT(2.0)  

# Optimization hyperparameters
kl_target = FT(0.1)
eigenvalue_tol_scale = FT(1e-2)
min_phi_step = FT(5e-4)
max_phi_step_solute = FT(0.35)
max_phi_step_solvent = FT(0.035)
tiny_alpha_cutoff = FT(0.015625)
max_tiny_alpha_hits = 2
restart_rmsd_tol_nm = FT(1e-5)
optimize_solvent = false

theoretical_ess_ratio = exp(FT(-2.0) * kl_target)
ess_threshold_ratio = FT(0.22) * theoretical_ess_ratio
awh_min_linear_neff = 3000
awh_split_tol_kT = FT(0.5)
awh_parity_tol_kT = FT(0.1)
awh_tail_lag = 10
awh_min_round_trips = 3
awh_endpoint_min_fraction = FT(0.03)
awh_stageA_stable_blocks = 2

active_bias_solv = nothing  
active_bias_vac  = nothing  
restart_cache_solv = nothing
restart_cache_vac = nothing

# Scaled sigmoid transformation variables
k_sigmoid = FT(1.0)

# ==============================================================================
# --- INITIALIZE PARAMETER MAPPINGS (RUNS ONCE) ---
# ==============================================================================
theta_ref    = Vector{FT}()
phi_active   = Vector{FT}()
theta_active = Vector{FT}()
param_names  = Vector{String}()  

sys_dummy = System("ethanol_solv.pdb", ff; array_type=AT, nonbonded_method=:none)
processed_atom_types = Dict{String, Int}()  
atoms_cpu = Molly.from_device(sys_dummy.atoms)  

solute_param_indices = Int[]
solvent_param_indices = Int[]

for idx in eachindex(atoms_cpu)  
    atom  = atoms_cpu[idx]  
    atype = String(sys_dummy.atoms_data[idx].atom_type)  
    
    if !haskey(processed_atom_types, atype)  
        sigma_raw = FT(ustrip(atom.σ))
        if sigma_raw <= FT(1e-6) || sigma_raw == one(FT)
            sigma_raw = FT(0.15)
        end
        
        push!(theta_ref, sigma_raw)
        push!(param_names, "atom_$(atype)_σ")  
        sigma_idx = length(theta_ref)  
        
        epsilon_raw = FT(ustrip(atom.ϵ))
        if epsilon_raw <= FT(1e-6)
            epsilon_raw = FT(1e-4)
        end
        
        push!(theta_ref, epsilon_raw)
        push!(param_names, "atom_$(atype)_ϵ") 
        epsilon_idx = length(theta_ref)

        processed_atom_types[atype] = sigma_idx  

        if idx ∈ solute_idx
            push!(solute_param_indices, sigma_idx, epsilon_idx)
        else
            push!(solvent_param_indices, sigma_idx, epsilon_idx)
        end
    end  
end  

trainable_param_indices = optimize_solvent ? collect(eachindex(theta_ref)) : copy(solute_param_indices)
trainable_param_names = String[]
for idx in trainable_param_indices
    push!(trainable_param_names, param_names[idx])
end
trainable_position_map = Dict{Int, Int}()
for (i_local, i_global) in enumerate(trainable_param_indices)
    trainable_position_map[i_global] = i_local
end
all_trainable_local_indices = collect(eachindex(trainable_param_indices))

theta_min = zeros(FT, length(theta_ref))
theta_max = zeros(FT, length(theta_ref))
phi_0     = zeros(FT, length(theta_ref))

for i in 1:length(theta_ref)
    p_name = param_names[i]
    
    atype_str = split(p_name, "_")[2]
    is_hydrogen = startswith(lowercase(atype_str), "h")
    
    if occursin("σ", p_name)
        if is_hydrogen
            theta_min[i] = FT(0.1)
            theta_max[i] = FT(0.4)
        else
            theta_min[i] = FT(0.2)
            theta_max[i] = FT(0.5)
        end
    elseif occursin("ϵ", p_name)
        if is_hydrogen
            theta_min[i] = FT(0.0)
            theta_max[i] = FT(0.5)
        else
            theta_min[i] = FT(0.0)
            theta_max[i] = FT(1.5)
        end
    end
    
    val_ref = clamp(theta_ref[i], theta_min[i] + FT(1e-4), theta_max[i] - FT(1e-4))
    val = (theta_max[i] - val_ref) / (val_ref - theta_min[i])
    phi_0[i] = (FT(1.0) / k_sigmoid) * log(val)
end

phi_active = zeros(FT, length(theta_ref))
theta_active = map_phi_to_theta(phi_active, theta_min, theta_max, phi_0, k_sigmoid)

idxs_solv = build_index_maps(sys_dummy)  
idxs_vac  = build_index_maps(System("ethanol_vac.pdb", ff; array_type=AT, nonbonded_method=:none))

# ==============================================================================
# --- MAIN PIPELINE ORCHESTRATION ---
# ==============================================================================

for macro_epoch in 1:max_macro_epochs
    println("\n>>> STARTING MACRO EPOCH $macro_epoch <<<")  

    global active_bias_solv, active_bias_vac, phi_active, theta_active, restart_cache_solv, restart_cache_vac

    warm_start_solv = macro_epoch > 1 && !isnothing(restart_cache_solv)
    warm_start_vac  = macro_epoch > 1 && !isnothing(restart_cache_vac)

    logger_solv = AWHEnsembleLogger(T_coord, T_vol, T_en, production_log_interval)
    awh_solv, sys_solv = setup_alchemical_awh(
        "ethanol_solv.pdb",
        solute_idx;
        is_vacuum=false,
        logger=logger_solv,
        injected_bias=active_bias_solv,
        optimized_params=theta_active,
        param_idxs=idxs_solv,
        restart_state=warm_start_solv ? restart_cache_solv : nothing,
        restart_active_idx=warm_start_solv ? restart_cache_solv.active_idx : 1,
        warm_start=warm_start_solv
    )

    logger_vac = AWHEnsembleLogger(T_coord, T_vol, T_en, production_log_interval)
    awh_vac, sys_vac = setup_alchemical_awh(
        "ethanol_vac.pdb",
        solute_idx;
        is_vacuum=true,
        logger=logger_vac,
        injected_bias=active_bias_vac,
        optimized_params=theta_active,
        param_idxs=idxs_vac,
        restart_state=warm_start_vac ? restart_cache_vac : nothing,
        restart_active_idx=warm_start_vac ? restart_cache_vac.active_idx : 1,
        warm_start=warm_start_vac
    )

    if warm_start_solv
        restart_rmsd = coords_rmsd_nm(restart_cache_solv.coords, awh_solv.state.active_sys.coords)
        prev_vol = FT(ustrip(volume(restart_cache_solv.boundary)))
        new_vol = FT(ustrip(volume(awh_solv.state.active_sys.boundary)))
        idx_match = restart_cache_solv.active_idx == awh_solv.state.active_idx
        @info "Macro Start (Solvent): warm_start=true | λ_prev=$(restart_cache_solv.active_idx) | λ_new=$(awh_solv.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6)) | volume_ratio=$(round(new_vol / prev_vol, digits=6))"
        if restart_rmsd > restart_rmsd_tol_nm
            @info "  [!] Solvent restart RMSD above tolerance ($(restart_rmsd_tol_nm) nm)."
        end
    else
        @info "Macro Start (Solvent): warm_start=false (cold-start from PDB)."
    end

    if warm_start_vac
        restart_rmsd = coords_rmsd_nm(restart_cache_vac.coords, awh_vac.state.active_sys.coords)
        idx_match = restart_cache_vac.active_idx == awh_vac.state.active_idx
        @info "Macro Start (Vacuum): warm_start=true | λ_prev=$(restart_cache_vac.active_idx) | λ_new=$(awh_vac.state.active_idx) | idx_match=$idx_match | restart_rmsd_nm=$(round(restart_rmsd, digits=6))"
        if restart_rmsd > restart_rmsd_tol_nm
            @info "  [!] Vacuum restart RMSD above tolerance ($(restart_rmsd_tol_nm) nm)."
        end
    else
        @info "Macro Start (Vacuum): warm_start=false (cold-start from PDB)."
    end

    e_unit = sys_solv.energy_units  
    beta_val = FT(1.0 / ustrip(uconvert(e_unit, Unitful.R * T0)))  
    P0_energy_per_vol = FT(ustrip(uconvert(e_unit, P0 * FT(1.0)u"nm^3" * Unitful.Na)))
    
    dG_exp_physical = FT(-5.01) * FT(4.184)   
    dG_exp = dG_exp_physical * beta_val   

    awh_ready = false
    awh_readiness_reason = :not_checked
    split_gap = FT(Inf)
    split_gap_solv = FT(Inf)
    split_gap_vac = FT(Inf)
    parity_gap_solv = FT(Inf)
    parity_gap_vac = FT(Inf)

    local awh_solv_prod
    local awh_vac_prod
    local logger_solv_prod
    local logger_vac_prod
    local nbrs_solv
    local nbrs_vac
    local u_solv_ref
    local u_vac_ref

    stageA_stats_default = (
        ready = false,
        df_ready = false,
        df_mean = FT(Inf),
        linear_neff = zero(FT),
        neff_ready = false,
        round_trips = 0,
        round_trip_ready = false,
        endpoint_low = zero(FT),
        endpoint_high = zero(FT),
        endpoint_ready = false,
        n_hist = 0
    )
    stageB_stats_default = (
        ready = false,
        split_ready = false,
        split_gap = FT(Inf),
        parity_ready = false,
        parity_gap = FT(Inf),
        n_frames = 0,
        dG_half_1 = FT(NaN),
        dG_half_2 = FT(NaN)
    )

    stageA_stats_solv = stageA_stats_default
    stageA_stats_vac = stageA_stats_default
    stageB_stats_solv = stageB_stats_default
    stageB_stats_vac = stageB_stats_default

    stageA_streak_solv = 0
    stageA_streak_vac = 0
    spent_steps_solv = 0
    spent_steps_vac = 0
    leg_status_solv = :active
    leg_status_vac = :active

    active_bias_solv = extract_awh_data(awh_solv)
    active_bias_vac = extract_awh_data(awh_vac)

    if awh_solv.state.in_initial_stage
        rewarm_steps = min(md_steps_rewarm, md_steps_budget)
        println("Running Solvated AWH Leg (Initial Rewarm)...")
        simulate!(awh_solv, rewarm_steps)
        spent_steps_solv += rewarm_steps
        active_bias_solv = extract_awh_data(awh_solv)
    end

    if awh_vac.state.in_initial_stage
        rewarm_steps = min(md_steps_rewarm, md_steps_budget)
        println("Running Vacuum AWH Leg (Initial Rewarm)...")
        simulate!(awh_vac, rewarm_steps)
        spent_steps_vac += rewarm_steps
        active_bias_vac = extract_awh_data(awh_vac)
    end

    while true
        if leg_status_solv == :active
            remaining_steps_solv = md_steps_budget - spent_steps_solv
            if remaining_steps_solv <= 0
                leg_status_solv = :budget_exhausted
            else
                block_steps = min(md_steps_block, remaining_steps_solv)
                println("Running Solvated AWH Leg (Stage A Block)...")
                simulate!(awh_solv, block_steps)
                spent_steps_solv += block_steps
                active_bias_solv = extract_awh_data(awh_solv)

                stageA_stats_solv = evaluate_stage_a_readiness(
                    awh_solv, awh_convergence_tol;
                    tail_lag=awh_tail_lag,
                    min_linear_neff=awh_min_linear_neff,
                    min_round_trips=awh_min_round_trips,
                    endpoint_min_fraction=awh_endpoint_min_fraction
                )

                if stageA_stats_solv.ready
                    stageA_streak_solv += 1
                else
                    stageA_streak_solv = 0
                end

                if stageA_streak_solv >= awh_stageA_stable_blocks
                    stageB_stats_solv = run_stage_b_probe(
                        awh_solv, sys_solv, theta_active, param_names, idxs_solv,
                        beta_val, awh_split_tol_kT, awh_parity_tol_kT;
                        md_steps_probe=md_steps_probe_solv,
                        leg_name="solvent",
                        include_pv=true,
                        P0_energy_per_vol=P0_energy_per_vol
                    )
                    split_gap_solv = stageB_stats_solv.split_gap
                    parity_gap_solv = stageB_stats_solv.parity_gap
                    if stageB_stats_solv.ready
                        leg_status_solv = :ready_frozen
                        @info "Solvent leg frozen after passing Stage B (split_gap=$(round(split_gap_solv, digits=4)) kT, parity_gap=$(round(parity_gap_solv, digits=4)) kT)."
                    else
                        stageA_streak_solv = 0
                    end
                end

                if leg_status_solv == :active && spent_steps_solv >= md_steps_budget
                    leg_status_solv = :budget_exhausted
                end
            end
        end

        if leg_status_vac == :active
            remaining_steps_vac = md_steps_budget - spent_steps_vac
            if remaining_steps_vac <= 0
                leg_status_vac = :budget_exhausted
            else
                block_steps = min(md_steps_block, remaining_steps_vac)
                println("Running Vacuum AWH Leg (Stage A Block)...")
                simulate!(awh_vac, block_steps)
                spent_steps_vac += block_steps
                active_bias_vac = extract_awh_data(awh_vac)

                stageA_stats_vac = evaluate_stage_a_readiness(
                    awh_vac, awh_convergence_tol;
                    tail_lag=awh_tail_lag,
                    min_linear_neff=awh_min_linear_neff,
                    min_round_trips=awh_min_round_trips,
                    endpoint_min_fraction=awh_endpoint_min_fraction
                )

                if stageA_stats_vac.ready
                    stageA_streak_vac += 1
                else
                    stageA_streak_vac = 0
                end

                if stageA_streak_vac >= awh_stageA_stable_blocks
                    stageB_stats_vac = run_stage_b_probe(
                        awh_vac, sys_vac, theta_active, param_names, idxs_vac,
                        beta_val, awh_split_tol_kT, awh_parity_tol_kT;
                        md_steps_probe=md_steps_probe_vac,
                        leg_name="vacuum",
                        include_pv=false
                    )
                    split_gap_vac = stageB_stats_vac.split_gap
                    parity_gap_vac = stageB_stats_vac.parity_gap
                    if stageB_stats_vac.ready
                        leg_status_vac = :ready_frozen
                        @info "Vacuum leg frozen after passing Stage B (split_gap=$(round(split_gap_vac, digits=4)) kT, parity_gap=$(round(parity_gap_vac, digits=4)) kT)."
                    else
                        stageA_streak_vac = 0
                    end
                end

                if leg_status_vac == :active && spent_steps_vac >= md_steps_budget
                    leg_status_vac = :budget_exhausted
                end
            end
        end

        @info "AWH Block Status: solv=$(leg_status_solv) vac=$(leg_status_vac) | spent_ns_solv=$(round(steps_to_ns(spent_steps_solv), digits=2))/$(round(ustrip(u"ns", awh_budget_time), digits=2)) | spent_ns_vac=$(round(steps_to_ns(spent_steps_vac), digits=2))/$(round(ustrip(u"ns", awh_budget_time), digits=2))"
        @info "  Stage A (Solv): df=$(round(stageA_stats_solv.df_mean, digits=6)) (ok=$(stageA_stats_solv.df_ready)) | neff=$(round(stageA_stats_solv.linear_neff, digits=1)) (ok=$(stageA_stats_solv.neff_ready)) | rt=$(stageA_stats_solv.round_trips) (ok=$(stageA_stats_solv.round_trip_ready)) | endpt=($(round(stageA_stats_solv.endpoint_low, digits=3)), $(round(stageA_stats_solv.endpoint_high, digits=3))) (ok=$(stageA_stats_solv.endpoint_ready)) | streak=$stageA_streak_solv"
        @info "  Stage A (Vac):  df=$(round(stageA_stats_vac.df_mean, digits=6)) (ok=$(stageA_stats_vac.df_ready)) | neff=$(round(stageA_stats_vac.linear_neff, digits=1)) (ok=$(stageA_stats_vac.neff_ready)) | rt=$(stageA_stats_vac.round_trips) (ok=$(stageA_stats_vac.round_trip_ready)) | endpt=($(round(stageA_stats_vac.endpoint_low, digits=3)), $(round(stageA_stats_vac.endpoint_high, digits=3))) (ok=$(stageA_stats_vac.endpoint_ready)) | streak=$stageA_streak_vac"
        if stageB_stats_solv.n_frames > 0
            @info "  Stage B (Solv): split_gap=$(round(stageB_stats_solv.split_gap, digits=4)) kT (ok=$(stageB_stats_solv.split_ready)) | parity_gap=$(round(stageB_stats_solv.parity_gap, digits=4)) kT (ok=$(stageB_stats_solv.parity_ready)) | frames=$(stageB_stats_solv.n_frames)"
        end
        if stageB_stats_vac.n_frames > 0
            @info "  Stage B (Vac):  split_gap=$(round(stageB_stats_vac.split_gap, digits=4)) kT (ok=$(stageB_stats_vac.split_ready)) | parity_gap=$(round(stageB_stats_vac.parity_gap, digits=4)) kT (ok=$(stageB_stats_vac.parity_ready)) | frames=$(stageB_stats_vac.n_frames)"
        end

        if leg_status_solv == :ready_frozen && leg_status_vac == :ready_frozen
            awh_ready = true
            awh_readiness_reason = :ready
            break
        end

        if leg_status_solv == :budget_exhausted || leg_status_vac == :budget_exhausted
            awh_ready = false
            awh_readiness_reason = :awh_not_ready_budget
            break
        end
    end

    if !awh_ready
        @info "Macro $macro_epoch skipped: strict AWH readiness failed ($awh_readiness_reason). Status=(solv=$leg_status_solv, vac=$leg_status_vac), StageA df=(solv=$(round(stageA_stats_solv.df_mean, digits=6)), vac=$(round(stageA_stats_vac.df_mean, digits=6))), neff=(solv=$(round(stageA_stats_solv.linear_neff, digits=1)), vac=$(round(stageA_stats_vac.linear_neff, digits=1))), rt=(solv=$(stageA_stats_solv.round_trips), vac=$(stageA_stats_vac.round_trips)), parity=(solv=$(round(parity_gap_solv, digits=4)), vac=$(round(parity_gap_vac, digits=4))) kT"
        continue
    end

    split_gap = max(split_gap_solv, split_gap_vac)
    active_bias_solv = extract_awh_data(awh_solv)
    active_bias_vac = extract_awh_data(awh_vac)

    println("Running Production Runs (Extended Ensemble) [Readiness-Passed Dataset]...")
    clear_awh_logger_histories!(awh_solv)
    clear_awh_logger_histories!(awh_vac)

    awh_solv_prod = AWHSimulation(awh_solv.state; num_md_steps=awh_solv.n_md_steps, update_freq=typemax(Int), well_tempered_factor=Inf)
    awh_solv_prod.state.active_sys.loggers.awh_logger.should_log = true
    simulate!(awh_solv_prod, md_steps_prod)

    awh_vac_prod = AWHSimulation(awh_vac.state; num_md_steps=awh_vac.n_md_steps, update_freq=typemax(Int), well_tempered_factor=Inf)
    awh_vac_prod.state.active_sys.loggers.awh_logger.should_log = true
    simulate!(awh_vac_prod, md_steps_prod)

    restart_cache_solv = capture_restart_state(awh_solv_prod)
    restart_cache_vac  = capture_restart_state(awh_vac_prod)
    @info "Captured restart state for next macro epoch: λ_solv=$(restart_cache_solv.active_idx) | λ_vac=$(restart_cache_vac.active_idx)"

    logger_solv_prod = get_production_logger(awh_solv_prod, "solvent")
    logger_vac_prod  = get_production_logger(awh_vac_prod, "vacuum")

    nbrs_solv = precompute_neighbors(logger_solv_prod, awh_solv_prod.state.active_sys)
    nbrs_vac  = precompute_neighbors(logger_vac_prod, awh_vac_prod.state.active_sys)

    u_solv_ref, _ = evaluate_ensemble(
        logger_solv_prod, nbrs_solv, awh_solv_prod, sys_solv,
        theta_active, param_names, idxs_solv...; compute_gradients=false
    )
    u_vac_ref, _  = evaluate_ensemble(
        logger_vac_prod, nbrs_vac, awh_vac_prod, sys_vac,
        theta_active, param_names, idxs_vac...; compute_gradients=false
    )
    GC.gc()

    println("FIRST GRADS")

    M_solv_FT = FT(length(logger_solv_prod.active_idx_history))
    M_vac_FT  = FT(length(logger_vac_prod.active_idx_history))
    
    ess_threshold_solv = M_solv_FT * ess_threshold_ratio
    ess_threshold_vac  = M_vac_FT * ess_threshold_ratio
 
    N_base_solv = FT(awh_solv.initial_sampl_n + awh_solv.state.N_eff)  
    N_base_vac  = FT(awh_vac.initial_sampl_n + awh_vac.state.N_eff)  
    
    tiny_alpha_hits = 0
    phase2_exit_reason = :running
    macro_start_residual = FT(NaN)
    macro_end_residual = FT(NaN)
    best_macro_abs_residual = FT(Inf)
    best_macro_residual = FT(NaN)
    best_macro_epoch = 0
    phi_best_macro = copy(phi_active)
    theta_best_macro = copy(theta_active)
    solvent_theta_start = isempty(solvent_param_indices) ? FT[] : copy(theta_active[solvent_param_indices])
    
    inner_epoch = 1  
    while true  
        if optimize_solvent
            active_global_indices = (inner_epoch % 2 != 0) ? solute_param_indices : solvent_param_indices
            block_name = (inner_epoch % 2 != 0) ? "Solute" : "Solvent"
        else
            active_global_indices = trainable_param_indices
            block_name = "Solute"
        end
        active_trainable_indices = Int[]
        for idx_global in active_global_indices
            if haskey(trainable_position_map, idx_global)
                push!(active_trainable_indices, trainable_position_map[idx_global])
            end
        end
        @info "  >> Alternating Optimization: Active Block = $block_name"

        if isempty(active_trainable_indices)
            @info "  [!] No parameters active for this block. Skipping."
            inner_epoch += 1
            continue
        end

        u_solv_eval, grads_solv_eval_theta = evaluate_ensemble(logger_solv_prod, nbrs_solv, awh_solv_prod, sys_solv,   
                                                               theta_active, param_names, idxs_solv...; compute_gradients=true)  
        u_vac_eval, grads_vac_eval_theta   = evaluate_ensemble(logger_vac_prod, nbrs_vac, awh_vac_prod, sys_vac,   
                                                               theta_active, param_names, idxs_vac...; compute_gradients=true)  

        grads_solv_eval_phi = Dict{String, Matrix{FT}}()
        grads_vac_eval_phi  = Dict{String, Matrix{FT}}()
        
        chain_rule_multiplier = get_chain_rule_multiplier(theta_active, theta_min, theta_max, k_sigmoid)
        
        for (i, p_key) in enumerate(param_names)
            grads_solv_eval_phi[p_key] = grads_solv_eval_theta[p_key] .* chain_rule_multiplier[i]
            grads_vac_eval_phi[p_key]  = grads_vac_eval_theta[p_key] .* chain_rule_multiplier[i]
        end

        volumes_solv = FT.(ustrip.(logger_solv_prod.volume_history))

        w_norm_solv, ess_solv = compute_weights_and_ess(u_solv_eval, u_solv_ref, logger_solv_prod.active_idx_history, beta_val, volumes_solv, P0_energy_per_vol)  
        w_norm_vac, ess_vac   = compute_weights_and_ess(u_vac_eval, u_vac_ref, logger_vac_prod.active_idx_history, beta_val)  

        N_active_solv = N_base_solv * (ess_solv / M_solv_FT)  
        N_active_vac  = N_base_vac  * (ess_vac / M_vac_FT)  

        if ess_solv < ess_threshold_solv || ess_vac < ess_threshold_vac  
            println("  [!] ESS threshold broken during state evaluation. Exiting Phase 2.")  
            phase2_exit_reason = :ess_threshold
            break  
        end  

        s_mean_solv, fim_solv = compute_empirical_gradients_and_fim(trainable_param_names, grads_solv_eval_phi, w_norm_solv, logger_solv_prod.active_idx_history, beta_val)
        s_mean_vac, fim_vac    = compute_empirical_gradients_and_fim(trainable_param_names, grads_vac_eval_phi, w_norm_vac, logger_vac_prod.active_idx_history, beta_val)

        bias_solv = active_bias_solv.f .+ active_bias_solv.log_rho
        bias_vac  = active_bias_vac.f  .+ active_bias_vac.log_rho

        grad_F_solv_1, F_solv_1_current = compute_global_endpoint_gradients(  
            trainable_param_names, grads_solv_eval_phi, u_solv_eval, u_solv_ref, logger_solv_prod.active_idx_history, 1, beta_val, bias_solv, volumes_solv, P0_energy_per_vol
        )  
        grad_F_solv_0, F_solv_0_current = compute_global_endpoint_gradients(  
            trainable_param_names, grads_solv_eval_phi, u_solv_eval, u_solv_ref, logger_solv_prod.active_idx_history, num_lambda_states, beta_val, bias_solv, volumes_solv, P0_energy_per_vol
        )  
        grad_F_vac_1, F_vac_1_current = compute_global_endpoint_gradients(  
            trainable_param_names, grads_vac_eval_phi, u_vac_eval, u_vac_ref, logger_vac_prod.active_idx_history, 1, beta_val, bias_vac
        )  
        grad_F_vac_0, F_vac_0_current = compute_global_endpoint_gradients(  
            trainable_param_names, grads_vac_eval_phi, u_vac_eval, u_vac_ref, logger_vac_prod.active_idx_history, num_lambda_states, beta_val, bias_vac
        )  

        grad_dG_solv = grad_F_solv_1 .- grad_F_solv_0  
        grad_dG_vac  = grad_F_vac_1  .- grad_F_vac_0  

        # Directly compute the free energy differences from the exact MBAR estimator
        dG_solv_current  = F_solv_1_current - F_solv_0_current  
        dG_vac_current   = F_vac_1_current - F_vac_0_current  

        dG_pred = dG_solv_current - dG_vac_current + dG_std_corr  
        error_residual = dG_pred - dG_exp  
        if isnan(macro_start_residual)
            macro_start_residual = error_residual
        end

        dL_dE = abs(error_residual) <= huber_delta ? error_residual : huber_delta * sign(error_residual)  
        grad_loss = dL_dE .* (grad_dG_solv .- grad_dG_vac)
        fim_joint = fim_solv .+ fim_vac

        # --- ISOLATE ACTIVE BLOCK MATRICES ---
        grad_loss_active = grad_loss[active_trainable_indices]
        fim_active = fim_joint[active_trainable_indices, active_trainable_indices]

        # --- Thresholded Diagonal Preconditioning ---
        # --- Thresholded Diagonal Preconditioning ---
        fim_diag = diag(fim_active)
        
        # Isolate parameters with negligible variance to prevent 1/d explosion
        variance_threshold = maximum(fim_diag) * FT(1e-5)
        D_vec = [d > variance_threshold ? FT(1.0) / sqrt(d) : zero(FT) for d in fim_diag]
        D_mat = Diagonal(D_vec)

        fim_corr = D_mat * fim_active * D_mat
        grad_loss_scaled = D_vec .* grad_loss_active

        # Exact Moore-Penrose Pseudo-Inverse (Preserves Natural Gradient Metric)
        decomp = eigen(Symmetric(fim_corr))
        vals, vecs = decomp.values, decomp.vectors
        
        eigenvalue_tol = maximum(vals) * eigenvalue_tol_scale
        inv_vals = [v > eigenvalue_tol ? 1.0/v : zero(FT) for v in vals]
        fim_corr_inv = vecs * Diagonal(inv_vals) * transpose(vecs)
        
        base_step_scaled = fim_corr_inv * grad_loss_scaled
        base_step_active = D_vec .* base_step_scaled

        estimated_KL = 0.5 * dot(base_step_active, fim_active * base_step_active)
        if estimated_KL > kl_target
            kl_scaling = sqrt(kl_target / estimated_KL)
            update_direction_active = base_step_active * kl_scaling
        else
            kl_scaling = 1.0
            update_direction_active = base_step_active
        end

        n_truncated = count(v -> v <= eigenvalue_tol, vals)
        fim_cond_raw = cond(fim_corr)

        # --- Infinity-Norm Trust Region ---
        max_phi_update = maximum(abs.(update_direction_active))
        
        # Solute can step large; bulk solvent must be tightly constrained
        max_allowed_phi_step = (block_name == "Solute") ? max_phi_step_solute : max_phi_step_solvent
        
        if max_phi_update > max_allowed_phi_step
            clip_scaling = max_allowed_phi_step / max_phi_update
            update_direction_active .*= clip_scaling
            @info "  [!] Step clipped by infinity-norm (Scaling: $(round(clip_scaling, digits=4)))"
        end

        # Map back to full parameter space to evaluate step
        update_direction_train = zeros(FT, length(trainable_param_indices))
        update_direction_train[active_trainable_indices] = update_direction_active
        update_direction = zeros(FT, length(param_names))
        for (i_local, i_global) in enumerate(trainable_param_indices)
            update_direction[i_global] = update_direction_train[i_local]
        end

        # --- Backtracking Line Search ---
        alpha = FT(1.0)
        phi_prop = copy(phi_active)
        theta_prop = copy(theta_active)
        line_search_success = false
        
        ess_solv_prop = zero(FT)
        ess_vac_prop  = zero(FT)
        accepted_residual = error_residual
        
        for ls_iter in 1:7
            phi_prop .= phi_active .- alpha .* update_direction
            theta_prop .= map_phi_to_theta(phi_prop, theta_min, theta_max, phi_0, k_sigmoid)

            # Evaluate proposed ensembles (No gradients needed)
            u_solv_prop, _ = evaluate_ensemble(logger_solv_prod, nbrs_solv, awh_solv_prod, sys_solv, theta_prop, param_names, idxs_solv...; compute_gradients=false)
            u_vac_prop, _ = evaluate_ensemble(logger_vac_prod, nbrs_vac, awh_vac_prod, sys_vac, theta_prop, param_names, idxs_vac...; compute_gradients=false)

            _, ess_solv_prop = compute_weights_and_ess(u_solv_prop, u_solv_ref, logger_solv_prod.active_idx_history, beta_val, volumes_solv, P0_energy_per_vol)
            _, ess_vac_prop = compute_weights_and_ess(u_vac_prop, u_vac_ref, logger_vac_prod.active_idx_history, beta_val)

            # --- NEW: Evaluate Objective Descent ---
            _, F_solv_1_prop = compute_global_endpoint_gradients(trainable_param_names, grads_solv_eval_phi, u_solv_prop, u_solv_ref, logger_solv_prod.active_idx_history, 1, beta_val, bias_solv, volumes_solv, P0_energy_per_vol; compute_gradients=false)
            _, F_solv_0_prop = compute_global_endpoint_gradients(trainable_param_names, grads_solv_eval_phi, u_solv_prop, u_solv_ref, logger_solv_prod.active_idx_history, num_lambda_states, beta_val, bias_solv, volumes_solv, P0_energy_per_vol; compute_gradients=false)
            _, F_vac_1_prop = compute_global_endpoint_gradients(trainable_param_names, grads_vac_eval_phi, u_vac_prop, u_vac_ref, logger_vac_prod.active_idx_history, 1, beta_val, bias_vac; compute_gradients=false)
            _, F_vac_0_prop = compute_global_endpoint_gradients(trainable_param_names, grads_vac_eval_phi, u_vac_prop, u_vac_ref, logger_vac_prod.active_idx_history, num_lambda_states, beta_val, bias_vac; compute_gradients=false)

            # CORRECTED: Solv - Vac
            dG_pred_prop = (F_solv_1_prop - F_solv_0_prop) - (F_vac_1_prop - F_vac_0_prop) + dG_std_corr
            error_residual_prop = dG_pred_prop - dG_exp

            @info "    LS Iter $ls_iter (α=$(alpha)): Solv ESS = $(round(ess_solv_prop, digits=1)) | Vac ESS = $(round(ess_vac_prop, digits=1)) | Res = $(round(error_residual_prop, digits=3))"

            # Allow 5% relative noise in the descent condition due to offline MBAR estimator variance
            noise_tolerance = FT(0.05) * abs(error_residual)

            # Enforce both ESS bounds AND noise-tolerant objective descent
            if ess_solv_prop >= ess_threshold_solv && ess_vac_prop >= ess_threshold_vac && abs(error_residual_prop) <= abs(error_residual) + noise_tolerance
                line_search_success = true
                phi_active .= phi_prop
                theta_active .= theta_prop
                accepted_residual = error_residual_prop
                @info "    -> Line search converged."
                break
            else
                alpha *= FT(0.5)
            end
        end
        
        if alpha <= tiny_alpha_cutoff
            tiny_alpha_hits += 1
        else
            tiny_alpha_hits = 0
        end
        
        macro_end_residual = accepted_residual
        if abs(accepted_residual) < best_macro_abs_residual
            best_macro_abs_residual = abs(accepted_residual)
            best_macro_residual = accepted_residual
            best_macro_epoch = inner_epoch
            phi_best_macro .= phi_active
            theta_best_macro .= theta_active
        end

        # ======================================================================
        # METRICS LOGGING
        # ======================================================================
        norm_grad_loss = norm(grad_loss_active)
        max_grad_loss  = maximum(abs.(grad_loss_active))
        actual_max_phi_step = maximum(abs.(update_direction_active)) * alpha
        
        @info "--- Current Parameter State ---"
        for i in 1:length(param_names)
            @info "  $(param_names[i]): $(round(theta_active[i], digits=6))"
        end
        
        @info "--- Optimization Metrics (Epoch $inner_epoch - Block: $block_name) ---"  
        @info "  Prediction:  ∆G_pred = $(round(dG_pred, digits=3)) kT | Target = $(round(dG_exp, digits=3)) kT"  
        @info "  Error:       Residual = $(round(error_residual, digits=3)) | Huber dL/dE = $(round(dL_dE, digits=3))"  
        @info "  Gradients:   Norm = $(round(norm_grad_loss, digits=5)) | Max = $(round(max_grad_loss, digits=5))"  
        @info "  FIM (Corr):  Raw Cond Number = $(round(fim_cond_raw, digits=2)) | Truncated Eigs = $n_truncated / $(length(vals))"  
        @info "  Effective Samples: Solv = $(round(N_active_solv, digits=1)) | Vac = $(round(N_active_vac, digits=1))"
        @info "  KL Bound:    Est. KL = $(round(estimated_KL, digits=4)) | Target = $kl_target | Scaling = $(round(kl_scaling, digits=4))"
        @info "  Line Search: Converged α = $alpha | Final Solv ESS = $(round(ess_solv_prop, digits=1)) | Vac ESS = $(round(ess_vac_prop, digits=1))"
        @info "  Actual Step: Max ϕ ∆ = $(round(actual_max_phi_step, digits=6)) (α=$alpha)"  
        @info "  Params (σ,ϵ):Min = $(round(minimum(theta_active[active_global_indices]), digits=5)) | Max = $(round(maximum(theta_active[active_global_indices]), digits=5))"  
        println("---------------------------------------------------\n")

        if tiny_alpha_hits >= max_tiny_alpha_hits
            @info "  [!] Repeated tiny line-search α detected ($tiny_alpha_hits consecutive epochs with α <= $tiny_alpha_cutoff). Triggering Phase 3 resimulation."
            phase2_exit_reason = :tiny_alpha
            break
        end

        if !line_search_success || actual_max_phi_step < min_phi_step
            @info "  [!] Line search failed or step vanished. Triggering Phase 3 resimulation."
            phase2_exit_reason = :step_vanish
            break
        end

        inner_epoch += 1
    end

    if (phase2_exit_reason == :step_vanish || phase2_exit_reason == :tiny_alpha) && best_macro_epoch > 0
        phi_active .= phi_best_macro
        theta_active .= theta_best_macro
        macro_end_residual = best_macro_residual
        @info "  [!] Restored best inner-loop state from Epoch $best_macro_epoch (Residual = $(round(best_macro_residual, digits=3)))."
    end

    if !optimize_solvent && !isempty(solvent_param_indices)
        solvent_drift = maximum(abs.(theta_active[solvent_param_indices] .- solvent_theta_start))
        @info "Solvent Invariant: max |Δθ_solvent| = $(round(solvent_drift, digits=8))"
    end
    
    @info "Macro $macro_epoch Residual Summary: Start = $(round(macro_start_residual, digits=3)) | Best = $(round(best_macro_residual, digits=3)) (Epoch $best_macro_epoch) | End = $(round(macro_end_residual, digits=3)) | Exit = $phase2_exit_reason | AWH split-gap(max) = $(round(split_gap, digits=4)) kT | Parity gaps = (solv=$(round(parity_gap_solv, digits=4)), vac=$(round(parity_gap_vac, digits=4))) kT"

end
