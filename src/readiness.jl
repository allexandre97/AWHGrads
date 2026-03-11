# Readiness and probe helpers extracted from main_alch.jl

function phase_timing_metadata(
    phase::AbstractString,
    leg_name::AbstractString;
    md_steps::Union{Nothing, Int}=nothing,
    wall_s::Union{Nothing, Real}=nothing,
)
    md_ns = isnothing(md_steps) ? nothing : steps_to_ns(md_steps)
    wall_s_val = isnothing(wall_s) ? nothing : Float64(wall_s)
    steps_per_s = if isnothing(md_steps) || isnothing(wall_s_val) || wall_s_val <= 0.0
        nothing
    else
        Float64(md_steps) / wall_s_val
    end
    return (
        phase = String(phase),
        leg = String(leg_name),
        md_steps = md_steps,
        md_ns = md_ns,
        wall_s = wall_s_val,
        steps_per_s = steps_per_s,
    )
end


function timed_phase(
    phase::AbstractString,
    leg_name::AbstractString,
    op::Function;
    md_steps::Union{Nothing, Int}=nothing,
)
    start_meta = phase_timing_metadata(phase, leg_name; md_steps=md_steps)
    if isnothing(start_meta.md_steps)
        @info "AWH Timing Start: phase=$(start_meta.phase) | leg=$(start_meta.leg)"
    else
        @info "AWH Timing Start: phase=$(start_meta.phase) | leg=$(start_meta.leg) | md_steps=$(start_meta.md_steps) | md_ns=$(round(start_meta.md_ns, digits=4))"
    end

    t0_ns = time_ns()
    result = op()
    elapsed_s = (time_ns() - t0_ns) / 1e9

    end_meta = phase_timing_metadata(phase, leg_name; md_steps=md_steps, wall_s=elapsed_s)
    if isnothing(end_meta.md_steps)
        @info "AWH Timing End: phase=$(end_meta.phase) | leg=$(end_meta.leg) | wall_s=$(round(end_meta.wall_s, digits=3))"
    else
        @info "AWH Timing End: phase=$(end_meta.phase) | leg=$(end_meta.leg) | md_steps=$(end_meta.md_steps) | md_ns=$(round(end_meta.md_ns, digits=4)) | wall_s=$(round(end_meta.wall_s, digits=3)) | steps_per_s=$(round(end_meta.steps_per_s, digits=2))"
    end

    return (result=result, timing=end_meta)
end


function timed_phase(
    op::Function,
    phase::AbstractString,
    leg_name::AbstractString;
    md_steps::Union{Nothing, Int}=nothing,
)
    return timed_phase(phase, leg_name, op; md_steps=md_steps)
end


function awh_linear_stage_stats(awh_sim::AWHSimulation, tol::FT; max_lag::Int=10)
    stats = awh_sim.state.stats
    linear_changes = FT[]
    for i in eachindex(stats.stage_history)
        if stats.stage_history[i] == :linear
            push!(linear_changes, FT(stats.max_delta_f_history[i]))
        end
    end
    linear_updates = length(linear_changes)
    linear_neff = awh_sim.state.in_initial_stage ? zero(FT) : FT(awh_sim.state.N_eff)
    if linear_updates == 0
        return false, FT(Inf), linear_neff
    end
    first_idx = max(1, linear_updates - max_lag + 1)
    tail_changes = linear_changes[first_idx:end]
    nonzero_changes = filter(x -> x != zero(FT), tail_changes)
    mean_change = isempty(nonzero_changes) ? zero(FT) : sum(nonzero_changes) / FT(length(nonzero_changes))
    return mean_change <= tol, mean_change, linear_neff
end

function estimate_leg_dg_from_reference(
    energies::Matrix{FT},
    active_lambda_idx::Vector{Int},
    awh_bias::Vector{FT},
    num_lambda_states::Int,
    beta::FT;
    volumes::Vector{FT}=FT[],
    P0_energy_per_vol::FT=zero(FT)
) where {FT <: AbstractFloat}
    dummy_names = String[]
    dummy_grads = Dict{String, Matrix{FT}}()
    _, F_1 = compute_global_endpoint_gradients(
        dummy_names, dummy_grads, energies, energies, active_lambda_idx, 1,
        beta, awh_bias, volumes, P0_energy_per_vol; compute_gradients=false
    )
    _, F_0 = compute_global_endpoint_gradients(
        dummy_names, dummy_grads, energies, energies, active_lambda_idx, num_lambda_states,
        beta, awh_bias, volumes, P0_energy_per_vol; compute_gradients=false
    )
    return F_1 - F_0
end


function split_half_ranges(n::Int)
    n_first = fld(n, 2)
    return (1:n_first, (n_first + 1):n)
end


function count_full_round_trips(active_idx_history::Vector{Int}, low_idx::Int, high_idx::Int)
    if isempty(active_idx_history)
        return 0
    end
    phase = 0
    n_round_trips = 0
    for idx in active_idx_history
        if phase == 0
            if idx == low_idx
                phase = 1
            end
        elseif phase == 1
            if idx == high_idx
                phase = 2
            end
        else
            if idx == low_idx
                n_round_trips += 1
                phase = 1
            end
        end
    end
    return n_round_trips
end


function endpoint_occupancy_fractions(active_idx_history::Vector{Int}, low_idx::Int, high_idx::Int)
    n_frames = length(active_idx_history)
    if n_frames == 0
        return zero(FT), zero(FT)
    end
    low_frac = FT(count(==(low_idx), active_idx_history)) / FT(n_frames)
    high_frac = FT(count(==(high_idx), active_idx_history)) / FT(n_frames)
    return low_frac, high_frac
end


function evaluate_stage_a_readiness(
    awh_sim::AWHSimulation,
    awh_tol::FT;
    tail_lag::Int,
    min_linear_neff::Int,
    min_round_trips::Int,
    endpoint_min_fraction::FT,
    low_idx::Int=1,
    high_idx::Int
) where {FT <: AbstractFloat}
    df_ready, df_mean, linear_neff = awh_linear_stage_stats(awh_sim, awh_tol; max_lag=tail_lag)
    neff_ready = !awh_sim.state.in_initial_stage && linear_neff >= FT(min_linear_neff)

    idx_history = get_awh_active_idx_history(awh_sim)
    round_trips = count_full_round_trips(idx_history, low_idx, high_idx)
    round_trip_ready = round_trips >= min_round_trips

    endpoint_low, endpoint_high = endpoint_occupancy_fractions(idx_history, low_idx, high_idx)
    endpoint_ready = endpoint_low >= endpoint_min_fraction && endpoint_high >= endpoint_min_fraction

    ready = df_ready && neff_ready && round_trip_ready && endpoint_ready
    return (
        ready = ready,
        df_ready = df_ready,
        df_mean = df_mean,
        linear_neff = linear_neff,
        neff_ready = neff_ready,
        round_trips = round_trips,
        round_trip_ready = round_trip_ready,
        endpoint_low = endpoint_low,
        endpoint_high = endpoint_high,
        endpoint_ready = endpoint_ready,
        n_hist = length(idx_history)
    )
end


function run_stage_b_probe(
    awh_sim::AWHSimulation,
    sys_base,
    theta_params::Vector{FT},
    param_names::Vector{String},
    idxs,
    num_lambda_states::Int,
    beta::FT,
    awh_split_tol_kT::FT,
    awh_parity_tol_kT::FT;
    md_steps_probe::Int,
    leg_name::String,
    include_pv::Bool=false,
    P0_energy_per_vol::FT=zero(FT)
) where {FT <: AbstractFloat}
    if md_steps_probe <= 0
        @info "Stage B ($(leg_name)) skipped: md_steps_probe <= 0 (md_steps_probe=$md_steps_probe)."
        return (
            ready = false,
            split_ready = false,
            split_gap = FT(Inf),
            parity_ready = false,
            parity_gap = FT(Inf),
            n_frames = 0,
            dG_half_1 = FT(NaN),
            dG_half_2 = FT(NaN)
        )
    end

    probe_sim = AWHSimulation(
        deepcopy(awh_sim.state);
        num_md_steps=awh_sim.n_md_steps,
        update_freq=typemax(Int),
        well_tempered_factor=Inf,
        coverage_type=:physical
    )
    clear_awh_logger_histories!(probe_sim)
    probe_sim.state.active_sys.loggers.awh_logger.should_log = true
    probe_md_timed = timed_phase("Stage B Probe MD", leg_name; md_steps=md_steps_probe) do
        simulate!(probe_sim, md_steps_probe)
    end

    logger_probe = get_production_logger(probe_sim, "$leg_name probe")
    n_frames = length(logger_probe.active_idx_history)
    if n_frames < 2
        @info "Stage B ($(leg_name)) early exit: insufficient probe frames (n_frames=$n_frames, required=2) | probe_md_wall_s=$(round(probe_md_timed.timing.wall_s, digits=3)) | probe_md_steps_per_s=$(round(probe_md_timed.timing.steps_per_s, digits=2))"
        return (
            ready = false,
            split_ready = false,
            split_gap = FT(Inf),
            parity_ready = false,
            parity_gap = FT(Inf),
            n_frames = n_frames,
            dG_half_1 = FT(NaN),
            dG_half_2 = FT(NaN)
        )
    end

    nbrs_probe_timed = timed_phase("Stage B Neighbor Precompute", leg_name) do
        precompute_neighbors(logger_probe, probe_sim.state.active_sys)
    end
    nbrs_probe = nbrs_probe_timed.result

    ensemble_eval_timed = timed_phase("Stage B Ensemble Eval", leg_name) do
        evaluate_ensemble(
            logger_probe, nbrs_probe, probe_sim, sys_base,
            theta_params, param_names, idxs...; compute_gradients=false
        )
    end
    u_probe_ref, _ = ensemble_eval_timed.result

    bias_data = extract_awh_data(awh_sim)
    awh_bias = bias_data.f .+ bias_data.log_rho

    half_1, half_2 = split_half_ranges(n_frames)
    if isempty(half_1) || isempty(half_2)
        @info "Stage B ($(leg_name)) early exit: split ranges are empty (n_frames=$n_frames) | probe_md_wall_s=$(round(probe_md_timed.timing.wall_s, digits=3)) | nbrs_wall_s=$(round(nbrs_probe_timed.timing.wall_s, digits=3)) | eval_wall_s=$(round(ensemble_eval_timed.timing.wall_s, digits=3))"
        return (
            ready = false,
            split_ready = false,
            split_gap = FT(Inf),
            parity_ready = false,
            parity_gap = FT(Inf),
            n_frames = n_frames,
            dG_half_1 = FT(NaN),
            dG_half_2 = FT(NaN)
        )
    end

    split_parity_timed = timed_phase("Stage B Split-Parity", leg_name) do
        volumes_probe = include_pv ? FT.(ustrip.(logger_probe.volume_history)) : FT[]
        dG_half_1 = estimate_leg_dg_from_reference(
            u_probe_ref[half_1, :],
            logger_probe.active_idx_history[half_1],
            awh_bias,
            num_lambda_states,
            beta;
            volumes=include_pv ? volumes_probe[half_1] : FT[],
            P0_energy_per_vol=include_pv ? P0_energy_per_vol : zero(FT)
        )
        dG_half_2 = estimate_leg_dg_from_reference(
            u_probe_ref[half_2, :],
            logger_probe.active_idx_history[half_2],
            awh_bias,
            num_lambda_states,
            beta;
            volumes=include_pv ? volumes_probe[half_2] : FT[],
            P0_energy_per_vol=include_pv ? P0_energy_per_vol : zero(FT)
        )

        split_gap = abs(dG_half_1 - dG_half_2)
        split_ready = split_gap <= awh_split_tol_kT

        F_mbar_profile = include_pv ?
            compute_full_mbar_profile(u_probe_ref, u_probe_ref, awh_bias, beta; volumes=volumes_probe, P0_energy_per_vol=P0_energy_per_vol) :
            compute_full_mbar_profile(u_probe_ref, u_probe_ref, awh_bias, beta)
        parity_gap = compute_parity_gap(F_mbar_profile, awh_bias; ref_idx=1)
        parity_ready = parity_gap <= awh_parity_tol_kT

        return (
            dG_half_1=dG_half_1,
            dG_half_2=dG_half_2,
            split_gap=split_gap,
            split_ready=split_ready,
            parity_gap=parity_gap,
            parity_ready=parity_ready,
        )
    end

    split_parity = split_parity_timed.result
    dG_half_1 = split_parity.dG_half_1
    dG_half_2 = split_parity.dG_half_2
    split_gap = split_parity.split_gap
    split_ready = split_parity.split_ready
    parity_gap = split_parity.parity_gap
    parity_ready = split_parity.parity_ready

    return (
        ready = split_ready && parity_ready,
        split_ready = split_ready,
        split_gap = split_gap,
        parity_ready = parity_ready,
        parity_gap = parity_gap,
        n_frames = n_frames,
        dG_half_1 = dG_half_1,
        dG_half_2 = dG_half_2
    )
end
