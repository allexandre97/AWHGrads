# Readiness and probe helpers extracted from main_alch.jl

"""
    phase_timing_metadata(phase, leg_name; md_steps=nothing, wall_s=nothing)

Build a small timing record for log messages emitted around a simulation phase.
"""
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


"""
    timed_phase(phase, leg_name, op; md_steps=nothing)

Run `op()` while emitting standardized start/end timing logs. The return value is
the named tuple `(result, timing)`.
"""
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


"""
    timed_phase(op, phase, leg_name; md_steps=nothing)

Convenience argument order for `timed_phase`.
"""
function timed_phase(
    op::Function,
    phase::AbstractString,
    leg_name::AbstractString;
    md_steps::Union{Nothing, Int}=nothing,
)
    return timed_phase(phase, leg_name, op; md_steps=md_steps)
end


"""
    awh_linear_stage_stats(awh_sim, tol; max_lag=10)

Summarize the recent linear-stage bias changes and return
`(df_ready, mean_change, linear_neff)`.
"""
function awh_linear_stage_stats(awh_sim, tol::FT; max_lag::Int=10)
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


"""
    estimate_lambda_history_ess(active_lambda_idx, FT=Float64)

Estimate the effective sample size of the visited λ-index history using a
positive-sequence integrated autocorrelation-time estimator.
"""
function estimate_lambda_history_ess(active_lambda_idx::AbstractVector{<:Real}, ::Type{FT}=Float64) where {FT <: AbstractFloat}
    n = length(active_lambda_idx)
    if n <= 1
        return one(FT)
    end

    centered = FT.(active_lambda_idx)
    centered .-= sum(centered) / FT(n)
    denom = sum(abs2, centered)
    if !(denom > zero(FT))
        return one(FT)
    end

    max_lag = min(div(n, 5), 200)
    rho_sum = zero(FT)
    for lag in 1:max_lag
        numer = zero(FT)
        @inbounds for i in 1:(n - lag)
            numer += centered[i] * centered[i + lag]
        end
        rho_k = numer / denom
        if rho_k <= zero(FT)
            break
        end
        rho_sum += rho_k
    end

    tau_int = one(FT) + FT(2) * rho_sum
    return max(one(FT), FT(n) / tau_int)
end


"""
    estimate_leg_dg_from_reference(energies, active_lambda_idx, awh_bias,
                                   num_lambda_states, beta; volumes=FT[],
                                   P0_energy_per_vol=zero(FT))

Estimate a leg free energy as the difference between the two endpoint free
energies reconstructed from the reference ensemble.
"""
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


"""
    split_half_ranges(n)

Split `1:n` into contiguous first-half and second-half ranges.
"""
function split_half_ranges(n::Int)
    n_first = fld(n, 2)
    return (1:n_first, (n_first + 1):n)
end

"""
    probe_frame_indices(n_frames; frame_stride=1, min_frames=2, max_frames=0)

Select probe-frame indices while guaranteeing inclusion of the last frame and
respecting optional thinning and capping rules.
"""
function probe_frame_indices(
    n_frames::Int;
    frame_stride::Int=1,
    min_frames::Int=2,
    max_frames::Int=0,
)
    if n_frames <= 0
        return Int[]
    end

    stride = max(1, frame_stride)
    required = min(n_frames, max(2, min_frames))
    idxs = collect(1:stride:n_frames)
    if isempty(idxs) || last(idxs) != n_frames
        push!(idxs, n_frames)
    end

    # When the raw stride still leaves too many frames, switch to evenly spaced
    # sampling so the probe cost stays bounded.
    if max_frames > 0 && length(idxs) > max_frames
        target = min(n_frames, max(2, max_frames))
        idxs = unique(clamp.(round.(Int, range(1, n_frames, length=target)), 1, n_frames))
    end

    if length(idxs) < required
        idxs = unique(clamp.(round.(Int, range(1, n_frames, length=required)), 1, n_frames))
    end

    sort!(idxs)
    if isempty(idxs) || last(idxs) != n_frames
        push!(idxs, n_frames)
        sort!(idxs)
    end
    return idxs
end


"""
    discard_leading_probe_frames(frame_idxs; discard_fraction=0.0)

Drop an initial fraction of already-selected probe frames. This is used to
ignore the earliest part of the probe trajectory during Stage B reweighting.
"""
function discard_leading_probe_frames(frame_idxs::Vector{Int}; discard_fraction::Real=0.0)
    if isempty(frame_idxs)
        return Int[]
    end
    if !(0.0 <= discard_fraction < 1.0)
        throw(ArgumentError("discard_fraction must lie in [0, 1)."))
    end

    n_discard = floor(Int, length(frame_idxs) * discard_fraction)
    if n_discard <= 0
        return copy(frame_idxs)
    end
    if n_discard >= length(frame_idxs)
        return Int[]
    end

    return copy(frame_idxs[(n_discard + 1):end])
end


"""
    select_probe_frame_indices(n_frames; kwargs...)

Return both the initially selected probe indices and the subset retained after
leading-frame discard.
"""
function select_probe_frame_indices(
    n_frames::Int;
    frame_stride::Int=1,
    min_frames::Int=2,
    max_frames::Int=0,
    discard_fraction::Real=0.0,
)
    selected_idxs = probe_frame_indices(
        n_frames;
        frame_stride=frame_stride,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    retained_idxs = discard_leading_probe_frames(selected_idxs; discard_fraction=discard_fraction)
    return (selected=selected_idxs, retained=retained_idxs)
end


"""
    count_full_round_trips(active_idx_history, low_idx, high_idx)

Count completed `low -> high -> low` round trips in the λ-index history.
"""
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


"""
    endpoint_occupancy_fractions(active_idx_history, low_idx, high_idx)

Return the fractions of frames spent at the low and high endpoint λ states.
"""
function endpoint_occupancy_fractions(active_idx_history::Vector{Int}, low_idx::Int, high_idx::Int)
    n_frames = length(active_idx_history)
    if n_frames == 0
        return zero(FT), zero(FT)
    end
    low_frac = FT(count(==(low_idx), active_idx_history)) / FT(n_frames)
    high_frac = FT(count(==(high_idx), active_idx_history)) / FT(n_frames)
    return low_frac, high_frac
end


"""
    evaluate_stage_a_readiness(awh_sim, awh_tol; kwargs...)

Evaluate the cheap, continuously updated readiness metrics used before launching
an expensive Stage B probe.
"""
function evaluate_stage_a_readiness(
    awh_sim,
    awh_tol::FT;
    tail_lag::Int,
    min_lambda_ess::Int,
    min_linear_neff::Int,
    min_round_trips::Int,
    endpoint_min_fraction::FT,
    low_idx::Int=1,
    high_idx::Int
) where {FT <: AbstractFloat}
    df_ready, df_mean, linear_neff = awh_linear_stage_stats(awh_sim, awh_tol; max_lag=tail_lag)
    neff_ready = !awh_sim.state.in_initial_stage && linear_neff >= FT(min_linear_neff)

    # Stage A deliberately mixes bias-stability and trajectory-mixing criteria:
    # a flat bias alone is not enough if λ exploration is still poor.
    idx_history = get_awh_active_idx_history(awh_sim)
    lambda_ess = estimate_lambda_history_ess(idx_history, FT)
    lambda_ess_ready = lambda_ess >= FT(min_lambda_ess)
    round_trips = count_full_round_trips(idx_history, low_idx, high_idx)
    round_trip_ready = round_trips >= min_round_trips

    endpoint_low, endpoint_high = endpoint_occupancy_fractions(idx_history, low_idx, high_idx)
    endpoint_ready = endpoint_low >= endpoint_min_fraction && endpoint_high >= endpoint_min_fraction

    ready = df_ready && lambda_ess_ready && round_trip_ready && endpoint_ready
    return (
        ready = ready,
        df_ready = df_ready,
        df_mean = df_mean,
        lambda_ess = lambda_ess,
        lambda_ess_ready = lambda_ess_ready,
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


"""
    run_stage_b_probe(awh_sim, sys_base, theta_params, param_names, idxs,
                      num_lambda_states, beta, awh_split_tol_kT,
                      awh_parity_tol_kT; kwargs...)

Clone the current leg, run a short probe trajectory, and test whether the probe
is consistent under split-half reweighting and MBAR/AWH profile parity checks.
"""
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
    P0_energy_per_vol::FT=zero(FT),
    probe_frame_stride::Int=1,
    probe_min_frames::Int=2,
    probe_max_frames::Int=0,
    awh_probe_discard_fraction::Real=0.0,
    awh_control::AWHControlConfig=AWHControlConfig(),
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

    # Probe on a cloned state so a failed Stage B check does not disturb the
    # main leg that continues accumulating readiness statistics.
    probe_sim = AWHSimulation(
        deepcopy(awh_sim.state);
        num_md_steps=awh_sim.n_md_steps,
        awh_simulation_control_kwargs(awh_control)...,
    )
    clear_awh_logger_histories!(probe_sim)
    probe_sim.state.active_sys.loggers.awh_logger.should_log = true
    probe_md_timed = timed_phase("Stage B Probe MD", leg_name; md_steps=md_steps_probe) do
        simulate!(probe_sim, md_steps_probe)
    end

    logger_probe_raw = get_production_logger(probe_sim, "$leg_name probe")
    n_frames_raw = length(logger_probe_raw.active_idx_history)
    if n_frames_raw < 2
        @info "Stage B ($(leg_name)) early exit: insufficient probe frames (n_frames=$n_frames_raw, required=2) | probe_md_wall_s=$(round(probe_md_timed.timing.wall_s, digits=3)) | probe_md_steps_per_s=$(round(probe_md_timed.timing.steps_per_s, digits=2))"
        return (
            ready = false,
            split_ready = false,
            split_gap = FT(Inf),
            parity_ready = false,
            parity_gap = FT(Inf),
            n_frames = n_frames_raw,
            dG_half_1 = FT(NaN),
            dG_half_2 = FT(NaN)
        )
    end

    probe_selection = select_probe_frame_indices(
        n_frames_raw;
        frame_stride=probe_frame_stride,
        min_frames=probe_min_frames,
        max_frames=probe_max_frames,
        discard_fraction=awh_probe_discard_fraction,
    )
    n_frames_selected = length(probe_selection.selected)
    n_frames = length(probe_selection.retained)
    max_frames_msg = probe_max_frames > 0 ? string(probe_max_frames) : "none"
    @info "Stage B ($(leg_name)) probe frame selection: raw_frames=$n_frames_raw | selected_frames=$n_frames_selected | retained_frames=$n_frames | discard_fraction=$(round(Float64(awh_probe_discard_fraction), digits=3)) | stride=$(max(1, probe_frame_stride)) | min_frames=$(max(2, probe_min_frames)) | max_frames=$max_frames_msg"

    if n_frames < 2
        @info "Stage B ($(leg_name)) early exit: insufficient retained probe frames after discard (n_frames=$n_frames, required=2) | probe_md_wall_s=$(round(probe_md_timed.timing.wall_s, digits=3)) | probe_md_steps_per_s=$(round(probe_md_timed.timing.steps_per_s, digits=2))"
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

    logger_probe = subset_awh_logger_frames(logger_probe_raw, probe_selection.retained)
    n_frames = length(logger_probe.active_idx_history)

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
        # Split-half agreement checks time stability, while parity checks that
        # the sampled AWH bias matches the MBAR-reconstructed profile.
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
