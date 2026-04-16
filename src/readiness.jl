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



function analyze_stage_b_probe(
    probe_sim::AWHSimulation,
    bias_data,
    sys_base,
    theta_params::Vector{FT},
    param_names::Vector{String},
    idxs,
    coupled_state_idx::Int,
    decoupled_state_idx::Int,
    beta::BT,
    awh_split_tol_kT::ST,
    awh_parity_tol_kT::PTT;
    leg_name::String,
    include_pv::Bool=false,
    P0_energy_per_vol::PV=zero(BT),
    probe_frame_stride::Int=1,
    probe_min_frames::Int=2,
    probe_max_frames::Int=0,
    awh_probe_discard_fraction::Real=0.0,
    parity_gate_mode::Symbol=:support_aware_max,
    parity_support_threshold::SP=zero(BT),
    parity_near_pass_factor::NP=BT(2),
    support_allow_missing::Int=0,
    probe_hotspot_state_idxs::Vector{Int}=Int[],
    probe_hotspot_min_state_occupancy_floor::FT=zero(FT),
    probe_tail_state_idxs::Vector{Int}=Int[],
    probe_tail_min_state_occupancy_floor::FT=zero(FT),
    probe_md_timed=nothing,
) where {
    FT <: AbstractFloat,
    BT <: AbstractFloat,
    ST <: AbstractFloat,
    PTT <: AbstractFloat,
    PV <: AbstractFloat,
    SP <: AbstractFloat,
    NP <: AbstractFloat,
}
    ET = promote_energy_analysis_type(
        beta,
        awh_split_tol_kT,
        awh_parity_tol_kT,
        P0_energy_per_vol,
        parity_support_threshold,
        parity_near_pass_factor,
    )
    n_total_states = length(probe_sim.state.partition.λ_atoms)
    logger_probe_raw = get_production_logger(probe_sim, "$leg_name probe")
    n_frames_raw = length(logger_probe_raw.active_idx_history)
    if n_frames_raw < 2
        @info "Stage B ($(leg_name)) early exit: insufficient probe frames (n_frames=$n_frames_raw, required=2)$(stage_b_probe_timing_suffix(probe_md_timed))"
        return stage_b_probe_empty_result(ET, n_frames_raw, n_total_states)
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
        @info "Stage B ($(leg_name)) early exit: insufficient retained probe frames after discard (n_frames=$n_frames, required=2)$(stage_b_probe_timing_suffix(probe_md_timed))"
        return stage_b_probe_empty_result(ET, n_frames, n_total_states)
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

    log_gibbs_weights = awh_log_gibbs_weights(bias_data)
    volumes_probe = include_pv ? ET.(ustrip.(logger_probe.volume_history)) : ET[]
    log_mixture_denom = include_pv ?
        reference_log_mixture_denominator(
            u_probe_ref,
            log_gibbs_weights,
            beta;
            volumes=volumes_probe,
            P0_energy_per_vol=P0_energy_per_vol,
        ) :
        reference_log_mixture_denominator(
            u_probe_ref,
            log_gibbs_weights,
            beta,
        )

    split_parity_timed = timed_phase("Stage B Split-Parity", leg_name) do
        compute_stage_b_split_parity(
            u_probe_ref,
            log_mixture_denom,
            bias_data.f,
            coupled_state_idx,
            decoupled_state_idx,
            beta,
            awh_split_tol_kT,
            awh_parity_tol_kT,
            volumes=volumes_probe,
            P0_energy_per_vol=include_pv ? P0_energy_per_vol : zero(ET),
            parity_gate_mode=parity_gate_mode,
            parity_support_threshold=parity_support_threshold,
            parity_near_pass_factor=parity_near_pass_factor,
            support_allow_missing=support_allow_missing,
        )
    end

    split_parity = split_parity_timed.result
    dG_half_1 = split_parity.dG_half_1
    dG_half_2 = split_parity.dG_half_2
    split_gap = split_parity.split_gap
    split_ready = split_parity.split_ready
    parity_gap = split_parity.parity_gap
    parity_ready = split_parity.parity_ready
    probe_residence = residence_length_summary(logger_probe.active_idx_history, FT)
    probe_occupancies = state_occupancy_fractions(logger_probe.active_idx_history, size(u_probe_ref, 2), FT)
    probe_endpoint_low, probe_endpoint_high = endpoint_occupancy_fractions(
        logger_probe.active_idx_history,
        coupled_state_idx,
        decoupled_state_idx,
    )
    hotspot_occ_entries = String[]
    for idx in split_parity.parity_top_state_idxs
        if 1 <= idx <= length(probe_occupancies)
            push!(hotspot_occ_entries, "λ$(idx)=$(round(probe_occupancies[idx], digits=4))")
        end
    end
    effective_hotspot_state_idxs = isempty(probe_hotspot_state_idxs) ? probe_tail_state_idxs : probe_hotspot_state_idxs
    effective_hotspot_floor = probe_hotspot_min_state_occupancy_floor == zero(FT) ?
        probe_tail_min_state_occupancy_floor :
        probe_hotspot_min_state_occupancy_floor
    valid_hotspot_state_idxs = sort(unique(filter(idx -> 1 <= idx <= length(probe_occupancies), effective_hotspot_state_idxs)))
    hotspot_occupancy = isempty(valid_hotspot_state_idxs) ? zero(FT) : sum(probe_occupancies[valid_hotspot_state_idxs])
    hotspot_min_state_occupancy = isempty(valid_hotspot_state_idxs) ? zero(FT) : minimum(probe_occupancies[valid_hotspot_state_idxs])
    hotspot_low_occupancy_states = isempty(valid_hotspot_state_idxs) ? Int[] : [
        idx for idx in valid_hotspot_state_idxs if probe_occupancies[idx] < effective_hotspot_floor
    ]
    parity_residual = hasproperty(split_parity, :parity_residual) ? split_parity.parity_residual : ET[]
    tail_residual_entries = isempty(valid_hotspot_state_idxs) || isempty(parity_residual) ? String[] : [
        "λ$(idx)=$(round(parity_residual[idx], digits=4))" for idx in valid_hotspot_state_idxs
    ]
    diagnostics = split_parity.diagnostics *
        " | probe_switches=$(probe_residence.switch_count)" *
        " | probe_dwell_samples=(mean=$(round(probe_residence.mean_residence, digits=2)), med=$(round(probe_residence.median_residence, digits=2)))"
    if !isempty(hotspot_occ_entries)
        diagnostics *= " | probe_occ=[" * join(hotspot_occ_entries, "; ") * "]"
    end
    diagnostics *= " | probe_endpt=(low=$(round(probe_endpoint_low, digits=4)), high=$(round(probe_endpoint_high, digits=4)))"
    if !isempty(valid_hotspot_state_idxs)
        tail_low_msg = isempty(hotspot_low_occupancy_states) ? "-" : join(["λ$(idx)" for idx in hotspot_low_occupancy_states], ",")
        diagnostics *= " | probe_tail=(total=$(round(hotspot_occupancy, digits=4)), min=$(round(hotspot_min_state_occupancy, digits=4)), low=$tail_low_msg)"
    end
    if !isempty(tail_residual_entries)
        diagnostics *= " | tail_resid=[" * join(tail_residual_entries, "; ") * "]"
    end
    parity_worst_state_idx = split_parity.parity_worst_state_idx
    parity_worst_state_residual = split_parity.parity_worst_state_residual

    return (
        ready = split_parity.ready,
        split_ready = split_ready,
        split_gap = split_gap,
        parity_ready = parity_ready,
        parity_gap = parity_gap,
        raw_parity_gap = split_parity.raw_parity_gap,
        supported_parity_gap = split_parity.supported_parity_gap,
        endpoint_parity_gap = split_parity.endpoint_parity_gap,
        endpoint_parity_ready = split_parity.endpoint_parity_ready,
        n_total_states = n_total_states,
        n_frames = n_frames,
        dG_half_1 = dG_half_1,
        dG_half_2 = dG_half_2,
        diagnostics = diagnostics,
        parity_worst_state_idx = parity_worst_state_idx,
        parity_worst_state_residual = parity_worst_state_residual,
        n_supported_states = split_parity.n_supported_states,
        support_threshold = split_parity.support_threshold,
        support_coverage_ready = split_parity.support_coverage_ready,
        required_supported_states = split_parity.required_supported_states,
        failure_mode = split_parity.failure_mode,
        near_pass = split_parity.near_pass,
        probe_energies = u_probe_ref,
        probe_log_mixture_denom = log_mixture_denom,
        probe_volumes = volumes_probe,
    )
end


function advance_stage_b_probe!(
    probe_sim::AWHSimulation,
    bias_data,
    md_steps_segment::Int,
    sys_base,
    theta_params::Vector{FT},
    param_names::Vector{String},
    idxs,
    coupled_state_idx::Int,
    decoupled_state_idx::Int,
    beta::BT,
    awh_split_tol_kT::ST,
    awh_parity_tol_kT::PTT;
    leg_name::String,
    include_pv::Bool=false,
    P0_energy_per_vol::PV=zero(BT),
    probe_frame_stride::Int=1,
    probe_min_frames::Int=2,
    probe_max_frames::Int=0,
    awh_probe_discard_fraction::Real=0.0,
    parity_gate_mode::Symbol=:support_aware_max,
    parity_support_threshold::SP=zero(BT),
    parity_near_pass_factor::NP=BT(2),
    support_allow_missing::Int=0,
    probe_hotspot_state_idxs::Vector{Int}=Int[],
    probe_hotspot_min_state_occupancy_floor::FT=zero(FT),
    probe_tail_state_idxs::Vector{Int}=Int[],
    probe_tail_min_state_occupancy_floor::FT=zero(FT),
) where {
    FT <: AbstractFloat,
    BT <: AbstractFloat,
    ST <: AbstractFloat,
    PTT <: AbstractFloat,
    PV <: AbstractFloat,
    SP <: AbstractFloat,
    NP <: AbstractFloat,
}
    probe_md_timed = timed_phase("Stage B Probe MD", leg_name; md_steps=md_steps_segment) do
        simulate!(probe_sim, md_steps_segment)
    end
    return analyze_stage_b_probe(
        probe_sim,
        bias_data,
        sys_base,
        theta_params,
        param_names,
        idxs,
        coupled_state_idx,
        decoupled_state_idx,
        beta,
        awh_split_tol_kT,
        awh_parity_tol_kT;
        leg_name=leg_name,
        include_pv=include_pv,
        P0_energy_per_vol=P0_energy_per_vol,
        probe_frame_stride=probe_frame_stride,
        probe_min_frames=probe_min_frames,
        probe_max_frames=probe_max_frames,
        awh_probe_discard_fraction=awh_probe_discard_fraction,
        parity_gate_mode=parity_gate_mode,
        parity_support_threshold=parity_support_threshold,
        parity_near_pass_factor=parity_near_pass_factor,
        support_allow_missing=support_allow_missing,
        probe_hotspot_state_idxs=probe_hotspot_state_idxs,
        probe_hotspot_min_state_occupancy_floor=probe_hotspot_min_state_occupancy_floor,
        probe_tail_state_idxs=probe_tail_state_idxs,
        probe_tail_min_state_occupancy_floor=probe_tail_min_state_occupancy_floor,
        probe_md_timed=probe_md_timed,
    )
end


function start_stage_b_probe(
    awh_sim::AWHSimulation,
    sys_base,
    theta_params::Vector{FT},
    param_names::Vector{String},
    idxs,
    coupled_state_idx::Int,
    decoupled_state_idx::Int,
    beta::BT,
    awh_split_tol_kT::ST,
    awh_parity_tol_kT::PTT;
    md_steps_probe::Int,
    leg_name::String,
    probe_num_md_steps::Union{Nothing, Int}=nothing,
    include_pv::Bool=false,
    P0_energy_per_vol::PV=zero(BT),
    probe_frame_stride::Int=1,
    probe_min_frames::Int=2,
    probe_max_frames::Int=0,
    awh_probe_discard_fraction::Real=0.0,
    parity_gate_mode::Symbol=:support_aware_max,
    parity_support_threshold::SP=zero(BT),
    parity_near_pass_factor::NP=BT(2),
    support_allow_missing::Int=0,
    probe_hotspot_state_idxs::Vector{Int}=Int[],
    probe_hotspot_min_state_occupancy_floor::FT=zero(FT),
    probe_tail_state_idxs::Vector{Int}=Int[],
    probe_tail_min_state_occupancy_floor::FT=zero(FT),
) where {
    FT <: AbstractFloat,
    BT <: AbstractFloat,
    ST <: AbstractFloat,
    PTT <: AbstractFloat,
    PV <: AbstractFloat,
    SP <: AbstractFloat,
    NP <: AbstractFloat,
}
    probe_sim, bias_data = build_stage_b_probe_sim(
        awh_sim,
        md_steps_probe;
        probe_num_md_steps=probe_num_md_steps,
    )
    probe_stats = advance_stage_b_probe!(
        probe_sim,
        bias_data,
        md_steps_probe,
        sys_base,
        theta_params,
        param_names,
        idxs,
        coupled_state_idx,
        decoupled_state_idx,
        beta,
        awh_split_tol_kT,
        awh_parity_tol_kT;
        leg_name=leg_name,
        include_pv=include_pv,
        P0_energy_per_vol=P0_energy_per_vol,
        probe_frame_stride=probe_frame_stride,
        probe_min_frames=probe_min_frames,
        probe_max_frames=probe_max_frames,
        awh_probe_discard_fraction=awh_probe_discard_fraction,
        parity_gate_mode=parity_gate_mode,
        parity_support_threshold=parity_support_threshold,
        parity_near_pass_factor=parity_near_pass_factor,
        support_allow_missing=support_allow_missing,
        probe_hotspot_state_idxs=probe_hotspot_state_idxs,
        probe_hotspot_min_state_occupancy_floor=probe_hotspot_min_state_occupancy_floor,
        probe_tail_state_idxs=probe_tail_state_idxs,
        probe_tail_min_state_occupancy_floor=probe_tail_min_state_occupancy_floor,
    )
    return (probe_sim=probe_sim, bias_data=bias_data, stats=probe_stats)
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
    estimate_leg_dg_from_reference(energies, active_lambda_idx, log_gibbs_weights,
                                   coupled_state_idx, decoupled_state_idx, beta;
                                   volumes=FT[], P0_energy_per_vol=zero(FT))

Estimate a leg free energy as the difference between the two endpoint free
energies reconstructed from the reference ensemble.
"""
function estimate_leg_dg_from_reference(
    energies::AbstractMatrix{ET},
    active_lambda_idx::Vector{Int},
    log_gibbs_weights::AbstractVector{GT},
    coupled_state_idx::Int,
    decoupled_state_idx::Int,
    beta::BT;
    volumes::AbstractVector{VT}=BT[],
    P0_energy_per_vol::PT=zero(BT)
) where {ET <: AbstractFloat, GT <: AbstractFloat, BT <: AbstractFloat, VT <: AbstractFloat, PT <: AbstractFloat}
    AT = promote_energy_analysis_type(energies, log_gibbs_weights, beta, volumes, P0_energy_per_vol)
    dummy_names = String[]
    dummy_grads = Dict{String, Matrix{AT}}()
    _, F_1 = compute_global_endpoint_gradients(
        dummy_names, dummy_grads, energies, energies, active_lambda_idx, coupled_state_idx,
        beta, log_gibbs_weights, volumes, P0_energy_per_vol; compute_gradients=false
    )
    _, F_0 = compute_global_endpoint_gradients(
        dummy_names, dummy_grads, energies, energies, active_lambda_idx, decoupled_state_idx,
        beta, log_gibbs_weights, volumes, P0_energy_per_vol; compute_gradients=false
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
    state_occupancy_fractions(active_idx_history, n_states, FT)

Return the per-state fractions of frames spent in each active λ state.
"""
function state_occupancy_fractions(active_idx_history::Vector{Int}, n_states::Int, ::Type{FT}) where {FT <: AbstractFloat}
    n_states <= 0 && return FT[]

    occupancies = zeros(FT, n_states)
    n_frames = length(active_idx_history)
    if n_frames == 0
        return occupancies
    end

    inv_n_frames = inv(FT(n_frames))
    for idx in active_idx_history
        if 1 <= idx <= n_states
            occupancies[idx] += inv_n_frames
        end
    end
    return occupancies
end

"""
    recent_active_idx_history(active_idx_history, history_window_length)

Return the most recent `history_window_length` λ-history entries. Non-positive
window lengths keep the full history.
"""
function recent_active_idx_history(active_idx_history::Vector{Int}, history_window_length::Int)
    if history_window_length <= 0 || length(active_idx_history) <= history_window_length
        return copy(active_idx_history)
    end
    first_idx = length(active_idx_history) - history_window_length + 1
    return copy(@view active_idx_history[first_idx:end])
end

"""
    count_lambda_switches(active_idx_history)

Count the number of λ-state changes in an active-index history.
"""
function count_lambda_switches(active_idx_history::Vector{Int})
    n_frames = length(active_idx_history)
    if n_frames < 2
        return 0
    end
    n_switches = 0
    prev_idx = active_idx_history[1]
    for idx in @view active_idx_history[2:end]
        if idx != prev_idx
            n_switches += 1
            prev_idx = idx
        end
    end
    return n_switches
end

"""
    contiguous_residence_lengths(active_idx_history)

Return the contiguous λ-residence lengths measured in λ samples.
"""
function contiguous_residence_lengths(active_idx_history::Vector{Int})
    if isempty(active_idx_history)
        return Int[]
    end

    lengths = Int[]
    current_idx = active_idx_history[1]
    current_length = 1
    for idx in @view active_idx_history[2:end]
        if idx == current_idx
            current_length += 1
        else
            push!(lengths, current_length)
            current_idx = idx
            current_length = 1
        end
    end
    push!(lengths, current_length)
    return lengths
end

"""
    residence_length_summary(active_idx_history, FT)

Summarize λ residence lengths via switch count and mean/median dwell length in
units of λ samples.
"""
function residence_length_summary(active_idx_history::Vector{Int}, ::Type{FT}) where {FT <: AbstractFloat}
    lengths = contiguous_residence_lengths(active_idx_history)
    if isempty(lengths)
        return (switch_count=0, mean_residence=zero(FT), median_residence=zero(FT))
    end

    sorted_lengths = sort(lengths)
    n_lengths = length(sorted_lengths)
    median_residence = if isodd(n_lengths)
        FT(sorted_lengths[(n_lengths + 1) ÷ 2])
    else
        lower = sorted_lengths[n_lengths ÷ 2]
        upper = sorted_lengths[n_lengths ÷ 2 + 1]
        (FT(lower) + FT(upper)) / FT(2)
    end

    return (
        switch_count=count_lambda_switches(active_idx_history),
        mean_residence=FT(sum(lengths)) / FT(n_lengths),
        median_residence=median_residence,
    )
end

function awh_state_count(awh_sim, low_idx::Int=1, high_idx::Int=1)
    if hasproperty(awh_sim.state, :partition) && hasproperty(awh_sim.state.partition, :λ_atoms)
        return length(awh_sim.state.partition.λ_atoms)
    elseif hasproperty(awh_sim.state, :f)
        return length(awh_sim.state.f)
    elseif hasproperty(awh_sim.state, :rho)
        return length(awh_sim.state.rho)
    end

    idx_history = get_awh_active_idx_history(awh_sim)
    if isempty(idx_history)
        return max(low_idx, high_idx)
    end
    return max(high_idx, maximum(idx_history))
end

function advance_stage_a_streak(current_streak::Int, stage_a_ready::Bool)
    return stage_a_ready ? current_streak + 1 : 0
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
    history_window_length::Int=0,
    hotspot_state_idxs::Vector{Int}=Int[],
    tail_state_idxs::Vector{Int}=Int[],
    endpoint_state_idxs::Vector{Int}=Int[],
    hotspot_min_state_occupancy_floor::FT=zero(FT),
    tail_min_state_occupancy_floor::FT=zero(FT),
    endpoint_high_min_fraction_abs::FT=zero(FT),
    low_idx::Int=1,
    high_idx::Int
) where {FT <: AbstractFloat}
    df_ready, df_mean, linear_neff = awh_linear_stage_stats(awh_sim, awh_tol; max_lag=tail_lag)
    neff_ready = !awh_sim.state.in_initial_stage && linear_neff >= FT(min_linear_neff)

    # Stage A deliberately mixes bias-stability and trajectory-mixing criteria:
    # a flat bias alone is not enough if λ exploration is still poor. λ-history
    # ESS remains cumulative, but occupancy-style gates focus on recent behavior.
    idx_history_full = get_awh_active_idx_history(awh_sim)
    idx_history_recent = recent_active_idx_history(idx_history_full, history_window_length)
    lambda_ess = estimate_lambda_history_ess(idx_history_full, FT)
    tau_int_est = isempty(idx_history_full) ? zero(FT) : FT(length(idx_history_full)) / lambda_ess
    lambda_ess_ready = lambda_ess >= FT(min_lambda_ess)
    round_trips = count_full_round_trips(idx_history_full, low_idx, high_idx)
    round_trip_ready = round_trips >= min_round_trips

    endpoint_low, _ = endpoint_occupancy_fractions(idx_history_recent, low_idx, high_idx)
    n_states = awh_state_count(awh_sim, low_idx, high_idx)
    occupancies = state_occupancy_fractions(idx_history_recent, n_states, FT)
    valid_endpoint_state_idxs = sort(unique(filter(idx -> 1 <= idx <= length(occupancies), isempty(endpoint_state_idxs) ? [high_idx] : endpoint_state_idxs)))
    endpoint_high = isempty(valid_endpoint_state_idxs) ? zero(FT) : sum(occupancies[valid_endpoint_state_idxs])
    endpoint_high_required = max(endpoint_min_fraction * FT(max(1, length(valid_endpoint_state_idxs))), endpoint_high_min_fraction_abs)
    endpoint_ready = endpoint_low >= endpoint_min_fraction && endpoint_high >= endpoint_high_required
    low_occupancy_threshold = FT(0.01)
    min_state_occupancy = isempty(occupancies) ? zero(FT) : minimum(occupancies)
    low_occupancy_states = findall(x -> x < low_occupancy_threshold, occupancies)
    effective_hotspot_state_idxs = isempty(hotspot_state_idxs) ? tail_state_idxs : hotspot_state_idxs
    effective_hotspot_floor = hotspot_min_state_occupancy_floor == zero(FT) ?
        tail_min_state_occupancy_floor :
        hotspot_min_state_occupancy_floor
    valid_hotspot_state_idxs = sort(unique(filter(idx -> 1 <= idx <= length(occupancies), effective_hotspot_state_idxs)))
    hotspot_occupancy = isempty(valid_hotspot_state_idxs) ? zero(FT) : sum(occupancies[valid_hotspot_state_idxs])
    hotspot_min_state_occupancy = isempty(valid_hotspot_state_idxs) ? one(FT) : minimum(occupancies[valid_hotspot_state_idxs])
    hotspot_low_occupancy_states = isempty(valid_hotspot_state_idxs) ? Int[] : [
        idx for idx in valid_hotspot_state_idxs if occupancies[idx] < effective_hotspot_floor
    ]
    hotspot_ready = isempty(valid_hotspot_state_idxs) || hotspot_min_state_occupancy >= effective_hotspot_floor
    residence_summary = residence_length_summary(idx_history_recent, FT)

    ready = df_ready && lambda_ess_ready && round_trip_ready && endpoint_ready && hotspot_ready
    return (
        ready = ready,
        df_ready = df_ready,
        df_mean = df_mean,
        lambda_ess = lambda_ess,
        tau_int_est = tau_int_est,
        lambda_ess_ready = lambda_ess_ready,
        linear_neff = linear_neff,
        neff_ready = neff_ready,
        switch_count = residence_summary.switch_count,
        mean_residence = residence_summary.mean_residence,
        median_residence = residence_summary.median_residence,
        round_trips = round_trips,
        round_trip_ready = round_trip_ready,
        endpoint_low = endpoint_low,
        endpoint_high = endpoint_high,
        endpoint_high_required = endpoint_high_required,
        endpoint_ready = endpoint_ready,
        hotspot_occupancy = hotspot_occupancy,
        hotspot_min_state_occupancy = hotspot_min_state_occupancy,
        hotspot_ready = hotspot_ready,
        hotspot_low_occupancy_states = hotspot_low_occupancy_states,
        tail_occupancy = hotspot_occupancy,
        tail_min_state_occupancy = hotspot_min_state_occupancy,
        tail_ready = hotspot_ready,
        tail_low_occupancy_states = hotspot_low_occupancy_states,
        min_state_occupancy = min_state_occupancy,
        low_occupancy_states = low_occupancy_states,
        n_hist = length(idx_history_full),
        n_hist_recent = length(idx_history_recent)
    )
end


"""
    reset_frozen_bias_state!(awh_state)

Clear transient AWH accumulators on a cloned state so a frozen-bias probe or
production segment starts from the stored bias instead of inheriting a
partially filled update block from the main leg.
"""
function reset_frozen_bias_state!(awh_state)
    fill!(awh_state.w_seg, zero(eltype(awh_state.w_seg)))
    fill!(awh_state.w2_seg, zero(eltype(awh_state.w2_seg)))
    fill!(awh_state.w_last, zero(eltype(awh_state.w_last)))
    awh_state.n_accum = 0
    empty!(awh_state.visited_windows)
    empty!(awh_state.cv_buffer)
    return awh_state
end


"""
    build_frozen_bias_awh_sim(awh_sim, md_steps_segment; probe_num_md_steps=nothing)

Clone `awh_sim` for a short segment while freezing the AWH bias throughout that
segment. The returned `(frozen_sim, bias_data)` pair keeps logged frames, MBAR
denominators, and AWH profile comparisons tied to the same fixed reference
bias.
"""
function build_frozen_bias_awh_sim(
    awh_sim::AWHSimulation,
    md_steps_segment::Int;
    probe_num_md_steps::Union{Nothing, Int}=nothing,
)
    frozen_n_md_steps = isnothing(probe_num_md_steps) ? awh_sim.n_md_steps : probe_num_md_steps
    frozen_n_md_steps > 0 || throw(ArgumentError("`probe_num_md_steps` must be positive, got $frozen_n_md_steps."))
    n_segment_iterations = fld(max(0, md_steps_segment), frozen_n_md_steps)
    # `reset_frozen_bias_state!` restarts the cloned segment with `n_accum = 0`,
    # so freezing only requires `update_freq` to exceed the number of AWH
    # sampling iterations that will occur in this segment.
    frozen_update_freq = max(awh_sim.update_freq, n_segment_iterations + 1)

    bias_data = extract_awh_data(awh_sim)
    frozen_sim = AWHSimulation(
        deepcopy(awh_sim.state);
        num_md_steps=frozen_n_md_steps,
        update_freq=frozen_update_freq,
        well_tempered_factor=awh_sim.well_tempered_fac,
        coverage_threshold=awh_sim.coverage_threshold,
        significant_weight=awh_sim.significant_weight,
        coverage_type=awh_sim.coverage_type,
        log_freq=awh_sim.log_freq,
    )
    reset_frozen_bias_state!(frozen_sim.state)
    clear_awh_logger_histories!(frozen_sim)
    enable_awh_logger_histories!(frozen_sim)
    return frozen_sim, bias_data
end


"""
    reset_stage_b_probe_state!(awh_state)

Backward-compatible wrapper around `reset_frozen_bias_state!`.
"""
function reset_stage_b_probe_state!(awh_state)
    return reset_frozen_bias_state!(awh_state)
end


"""
    build_stage_b_probe_sim(awh_sim, md_steps_probe; probe_num_md_steps=nothing)

Backward-compatible wrapper around `build_frozen_bias_awh_sim`.
"""
function build_stage_b_probe_sim(
    awh_sim::AWHSimulation,
    md_steps_probe::Int;
    probe_num_md_steps::Union{Nothing, Int}=nothing,
)
    return build_frozen_bias_awh_sim(
        awh_sim,
        md_steps_probe;
        probe_num_md_steps=probe_num_md_steps,
    )
end

function stage_b_probe_timing_suffix(probe_md_timed)
    if isnothing(probe_md_timed)
        return ""
    end
    return " | probe_md_wall_s=$(round(probe_md_timed.timing.wall_s, digits=3)) | probe_md_steps_per_s=$(round(probe_md_timed.timing.steps_per_s, digits=2))"
end


function stage_b_probe_empty_result(::Type{ET}, n_frames::Int, n_total_states::Int) where {ET <: AbstractFloat}
    return (
        ready = false,
        split_ready = false,
        split_gap = ET(Inf),
        parity_ready = false,
        parity_gap = ET(Inf),
        raw_parity_gap = ET(Inf),
        supported_parity_gap = ET(Inf),
        endpoint_parity_gap = ET(Inf),
        endpoint_parity_ready = false,
        n_total_states = n_total_states,
        n_frames = n_frames,
        dG_half_1 = ET(NaN),
        dG_half_2 = ET(NaN),
        diagnostics = "",
        parity_worst_state_idx = 0,
        parity_worst_state_residual = ET(NaN),
        n_supported_states = 0,
        support_threshold = zero(ET),
        support_coverage_ready = false,
        required_supported_states = 0,
        failure_mode = :insufficient_frames,
        near_pass = false,
        probe_energies = zeros(ET, max(0, n_frames), max(0, n_total_states)),
        probe_log_mixture_denom = ET[],
        probe_volumes = ET[],
    )
end

"""
    stage_b_parity_diagnostics(parity_residual, ess_by_state; top_k=3)

Build a compact diagnostics string summarizing the worst parity-residual states
and their target-state ESS support.
"""
function stage_b_parity_diagnostics(
    parity_residual::Vector{FT},
    ess_by_state::Vector{FT};
    top_k::Int=3,
) where {FT <: AbstractFloat}
    n_states = length(parity_residual)
    if n_states == 0
        return "", 0, FT(NaN), Int[]
    end
    if length(ess_by_state) != n_states
        throw(ArgumentError("stage_b_parity_diagnostics expected matching lengths, got $(n_states) and $(length(ess_by_state))."))
    end

    abs_residual = abs.(parity_residual)
    worst_idx = argmax(abs_residual)
    worst_residual = parity_residual[worst_idx]

    k = min(max(1, top_k), n_states)
    top_idxs = sortperm(abs_residual; rev=true)[1:k]
    top_entries = String[]
    for idx in top_idxs
        push!(top_entries, "λ=$idx Δ=$(round(parity_residual[idx], digits=4)) ess=$(round(ess_by_state[idx], digits=1))")
    end

    summary = "worst=λ$(worst_idx) Δ=$(round(worst_residual, digits=4)) ess=$(round(ess_by_state[worst_idx], digits=1)) | top=[" * join(top_entries, "; ") * "]"
    return summary, Int(worst_idx), worst_residual, Int.(top_idxs)
end


"""
    validate_stage_b_parity_gate_mode(mode)

Validate the configured Stage B parity gate mode.
"""
function validate_stage_b_parity_gate_mode(mode::Symbol)
    if !(mode in (:raw_max, :support_aware_max))
        throw(ArgumentError("Unsupported Stage B parity gate mode: $(mode). Expected :raw_max or :support_aware_max."))
    end
    return mode
end

"""
    summarize_stage_b_parity_gate(parity_residual, ess_by_state, support_mask,
                                  support_threshold; top_k=3)

Build a compact diagnostics string summarizing both the raw worst state and the
support-aware gate state.
"""
function summarize_stage_b_parity_gate(
    parity_residual::Vector{FT},
    ess_by_state::Vector{FT},
    support_mask::AbstractVector{Bool},
    support_threshold::FT;
    top_k::Int=3,
) where {FT <: AbstractFloat}
    raw_summary, raw_worst_idx, raw_worst_residual, top_idxs = stage_b_parity_diagnostics(
        parity_residual,
        ess_by_state;
        top_k=top_k,
    )

    n_states = length(parity_residual)
    n_supported_states = count(support_mask)
    if n_supported_states == 0
        support_summary = "supported=0/$(n_states) ess>=$(round(support_threshold, digits=1))"
        return raw_summary * " | " * support_summary, raw_worst_idx, raw_worst_residual, top_idxs
    end

    supported_idxs = findall(support_mask)
    supported_abs = abs.(parity_residual[supported_idxs])
    gate_local_idx = argmax(supported_abs)
    gate_idx = supported_idxs[gate_local_idx]
    gate_residual = parity_residual[gate_idx]
    support_summary = "supported=$(n_supported_states)/$(n_states) ess>=$(round(support_threshold, digits=1)) gate_worst=λ$(gate_idx) Δ=$(round(gate_residual, digits=4)) ess=$(round(ess_by_state[gate_idx], digits=1))"
    return raw_summary * " | " * support_summary, raw_worst_idx, raw_worst_residual, top_idxs
end

"""
    stage_b_next_probe_policy(current_probe_steps, base_probe_steps, stats,
                              cooldown_blocks, near_pass_cooldown_blocks,
                              probe_growth_steps, probe_near_pass_scale,
                              probe_max_factor)

Choose the next probe size and cooldown policy after a Stage B decision.
"""
function stage_b_next_probe_policy(
    current_probe_steps::Int,
    base_probe_steps::Int,
    stats::StageBStats,
    cooldown_blocks::Int,
    near_pass_cooldown_blocks::Int,
    probe_growth_steps::Int,
    probe_near_pass_scale::FT,
    probe_max_factor::FT,
) where {FT <: AbstractFloat}
    base_steps = max(1, base_probe_steps)
    current_steps = max(1, current_probe_steps)
    max_steps = max(base_steps, Int(ceil(base_steps * max(one(FT), probe_max_factor))))

    if stats.ready
        return (next_probe_steps=current_steps, cooldown_blocks=0, policy=:passed)
    end

    if stats.near_pass
        near_pass_steps = max(1, Int(round(base_steps * clamp(probe_near_pass_scale, eps(FT), one(FT)))))
        return (
            next_probe_steps=clamp(near_pass_steps, 1, max_steps),
            cooldown_blocks=max(0, near_pass_cooldown_blocks),
            policy=:near_pass,
        )
    end

    # Sampling errors: Grow the probe to increase precision.
    if stats.failure_mode in (:low_support, :split)
        grown_steps = current_steps + probe_growth_steps
        return (
            next_probe_steps=clamp(grown_steps, 1, max_steps),
            cooldown_blocks=max(0, cooldown_blocks),
            policy=:grow_sampling,
        )
    end

    # Bias errors: Do NOT grow the probe. Stay at current size and let Stage A growth
    # (triggered by failed=true) fix the bias.
    if stats.failure_mode in (:raw_parity, :supported_parity, :endpoint_parity)
        return (
            next_probe_steps=current_steps,
            cooldown_blocks=max(0, cooldown_blocks),
            policy=:stay_bias_error,
        )
    end

    return (
        next_probe_steps=base_steps,
        cooldown_blocks=max(0, cooldown_blocks),
        policy=:base,
    )
end

"""
    compute_stage_b_split_parity(energies, log_mixture_denom, F_awh_profile,
                                 coupled_state_idx, decoupled_state_idx, beta,
                                 awh_split_tol_kT, awh_parity_tol_kT; kwargs...)

Evaluate Stage B split-half and MBAR/AWH parity checks for an arbitrary probe
dataset.
"""
function compute_stage_b_split_parity(
    energies::AbstractMatrix{ET},
    log_mixture_denom::AbstractVector{DT},
    F_awh_profile::AbstractVector{ATW},
    coupled_state_idx::Int,
    decoupled_state_idx::Int,
    beta::BT,
    awh_split_tol_kT::ST,
    awh_parity_tol_kT::PTT;
    volumes::AbstractVector{VT}=BT[],
    P0_energy_per_vol::PV=zero(BT),
    parity_diag_top_k::Int=3,
    parity_gate_mode::Symbol=:support_aware_max,
    parity_support_threshold::SP=zero(BT),
    parity_near_pass_factor::NP=BT(2),
    support_allow_missing::Int=0,
) where {
    ET <: AbstractFloat,
    DT <: AbstractFloat,
    ATW <: AbstractFloat,
    BT <: AbstractFloat,
    ST <: AbstractFloat,
    PTT <: AbstractFloat,
    VT <: AbstractFloat,
    PV <: AbstractFloat,
    SP <: AbstractFloat,
    NP <: AbstractFloat,
}
    gate_mode = validate_stage_b_parity_gate_mode(parity_gate_mode)
    n_frames, n_states = size(energies)
    AT = promote_energy_analysis_type(
        energies,
        log_mixture_denom,
        F_awh_profile,
        beta,
        awh_split_tol_kT,
        awh_parity_tol_kT,
        volumes,
        P0_energy_per_vol,
        parity_support_threshold,
        parity_near_pass_factor,
    )
    beta_AT = AT(beta)
    split_tol_AT = AT(awh_split_tol_kT)
    parity_tol_AT = AT(awh_parity_tol_kT)
    P0_AT = AT(P0_energy_per_vol)
    parity_support_threshold_AT = AT(parity_support_threshold)
    parity_near_pass_factor_AT = AT(parity_near_pass_factor)
    if n_states == 0
        throw(ArgumentError("compute_stage_b_split_parity received an empty state dimension."))
    end
    if length(log_mixture_denom) != n_frames
        throw(ArgumentError("compute_stage_b_split_parity expected log_mixture_denom length $n_frames, got $(length(log_mixture_denom))."))
    end
    if !isempty(volumes) && length(volumes) != n_frames
        throw(ArgumentError("compute_stage_b_split_parity expected `volumes` length $n_frames, got $(length(volumes))."))
    end
    if length(F_awh_profile) != n_states
        throw(ArgumentError("compute_stage_b_split_parity expected F_awh_profile length $n_states, got $(length(F_awh_profile))."))
    end
    if coupled_state_idx < 1 || coupled_state_idx > n_states
        throw(ArgumentError("compute_stage_b_split_parity got coupled_state_idx=$coupled_state_idx outside valid range 1:$n_states."))
    end
    if decoupled_state_idx < 1 || decoupled_state_idx > n_states
        throw(ArgumentError("compute_stage_b_split_parity got decoupled_state_idx=$decoupled_state_idx outside valid range 1:$n_states."))
    end
    support_allow_missing_clamped = clamp(support_allow_missing, 0, max(0, n_states - 1))
    required_supported_states = max(1, n_states - support_allow_missing_clamped)

    if n_frames < 2
        return (
            ready = false,
            split_ready = false,
            split_gap = AT(Inf),
            parity_ready = false,
            parity_gap = AT(Inf),
            raw_parity_gap = AT(Inf),
            supported_parity_gap = AT(Inf),
            endpoint_parity_gap = AT(Inf),
            endpoint_parity_ready = false,
            n_frames = n_frames,
            dG_half_1 = AT(NaN),
            dG_half_2 = AT(NaN),
            diagnostics = "insufficient frames for split/parity (n_frames=$n_frames)",
            parity_worst_state_idx = 0,
            parity_worst_state_residual = AT(NaN),
            parity_top_state_idxs = Int[],
            n_supported_states = 0,
            support_threshold = parity_support_threshold_AT,
            support_coverage_ready = false,
            required_supported_states = required_supported_states,
            failure_mode = :not_checked,
            near_pass = false,
        )
    end

    half_1, half_2 = split_half_ranges(n_frames)
    if isempty(half_1) || isempty(half_2)
        return (
            ready = false,
            split_ready = false,
            split_gap = AT(Inf),
            parity_ready = false,
            parity_gap = AT(Inf),
            raw_parity_gap = AT(Inf),
            supported_parity_gap = AT(Inf),
            endpoint_parity_gap = AT(Inf),
            endpoint_parity_ready = false,
            n_frames = n_frames,
            dG_half_1 = AT(NaN),
            dG_half_2 = AT(NaN),
            diagnostics = "empty split ranges (n_frames=$n_frames)",
            parity_worst_state_idx = 0,
            parity_worst_state_residual = AT(NaN),
            parity_top_state_idxs = Int[],
            n_supported_states = 0,
            support_threshold = parity_support_threshold_AT,
            support_coverage_ready = false,
            required_supported_states = required_supported_states,
            failure_mode = :not_checked,
            near_pass = false,
        )
    end

    volumes_half_1 = isempty(volumes) ? AT[] : AT.(volumes[half_1])
    volumes_half_2 = isempty(volumes) ? AT[] : AT.(volumes[half_2])
    volumes_full = isempty(volumes) ? AT[] : AT.(volumes)

    F_half_1 = compute_full_mbar_profile_from_log_mixture_denom(
        energies[half_1, :],
        log_mixture_denom[half_1],
        beta_AT;
        volumes=volumes_half_1,
        P0_energy_per_vol=P0_AT,
    )
    F_half_2 = compute_full_mbar_profile_from_log_mixture_denom(
        energies[half_2, :],
        log_mixture_denom[half_2],
        beta_AT;
        volumes=volumes_half_2,
        P0_energy_per_vol=P0_AT,
    )
    dG_half_1 = F_half_1[coupled_state_idx] - F_half_1[decoupled_state_idx]
    dG_half_2 = F_half_2[coupled_state_idx] - F_half_2[decoupled_state_idx]

    split_gap = abs(dG_half_1 - dG_half_2)
    split_ready = split_gap <= split_tol_AT

    F_mbar_profile = compute_full_mbar_profile_from_log_mixture_denom(
        energies,
        log_mixture_denom,
        beta_AT;
        volumes=volumes_full,
        P0_energy_per_vol=P0_AT,
    )
    F_mbar_aligned = F_mbar_profile .- F_mbar_profile[coupled_state_idx]
    F_awh_aligned = AT.(F_awh_profile) .- AT(F_awh_profile[coupled_state_idx])
    parity_residual = F_mbar_aligned .- F_awh_aligned
    raw_parity_gap = maximum(abs.(parity_residual))

    ess_by_state = compute_state_reweighting_ess_from_log_mixture_denom(
        energies,
        log_mixture_denom,
        beta_AT;
        volumes=volumes_full,
        P0_energy_per_vol=P0_AT,
    )
    support_threshold = max(zero(AT), parity_support_threshold_AT)
    support_mask = ess_by_state .>= support_threshold
    n_supported_states = count(support_mask)
    support_coverage_ready = n_supported_states >= required_supported_states
    supported_parity_gap = n_supported_states > 0 ? maximum(abs.(parity_residual[support_mask])) : AT(Inf)

    endpoint_idxs = unique([coupled_state_idx, decoupled_state_idx])
    endpoint_parity_gap = maximum(abs.(parity_residual[endpoint_idxs]))
    endpoint_parity_ready = endpoint_parity_gap <= parity_tol_AT
    endpoint_worst_local_idx = argmax(abs.(parity_residual[endpoint_idxs]))
    endpoint_worst_idx = endpoint_idxs[endpoint_worst_local_idx]
    endpoint_worst_residual = parity_residual[endpoint_worst_idx]

    parity_gap = if gate_mode == :raw_max
        raw_parity_gap
    else
        max(supported_parity_gap, endpoint_parity_gap)
    end

    supported_parity_ready = n_supported_states > 0 && supported_parity_gap <= parity_tol_AT
    parity_ready_core = if gate_mode == :raw_max
        raw_parity_gap <= parity_tol_AT
    else
        support_coverage_ready && supported_parity_ready
    end
    parity_ready = parity_ready_core && endpoint_parity_ready
    passed_raw_only = gate_mode == :support_aware_max && parity_ready && raw_parity_gap > parity_tol_AT

    diagnostics, parity_worst_state_idx, parity_worst_state_residual, parity_top_state_idxs = summarize_stage_b_parity_gate(
        parity_residual,
        ess_by_state,
        support_mask,
        support_threshold;
        top_k=parity_diag_top_k,
    )
    effective_internal_gap = gate_mode == :raw_max ? raw_parity_gap : supported_parity_gap
    effective_source = endpoint_parity_gap >= effective_internal_gap ? :endpoint : :internal
    diagnostics *= " | endpoint_worst=λ$(endpoint_worst_idx) Δ=$(round(endpoint_worst_residual, digits=4))"
    diagnostics *= " | gate=$(gate_mode) raw=$(round(raw_parity_gap, digits=4)) endpoint=$(round(endpoint_parity_gap, digits=4)) effective=$(round(parity_gap, digits=4)) effective_source=$(effective_source)"

    near_pass_threshold = parity_tol_AT * max(one(AT), parity_near_pass_factor_AT)
    near_pass = !parity_ready && !passed_raw_only && split_ready && endpoint_parity_ready && support_coverage_ready && supported_parity_gap <= near_pass_threshold

    failure_mode = if !split_ready
        :split
    elseif gate_mode == :support_aware_max && !support_coverage_ready
        :low_support
    elseif !endpoint_parity_ready
        :endpoint_parity
    elseif !parity_ready
        gate_mode == :raw_max ? :raw_parity : :supported_parity
    elseif passed_raw_only
        :passed_raw_only
    else
        :passed
    end

    return (
        ready = split_ready && parity_ready && !passed_raw_only,
        split_ready = split_ready,
        split_gap = split_gap,
        parity_ready = parity_ready,
        parity_gap = parity_gap,
        raw_parity_gap = raw_parity_gap,
        supported_parity_gap = supported_parity_gap,
        endpoint_parity_gap = endpoint_parity_gap,
        endpoint_parity_ready = endpoint_parity_ready,
        n_frames = n_frames,
        dG_half_1 = dG_half_1,
        dG_half_2 = dG_half_2,
        diagnostics = diagnostics,
        parity_worst_state_idx = parity_worst_state_idx,
        parity_worst_state_residual = parity_worst_state_residual,
        parity_top_state_idxs = parity_top_state_idxs,
        parity_residual = parity_residual,
        n_supported_states = n_supported_states,
        support_threshold = support_threshold,
        support_coverage_ready = support_coverage_ready,
        required_supported_states = required_supported_states,
        failure_mode = failure_mode,
        near_pass = near_pass,
    )
end



"""
    run_stage_b_probe(awh_sim, sys_base, theta_params, param_names, idxs,
                      coupled_state_idx, decoupled_state_idx, beta, awh_split_tol_kT,
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
    coupled_state_idx::Int,
    decoupled_state_idx::Int,
    beta::BT,
    awh_split_tol_kT::ST,
    awh_parity_tol_kT::PTT;
    md_steps_probe::Int,
    leg_name::String,
    probe_num_md_steps::Union{Nothing, Int}=nothing,
    include_pv::Bool=false,
    P0_energy_per_vol::PV=zero(BT),
    probe_frame_stride::Int=1,
    probe_min_frames::Int=2,
    probe_max_frames::Int=0,
    awh_probe_discard_fraction::Real=0.0,
    parity_gate_mode::Symbol=:support_aware_max,
    parity_support_threshold::SP=zero(BT),
    parity_near_pass_factor::NP=BT(2),
    support_allow_missing::Int=0,
    probe_hotspot_state_idxs::Vector{Int}=Int[],
    probe_hotspot_min_state_occupancy_floor::FT=zero(FT),
    probe_tail_state_idxs::Vector{Int}=Int[],
    probe_tail_min_state_occupancy_floor::FT=zero(FT),
) where {
    FT <: AbstractFloat,
    BT <: AbstractFloat,
    ST <: AbstractFloat,
    PTT <: AbstractFloat,
    PV <: AbstractFloat,
    SP <: AbstractFloat,
    NP <: AbstractFloat,
}
    if md_steps_probe <= 0
        ET = promote_energy_analysis_type(
            beta,
            awh_split_tol_kT,
            awh_parity_tol_kT,
            P0_energy_per_vol,
            parity_support_threshold,
            parity_near_pass_factor,
        )
        @info "Stage B ($(leg_name)) skipped: md_steps_probe <= 0 (md_steps_probe=$md_steps_probe)."
        return stage_b_probe_empty_result(ET, 0, 0)
    end
    return start_stage_b_probe(
        awh_sim,
        sys_base,
        theta_params,
        param_names,
        idxs,
        coupled_state_idx,
        decoupled_state_idx,
        beta,
        awh_split_tol_kT,
        awh_parity_tol_kT;
        md_steps_probe=md_steps_probe,
        leg_name=leg_name,
        probe_num_md_steps=probe_num_md_steps,
        include_pv=include_pv,
        P0_energy_per_vol=P0_energy_per_vol,
        probe_frame_stride=probe_frame_stride,
        probe_min_frames=probe_min_frames,
        probe_max_frames=probe_max_frames,
        awh_probe_discard_fraction=awh_probe_discard_fraction,
        parity_gate_mode=parity_gate_mode,
        parity_support_threshold=parity_support_threshold,
        parity_near_pass_factor=parity_near_pass_factor,
        support_allow_missing=support_allow_missing,
        probe_hotspot_state_idxs=probe_hotspot_state_idxs,
        probe_hotspot_min_state_occupancy_floor=probe_hotspot_min_state_occupancy_floor,
        probe_tail_state_idxs=probe_tail_state_idxs,
        probe_tail_min_state_occupancy_floor=probe_tail_min_state_occupancy_floor,
    ).stats
end
