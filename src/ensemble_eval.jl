"""
    steps_to_ns(n_steps)

Convert an MD step count into nanoseconds using the active module timestep.
"""
function steps_to_ns(n_steps::Int)
    return FT(ustrip(u"ns", n_steps * Δt))
end

function _reconstruct_with_cpu_neighbors(inter)
    if inter isa LennardJonesSoftCoreBeutler
        return LennardJonesSoftCoreBeutler(inter.cutoff, inter.α, true, inter.shortcut, inter.σ_mixing, inter.ϵ_mixing, inter.λ_mixing, inter.scheduler, inter.weight_special)
    elseif inter isa LennardJonesSoftCoreGapsys
        return LennardJonesSoftCoreGapsys(inter.cutoff, inter.α, true, inter.shortcut, inter.σ_mixing, inter.ϵ_mixing, inter.λ_mixing, inter.scheduler, inter.weight_special)
    elseif inter isa CoulombSoftCoreBeutler
        return CoulombSoftCoreBeutler(inter.cutoff, inter.α, true, inter.σ_mixing, inter.ϵ_mixing, inter.λ_mixing, inter.scheduler, inter.weight_special, inter.coulomb_const)
    elseif inter isa CoulombSoftCoreGapsys
        return CoulombSoftCoreGapsys(inter.cutoff, inter.α, inter.σQ, true, inter.λ_mixing, inter.scheduler, inter.weight_special, inter.coulomb_const)
    elseif inter isa CoulombSoftCoreBeutlerReactionField
        return CoulombSoftCoreBeutlerReactionField(inter.dist_cutoff, inter.solvent_dielectric, inter.α, true, inter.σ_mixing, inter.ϵ_mixing, inter.λ_mixing, inter.scheduler, inter.weight_special, inter.coulomb_const)
    elseif inter isa CoulombSoftCoreGapsysReactionField
        return CoulombSoftCoreGapsysReactionField(inter.dist_cutoff, inter.solvent_dielectric, inter.α, inter.σQ, true, inter.λ_mixing, inter.scheduler, inter.weight_special, inter.coulomb_const)
    elseif inter isa CoulombSoftCoreBeutlerEwald
        return CoulombSoftCoreBeutlerEwald(inter.dist_cutoff, inter.error_tol, inter.α, true, inter.σ_mixing, inter.ϵ_mixing, inter.λ_mixing, inter.scheduler, inter.weight_special, inter.coulomb_const, inter.α_ewald, inter.approximate_erfc)
    elseif inter isa CoulombSoftCoreGapsysEwald
        return CoulombSoftCoreGapsysEwald(inter.dist_cutoff, inter.error_tol, inter.α, inter.σQ, true, inter.λ_mixing, inter.scheduler, inter.weight_special, inter.coulomb_const, inter.α_ewald, inter.approximate_erfc)
    end
    return inter
end

function _fix_interactions_for_cpu(sys::System)
    new_p_inters = map(_reconstruct_with_cpu_neighbors, sys.pairwise_inters)
    return System(sys; pairwise_inters=new_p_inters, loggers=NamedTuple())
end

"""
    precompute_neighbors(logger, sys)

Replay the logged trajectory and cache a neighbor list for every saved frame.
This keeps the later energy/gradient evaluation pass deterministic and avoids
rebuilding neighbor lists inside the tight inner loops.
"""
function precompute_neighbors(logger, sys)
    sys = Molly.from_device(sys)  
    neighbors = []  
    num_frames = length(logger.active_idx_history)  

    nf = sys.neighbor_finder
    for f_idx in 1:num_frames
        coords = logger.coords_history[f_idx]
        vol    = logger.volume_history[f_idx]
        if Molly.has_infinite_boundary(sys.boundary)
            box = sys.boundary
        else
            side = cbrt(ustrip(vol)) * unit(vol)^(1/3)
            box  = CubicBoundary(side, side, side)
        end

        sys_temp = System(sys; coords = coords, boundary = box, neighbor_finder = nf)
        nbrs = find_neighbors(sys_temp)  
        push!(neighbors, nbrs)  
    end
    return neighbors  
end

function normalized_ensemble_eval_config(cfg::EnsembleEvalConfig)
    threads = max(1, min(cfg.threads, Threads.nthreads()))
    lambda_tile = max(1, cfg.lambda_tile)
    schedule = cfg.schedule
    if schedule ∉ (:dynamic, :static)
        throw(ArgumentError("EnsembleEvalConfig.schedule must be :dynamic or :static, got `$schedule`."))
    end
    progress_interval_seconds = cfg.progress_interval_seconds
    if !(progress_interval_seconds > 0)
        throw(ArgumentError("EnsembleEvalConfig.progress_interval_seconds must be > 0, got `$progress_interval_seconds`."))
    end
    return EnsembleEvalConfig(
        threads=threads,
        lambda_tile=lambda_tile,
        schedule=schedule,
        cache_unitless_frames=cfg.cache_unitless_frames,
        cache_unitless_templates=cfg.cache_unitless_templates,
        progress=cfg.progress,
        progress_interval_seconds=progress_interval_seconds,
    )
end

Base.@kwdef mutable struct EnsembleEvalProgressState
    enabled::Bool = false
    mode::Symbol = :energy
    total_frame_states::Int = 0
    total_tiles::Int = 0
    interval_ns::Int = 30_000_000_000
    start_ns::UInt64 = time_ns()
    last_report_ns::UInt64 = start_ns
    completed_frame_states::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    completed_tiles::Int = 0
    current_tile_start::Int = 0
    current_tile_stop::Int = 0
    current_tile_total_frame_states::Int = 0
    current_tile_completed_at_start::Int = 0
    lock::ReentrantLock = ReentrantLock()
end

function _ensemble_eval_progress_state(eval_cfg::EnsembleEvalConfig,
                                       compute_gradients::Bool,
                                       num_frames::Int,
                                       num_lambda::Int)
    now_ns = time_ns()
    total_frame_states = num_frames * num_lambda
    total_tiles = num_lambda == 0 ? 0 : cld(num_lambda, max(1, eval_cfg.lambda_tile))
    return EnsembleEvalProgressState(
        enabled=eval_cfg.progress && total_frame_states > 0,
        mode=compute_gradients ? :gradients : :energies,
        total_frame_states=total_frame_states,
        total_tiles=total_tiles,
        interval_ns=round(Int, eval_cfg.progress_interval_seconds * 1.0e9),
        start_ns=now_ns,
        last_report_ns=now_ns,
    )
end

function _ensemble_eval_emit_progress!(state::EnsembleEvalProgressState, reason::AbstractString; force::Bool=false)
    state.enabled || return nothing

    now_ns = time_ns()
    if !force && Int(now_ns - state.last_report_ns) < state.interval_ns
        return nothing
    end

    lock(state.lock)
    try
        now_ns = time_ns()
        completed = state.completed_frame_states[]
        if !force && completed < state.total_frame_states && Int(now_ns - state.last_report_ns) < state.interval_ns
            return nothing
        end

        elapsed_s = max(Float64(now_ns - state.start_ns) / 1.0e9, 0.0)
        completed_pct = state.total_frame_states == 0 ? 100.0 : 100.0 * completed / state.total_frame_states
        rate = elapsed_s > 0 ? completed / elapsed_s : 0.0
        remaining = max(0, state.total_frame_states - completed)
        eta_s = rate > 0 ? remaining / rate : Inf
        tile_completed = max(0, completed - state.current_tile_completed_at_start)
        tile_total = max(0, state.current_tile_total_frame_states)
        tile_pct = tile_total == 0 ? 100.0 : 100.0 * min(tile_completed, tile_total) / tile_total
        eta_msg = isfinite(eta_s) ? string(round(eta_s, digits=1)) : "n/a"

        println(
            "  [ensemble-eval] ",
            reason,
            " | mode=", state.mode,
            " | λ_tile=", state.current_tile_start, ":", state.current_tile_stop,
            " | tiles=", state.completed_tiles, "/", state.total_tiles,
            " | current_tile=", min(tile_completed, tile_total), "/", tile_total,
            " (", round(tile_pct, digits=1), "%)",
            " | total=", completed, "/", state.total_frame_states,
            " (", round(completed_pct, digits=1), "%)",
            " | elapsed_s=", round(elapsed_s, digits=1),
            " | rate=", round(rate, digits=1), " frame-states/s",
            " | eta_s=", eta_msg,
        )
        flush(stdout)
        flush(stderr)
        state.last_report_ns = now_ns
    finally
        unlock(state.lock)
    end

    return nothing
end

function _ensemble_eval_begin_tile!(state::EnsembleEvalProgressState,
                                    tile_indices,
                                    num_frames::Int)
    state.enabled || return nothing
    lock(state.lock)
    try
        state.current_tile_start = first(tile_indices)
        state.current_tile_stop = last(tile_indices)
        state.current_tile_total_frame_states = num_frames * length(tile_indices)
        state.current_tile_completed_at_start = state.completed_frame_states[]
    finally
        unlock(state.lock)
    end
    return _ensemble_eval_emit_progress!(state, "tile start"; force=true)
end

function _ensemble_eval_finish_tile!(state::EnsembleEvalProgressState)
    state.enabled || return nothing
    lock(state.lock)
    try
        state.completed_tiles += 1
    finally
        unlock(state.lock)
    end
    return _ensemble_eval_emit_progress!(state, "tile done"; force=true)
end

function _ensemble_eval_heartbeat!(state::EnsembleEvalProgressState, delta::Int)
    state.enabled || return nothing
    Threads.atomic_add!(state.completed_frame_states, delta)
    return _ensemble_eval_emit_progress!(state, "heartbeat")
end

function build_ensemble_eval_frame_cache(logger)
    num_frames = length(logger.active_idx_history)
    if num_frames == 0
        return (coords=Vector{Any}(), volumes=Float64[])
    end

    coord_type = typeof(ustrip.(logger.coords_history[1]))
    volume_type = typeof(ustrip(logger.volume_history[1]))
    coords = Vector{coord_type}(undef, num_frames)
    volumes = Vector{volume_type}(undef, num_frames)
    for frame_idx in 1:num_frames
        coords[frame_idx] = ustrip.(logger.coords_history[frame_idx])
        volumes[frame_idx] = ustrip(logger.volume_history[frame_idx])
    end
    return (coords=coords, volumes=volumes)
end

function prepare_enzyme_ready_system(sys::System)
    sys_cpu = Molly.from_device(sys)
    sys_fixed = _fix_interactions_for_cpu(sys_cpu)
    return ustrip(sys_fixed)
end

function validate_enzyme_ready_system(sys::System)
    prepared = prepare_enzyme_ready_system(sys)
    if typeof(sys) != typeof(prepared)
        throw(ArgumentError("Ensemble-eval templates passed to Enzyme must already be CPU and unitless. Build them with Molly.from_device(...) |> _fix_interactions_for_cpu |> ustrip before differentiation."))
    end
    return sys
end

function build_unitless_lambda_templates(
    awh_sim_prod,
    sys_base,
    lambda_indices=1:length(awh_sim_prod.state.partition.λ_atoms),
)
    isempty(lambda_indices) && return Any[]

    first_idx = first(lambda_indices)
    first_template = begin
        raw_template = System(
            sys_base;
            atoms=awh_sim_prod.state.partition.λ_atoms[first_idx],
            pairwise_inters=awh_sim_prod.state.state_pairwise_inters[first_idx],
            specific_inter_lists=awh_sim_prod.state.state_specific_inter_lists[first_idx],
            general_inters=awh_sim_prod.state.state_general_inters[first_idx],
        )
        validate_enzyme_ready_system(prepare_enzyme_ready_system(raw_template))
    end

    template_type = typeof(first_template)
    templates = Vector{template_type}(undef, length(lambda_indices))
    templates[1] = first_template

    for (i, lambda_idx) in enumerate(lambda_indices)
        i == 1 && continue
        raw_template = System(
            sys_base;
            atoms=awh_sim_prod.state.partition.λ_atoms[lambda_idx],
            pairwise_inters=awh_sim_prod.state.state_pairwise_inters[lambda_idx],
            specific_inter_lists=awh_sim_prod.state.state_specific_inter_lists[lambda_idx],
            general_inters=awh_sim_prod.state.state_general_inters[lambda_idx],
        )
        templates[i] = validate_enzyme_ready_system(prepare_enzyme_ready_system(raw_template))
    end

    return templates
end

function build_ensemble_eval_cache(
    logger,
    neighbors,
    awh_sim_prod,
    sys_base,
    eval_cfg::EnsembleEvalConfig=EnsembleEvalConfig(),
)
    cfg = normalized_ensemble_eval_config(eval_cfg)
    frame_cache = cfg.cache_unitless_frames ? build_ensemble_eval_frame_cache(logger) : nothing
    template_cache = cfg.cache_unitless_templates ? build_unitless_lambda_templates(awh_sim_prod, sys_base) : nothing
    active_idx_history = Int.(logger.active_idx_history)
    num_lambda = isnothing(template_cache) ? length(awh_sim_prod.state.partition.λ_atoms) : length(template_cache)
    energy_type = if !isempty(logger.potential_energy_history)
        typeof(ustrip(first(logger.potential_energy_history)))
    elseif !isnothing(template_cache) && !isempty(template_cache)
        Molly.nonbonded_energy_type(first(template_cache))
    else
        typeof(first(awh_sim_prod.state.f))
    end
    return (
        logger=logger,
        neighbors=neighbors,
        awh_sim_prod=awh_sim_prod,
        sys_base=sys_base,
        frame_cache=frame_cache,
        template_cache=template_cache,
        has_infinite_boundary=Molly.has_infinite_boundary(sys_base.boundary),
        infinite_boundary=sys_base.boundary,
        active_idx_history=active_idx_history,
        frame_count=length(active_idx_history),
        num_lambda=num_lambda,
        energy_type=energy_type,
        config=cfg,
    )
end

"""
    compact_ensemble_eval_cache(eval_cache)

Drop source objects that are redundant once the unitless frame/template caches
have been materialized. This keeps optimization artifacts from retaining the
full production logger and frozen AWH simulation graph alongside the replay
cache.
"""
function compact_ensemble_eval_cache(eval_cache)
    logger = isnothing(eval_cache.frame_cache) ? eval_cache.logger : nothing
    awh_sim_prod = isnothing(eval_cache.template_cache) ? eval_cache.awh_sim_prod : nothing
    sys_base = isnothing(eval_cache.template_cache) ? eval_cache.sys_base : nothing
    return (
        logger=logger,
        neighbors=eval_cache.neighbors,
        awh_sim_prod=awh_sim_prod,
        sys_base=sys_base,
        frame_cache=eval_cache.frame_cache,
        template_cache=eval_cache.template_cache,
        has_infinite_boundary=eval_cache.has_infinite_boundary,
        infinite_boundary=eval_cache.infinite_boundary,
        active_idx_history=eval_cache.active_idx_history,
        frame_count=eval_cache.frame_count,
        num_lambda=eval_cache.num_lambda,
        energy_type=eval_cache.energy_type,
        config=eval_cache.config,
    )
end

function ensemble_eval_frame_state(eval_cache, frame_idx::Int)
    if !isnothing(eval_cache.frame_cache)
        coords = eval_cache.frame_cache.coords[frame_idx]
        if eval_cache.has_infinite_boundary
            return coords, eval_cache.infinite_boundary
        end
        vol = eval_cache.frame_cache.volumes[frame_idx]
    else
        isnothing(eval_cache.logger) && throw(ArgumentError("Ensemble-eval cache has no frame cache and no source logger; cannot reconstruct replay frame state."))
        coords = ustrip.(eval_cache.logger.coords_history[frame_idx])
        if eval_cache.has_infinite_boundary
            return coords, eval_cache.infinite_boundary
        end
        vol = ustrip(eval_cache.logger.volume_history[frame_idx])
    end

    side_length = cbrt(vol)
    return coords, CubicBoundary(side_length, side_length, side_length)
end

function ensemble_eval_template_tile(eval_cache, tile_indices)
    if !isnothing(eval_cache.template_cache)
        return eval_cache.template_cache[tile_indices]
    end
    isnothing(eval_cache.awh_sim_prod) && throw(ArgumentError("Ensemble-eval cache has no template cache and no source AWH simulation; cannot rebuild λ-state templates."))
    return build_unitless_lambda_templates(eval_cache.awh_sim_prod, eval_cache.sys_base, tile_indices)
end

function run_ensemble_eval_workers!(
    op::Function,
    frame_indices::UnitRange{Int},
    n_workers::Int,
    schedule::Symbol,
)
    isempty(frame_indices) && return nothing
    worker_count = max(1, min(n_workers, length(frame_indices)))
    if worker_count == 1
        for frame_idx in frame_indices
            op(frame_idx, 1)
        end
        return nothing
    end

    tasks = Task[]
    if schedule == :static
        chunk_size = cld(length(frame_indices), worker_count)
        first_idx = first(frame_indices)
        last_idx = last(frame_indices)
        for worker in 1:worker_count
            chunk_start = first_idx + (worker - 1) * chunk_size
            chunk_end = min(last_idx, chunk_start + chunk_size - 1)
            chunk_start > chunk_end && continue
            push!(tasks, Threads.@spawn begin
                for frame_idx in chunk_start:chunk_end
                    op(frame_idx, worker)
                end
            end)
        end
    else
        next_offset = Threads.Atomic{Int}(0)
        first_idx = first(frame_indices)
        n_frames = length(frame_indices)
        for worker in 1:worker_count
            push!(tasks, Threads.@spawn begin
                while true
                    offset = Threads.atomic_add!(next_offset, 1)
                    offset >= n_frames && break
                    frame_idx = first_idx + offset
                    op(frame_idx, worker)
                end
            end)
        end
    end

    foreach(wait, tasks)
    return nothing
end


function _run_with_gc_disabled(f::Function; full_sweep::Bool=false)
    gc_was_enabled = GC.enable(false)
    try
        return f()
    finally
        GC.enable(gc_was_enabled)
        GC.gc(full_sweep)
    end
end

@inline function _ensemble_eval_batch_size(compute_gradients::Bool, worker_count::Int, num_frames::Int)
    if !compute_gradients
        return max(1, num_frames)
    end
    return max(1, min(num_frames, worker_count))
end

"""
    evaluate_ensemble(logger, neighbors, awh_sim_prod, sys_base, params, param_names,
                      atom_idxs, pairwise_idxs, specific_idxs, general_idxs;
                      compute_gradients=true)

Evaluate the stored production frames under every λ state for a candidate
parameter vector. When `compute_gradients=true`, the result also includes a
parameter-name keyed dictionary of frame-by-frame gradients.
"""
function evaluate_ensemble(logger, neighbors, awh_sim_prod, sys_base,
                           params::Vector{FT}, param_names::Vector{String},
                           atom_idxs, pairwise_idxs, specific_idxs, general_idxs;
                           compute_gradients::Bool=true,
                           eval_cfg::EnsembleEvalConfig=EnsembleEvalConfig())
    eval_cache = build_ensemble_eval_cache(logger, neighbors, awh_sim_prod, sys_base, eval_cfg)
    return evaluate_ensemble(
        eval_cache,
        params,
        param_names,
        atom_idxs,
        pairwise_idxs,
        specific_idxs,
        general_idxs;
        compute_gradients=compute_gradients,
    )
end

function evaluate_ensemble(eval_cache,
                           params::Vector{FT}, param_names::Vector{String},
                           atom_idxs, pairwise_idxs, specific_idxs, general_idxs;
                           compute_gradients::Bool=true)
    if compute_gradients
        return with_compiler_safe_logger() do
            _evaluate_ensemble_impl(
                eval_cache,
                params,
                param_names,
                atom_idxs,
                pairwise_idxs,
                specific_idxs,
                general_idxs;
                compute_gradients=compute_gradients,
            )
        end
    end

    return _evaluate_ensemble_impl(
        eval_cache,
        params,
        param_names,
        atom_idxs,
        pairwise_idxs,
        specific_idxs,
        general_idxs;
        compute_gradients=compute_gradients,
    )
end

function _evaluate_ensemble_impl(eval_cache,
                                 params::Vector{FT}, param_names::Vector{String},
                                 atom_idxs, pairwise_idxs, specific_idxs, general_idxs;
                                 compute_gradients::Bool)
    neighbors = eval_cache.neighbors
    num_frames = eval_cache.frame_count
    num_lambda = eval_cache.num_lambda
    num_params = length(params)  

    energy_type = eval_cache.energy_type
    energies = zeros(energy_type, num_frames, num_lambda)
    gradients_raw = compute_gradients ? [zeros(FT, num_frames, num_lambda) for _ in 1:num_params] : nothing

    if num_frames == 0 || num_lambda == 0
        return energies, Dict{String, Matrix{FT}}()
    end

    worker_count = max(1, min(eval_cache.config.threads, num_frames))
    lambda_tile = min(num_lambda, max(1, eval_cache.config.lambda_tile))
    thread_params = [deepcopy(params) for _ in 1:worker_count]
    thread_grads = compute_gradients ? [zeros(FT, num_params) for _ in 1:worker_count] : nothing
    progress_state = _ensemble_eval_progress_state(eval_cache.config, compute_gradients, num_frames, num_lambda)

    dummy_coords, dummy_box = ensemble_eval_frame_state(eval_cache, 1)
    dummy_nbrs = neighbors[1]
    warmup_tile = 1:min(num_lambda, lambda_tile)
    warmup_templates = ensemble_eval_template_tile(eval_cache, warmup_tile)
    p_warmup = thread_params[1]
    for sys_ref in warmup_templates
        p_warmup .= params
        evaluate_frame_energy(
            p_warmup,
            sys_ref,
            dummy_coords,
            dummy_box,
            dummy_nbrs,
            atom_idxs,
            pairwise_idxs,
            specific_idxs,
            general_idxs,
        )
    end

    for lambda_start in 1:lambda_tile:num_lambda
        lambda_stop = min(num_lambda, lambda_start + lambda_tile - 1)
        tile_indices = lambda_start:lambda_stop
        source_templates = ensemble_eval_template_tile(eval_cache, tile_indices)
        worker_templates = compute_gradients ? [deepcopy(source_templates) for _ in 1:worker_count] : nothing
        batch_size = _ensemble_eval_batch_size(compute_gradients, worker_count, num_frames)
        _ensemble_eval_begin_tile!(progress_state, tile_indices, num_frames)

        for frame_start in 1:batch_size:num_frames
            frame_stop = min(num_frames, frame_start + batch_size - 1)
            frame_batch = frame_start:frame_stop

            batch_runner = () -> run_ensemble_eval_workers!(
                frame_batch,
                worker_count,
                eval_cache.config.schedule,
            ) do frame_idx, worker
                coords, box = ensemble_eval_frame_state(eval_cache, frame_idx)
                nbrs = neighbors[frame_idx]
                p_local = thread_params[worker]

                for (local_idx, lambda_idx) in enumerate(tile_indices)
                    p_local .= params
                    sys_ref = if compute_gradients
                        worker_templates[worker][local_idx]
                    else
                        source_templates[local_idx]
                    end

                    if compute_gradients
                        g_local = thread_grads[worker]
                        u_kl = evaluate_frame_gradients(
                            sys_ref,
                            coords,
                            box,
                            nbrs,
                            p_local,
                            g_local,
                            atom_idxs,
                            pairwise_idxs,
                            specific_idxs,
                            general_idxs,
                        )
                        @inbounds for p in 1:num_params
                            gradients_raw[p][frame_idx, lambda_idx] = g_local[p]
                        end
                    else
                        u_kl = evaluate_frame_energy(
                            p_local,
                            sys_ref,
                            coords,
                            box,
                            nbrs,
                            atom_idxs,
                            pairwise_idxs,
                            specific_idxs,
                            general_idxs,
                        )
                    end

                    energies[frame_idx, lambda_idx] = u_kl
                end

                _ensemble_eval_heartbeat!(progress_state, length(tile_indices))
            end

            if compute_gradients
                _run_with_gc_disabled(batch_runner)
            else
                batch_runner()
            end
        end

        _ensemble_eval_finish_tile!(progress_state)
        if compute_gradients
            GC.gc(true)
        end
    end
    
    gradients = Dict{String, Matrix{FT}}()  
    if compute_gradients
        for (i, name) in enumerate(param_names)  
            gradients[name] = gradients_raw[i]  
        end  
    end
    
    return energies, gradients  
end
