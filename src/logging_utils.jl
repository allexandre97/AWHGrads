# Logger/history utilities extracted from main_alch.jl

function get_production_logger(awh_sim_prod::AWHSimulation, leg_name::String)
    logger_main = awh_sim_prod.state.active_sys.loggers.awh_logger
    if !isempty(logger_main.active_idx_history)
        return logger_main
    end

    logger_combined = deepcopy(logger_main)
    empty!(logger_combined.active_idx_history)
    empty!(logger_combined.coords_history)
    empty!(logger_combined.volume_history)
    empty!(logger_combined.potential_energy_history)

    for state_loggers in awh_sim_prod.state.state_loggers
        if hasproperty(state_loggers, :awh_logger)
            lg = state_loggers.awh_logger
            append!(logger_combined.active_idx_history, lg.active_idx_history)
            append!(logger_combined.coords_history, lg.coords_history)
            append!(logger_combined.volume_history, lg.volume_history)
            append!(logger_combined.potential_energy_history, lg.potential_energy_history)
        end
    end

    n_frames = length(logger_combined.active_idx_history)
    if n_frames == 0
        throw(ArgumentError("No production frames were logged for $leg_name leg. Check `production_log_interval`, `md_steps_prod`, and `awh_logger.should_log`."))
    end

    @info "Recovered production logger for $leg_name leg by merging per-state histories (frames = $n_frames)."
    return logger_combined
end


function clear_awh_logger_history!(logger)
    empty!(logger.active_idx_history)
    empty!(logger.coords_history)
    empty!(logger.volume_history)
    empty!(logger.potential_energy_history)
    return nothing
end

function subset_awh_logger_frames(logger, frame_idxs::Vector{Int})
    n_frames = length(logger.active_idx_history)
    if isempty(frame_idxs)
        throw(ArgumentError("subset_awh_logger_frames received an empty frame index set."))
    end
    if n_frames == 0
        throw(ArgumentError("subset_awh_logger_frames cannot operate on an empty logger history."))
    end

    idxs = sort(unique(frame_idxs))
    if any(i -> i < 1 || i > n_frames, idxs)
        throw(ArgumentError("subset_awh_logger_frames got indices outside valid range 1:$(n_frames)."))
    end

    logger_subset = deepcopy(logger)
    clear_awh_logger_history!(logger_subset)
    append!(logger_subset.active_idx_history, logger.active_idx_history[idxs])
    append!(logger_subset.coords_history, logger.coords_history[idxs])
    append!(logger_subset.volume_history, logger.volume_history[idxs])
    append!(logger_subset.potential_energy_history, logger.potential_energy_history[idxs])
    return logger_subset
end

##
function clear_awh_logger_histories!(awh_sim::AWHSimulation)
    if hasproperty(awh_sim.state.active_sys.loggers, :awh_logger)
        clear_awh_logger_history!(awh_sim.state.active_sys.loggers.awh_logger)
    end
    for state_loggers in awh_sim.state.state_loggers
        if hasproperty(state_loggers, :awh_logger)
            clear_awh_logger_history!(state_loggers.awh_logger)
        end
    end
    return nothing
end

function get_awh_active_idx_history(awh_sim)
    if !hasproperty(awh_sim, :state)
        return Int[]
    end

    state = awh_sim.state

    # Prefer Molly's lightweight AWH stats index stream for Stage A readiness.
    stats_idx_field = Symbol("active_\u03bb")
    if hasproperty(state, :stats) && hasproperty(state.stats, stats_idx_field)
        idx_history_stats = Int.(getproperty(state.stats, stats_idx_field))
        if !isempty(idx_history_stats)
            return idx_history_stats
        end
    end

    # Fallback to explicit logger histories.
    if hasproperty(state, :active_sys) &&
       hasproperty(state.active_sys, :loggers) &&
       hasproperty(state.active_sys.loggers, :awh_logger)
        logger_main = state.active_sys.loggers.awh_logger
        if !isempty(logger_main.active_idx_history)
            return copy(logger_main.active_idx_history)
        end
    end

    idx_history = Int[]
    if hasproperty(state, :state_loggers)
        for state_loggers in state.state_loggers
            if hasproperty(state_loggers, :awh_logger)
                append!(idx_history, state_loggers.awh_logger.active_idx_history)
            end
        end
    end
    return idx_history
end
