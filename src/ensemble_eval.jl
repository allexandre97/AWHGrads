# Ensemble evaluation helpers extracted from main_alch.jl

function steps_to_ns(n_steps::Int)
    return FT(ustrip(u"ns", n_steps * Δt))
end

##
function precompute_neighbors(logger, sys)
    sys = Molly.from_device(sys)  
    neighbors = []  
    num_frames = length(logger.active_idx_history)  
    for f_idx in 1:num_frames  
        coords = logger.coords_history[f_idx]  
        vol    = logger.volume_history[f_idx]  
        side   = cbrt(ustrip(vol)) * unit(vol)^(1/3)  
        box    = CubicBoundary(side, side, side)  

        sys = System(sys; coords = coords, boundary = box)  
        nbrs = find_neighbors(sys)  
        push!(neighbors, nbrs)  
    end
    return neighbors  
end

##
function evaluate_ensemble(logger, neighbors, awh_sim_prod, sys_base, 
                           params::Vector{FT}, param_names::Vector{String},
                           atom_idxs, pairwise_idxs, specific_idxs, general_idxs;
                           compute_gradients::Bool=true)  
    num_frames = length(logger.active_idx_history)  
    num_lambda = length(awh_sim_prod.state.partition.λ_atoms)  
    num_params = length(params)  
    
    energies = zeros(FT, num_frames, num_lambda)  
    gradients_raw = compute_gradients ? [zeros(FT, num_frames, num_lambda) for _ in 1:num_params] : nothing
    
    cpu_templates = Vector{System}(undef, num_lambda)  
    for l in 1:num_lambda  
        sys_template_gpu = System(
            sys_base;  
            atoms = awh_sim_prod.state.partition.λ_atoms[l],  
            pairwise_inters = awh_sim_prod.state.state_pairwise_inters[l],  
            specific_inter_lists = awh_sim_prod.state.state_specific_inter_lists[l],  
            general_inters = awh_sim_prod.state.state_general_inters[l]  
        )
        sys_cpu = Molly.from_device(sys_template_gpu)  
        sys_nounits = ustrip(sys_cpu)  
        cpu_templates[l] = Molly.from_device(sys_nounits)  
    end

    n_threads = Threads.nthreads()  
    thread_templates = [deepcopy(cpu_templates) for _ in 1:n_threads]  
    thread_params    = [deepcopy(params) for _ in 1:n_threads] 
    thread_grads     = compute_gradients ? [zeros(FT, num_params) for _ in 1:n_threads] : nothing
    
    if num_frames > 0 && num_lambda > 0  
        dummy_coords = ustrip.(logger.coords_history[1])  
        dummy_vol = ustrip(logger.volume_history[1])  
        dummy_side = cbrt(dummy_vol)  
        dummy_box = CubicBoundary(dummy_side, dummy_side, dummy_side)  
        
        for l in 1:num_lambda  
            evaluate_frame_energy(thread_params[1], thread_templates[1][l], dummy_coords, dummy_box, neighbors[1],   
                                  atom_idxs, pairwise_idxs, specific_idxs, general_idxs)  
        end  
    end
    
    progress = Threads.Atomic{Int}(0)  
    batch_size = n_threads   
    for b in 1:batch_size:num_frames  
        end_idx = min(b + batch_size - 1, num_frames)
        GC.enable(false)
        
        Threads.@threads :static for k in b:end_idx  
            tid = Threads.threadid()  
            coords = ustrip.(logger.coords_history[k])  
            vol = ustrip(logger.volume_history[k])  
            side_length = cbrt(vol)  
            box = CubicBoundary(side_length, side_length, side_length)  
            nbrs = neighbors[k]  
            
            p_local = thread_params[tid]  
            
            for l in 1:num_lambda  
                p_local .= params   
                sys_ref = thread_templates[tid][l]  
                
                if compute_gradients
                    g_local = thread_grads[tid]
                    u_kl = evaluate_frame_gradients(sys_ref, coords, box, nbrs, p_local, g_local,  
                                                    atom_idxs, pairwise_idxs, specific_idxs, general_idxs)  
                
                    for p in 1:num_params  
                        gradients_raw[p][k, l] = g_local[p]  
                    end 
                else
                    u_kl = evaluate_frame_energy(p_local, sys_ref, coords, box, nbrs, 
                                                 atom_idxs, pairwise_idxs, specific_idxs, general_idxs)
                end
                
                energies[k, l] = u_kl  
            end  
        end
        GC.enable(true)
        GC.gc(false)
    end  
    
    gradients = Dict{String, Matrix{FT}}()  
    if compute_gradients
        for (i, name) in enumerate(param_names)  
            gradients[name] = gradients_raw[i]  
        end  
    end
    
    return energies, gradients  
end
