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
    return System(sys; pairwise_inters=new_p_inters)
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
                           compute_gradients::Bool=true)
    if compute_gradients
        return with_compiler_safe_logger() do
            _evaluate_ensemble_impl(
                logger,
                neighbors,
                awh_sim_prod,
                sys_base,
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
        logger,
        neighbors,
        awh_sim_prod,
        sys_base,
        params,
        param_names,
        atom_idxs,
        pairwise_idxs,
        specific_idxs,
        general_idxs;
        compute_gradients=compute_gradients,
    )
end

function _evaluate_ensemble_impl(logger, neighbors, awh_sim_prod, sys_base,
                                 params::Vector{FT}, param_names::Vector{String},
                                 atom_idxs, pairwise_idxs, specific_idxs, general_idxs;
                                 compute_gradients::Bool)
    num_frames = length(logger.active_idx_history)  
    num_lambda = length(awh_sim_prod.state.partition.λ_atoms)  
    num_params = length(params)  
    
    # Store the templates still on GPU/in their original state.
    # We will convert them to CPU for each thread to ensure fresh, thread-safe FFTW plans for PME.
    raw_templates = Vector{System}(undef, num_lambda)
    for l in 1:num_lambda
        raw_templates[l] = System(
            sys_base;
            atoms = awh_sim_prod.state.partition.λ_atoms[l],
            pairwise_inters = awh_sim_prod.state.state_pairwise_inters[l],
            specific_inter_lists = awh_sim_prod.state.state_specific_inter_lists[l],
            general_inters = awh_sim_prod.state.state_general_inters[l]
        )
    end

    n_threads = Threads.nthreads()
    # Create unique CPU templates for each thread to ensure thread-safe FFTW plans.
    # This also applies our interaction fixes (use_neighbors=true for CPU).
    thread_templates = [Vector{System}(undef, num_lambda) for _ in 1:n_threads]
    for tid in 1:n_threads
        for l in 1:num_lambda
            sys_cpu = Molly.from_device(raw_templates[l])
            sys_fixed = _fix_interactions_for_cpu(sys_cpu)
            # Reconstruct with the same nonbonded energy type after stripping units.
            unitless_sys = ustrip(sys_fixed)
            thread_templates[tid][l] = System(unitless_sys; nonbonded_energy_type=sys_fixed.nonbonded_energy_type)
        end
    end

    energy_type = num_lambda > 0 ? Molly.nonbonded_energy_type(thread_templates[1][1]) : FT
    energies = zeros(energy_type, num_frames, num_lambda)
    gradients_raw = compute_gradients ? [zeros(FT, num_frames, num_lambda) for _ in 1:num_params] : nothing

    thread_params = [deepcopy(params) for _ in 1:n_threads] 
    thread_grads  = compute_gradients ? [zeros(FT, num_params) for _ in 1:n_threads] : nothing
    
    if num_frames > 0 && num_lambda > 0  
        dummy_coords = ustrip.(logger.coords_history[1])  
        dummy_vol = ustrip(logger.volume_history[1])  
        if Molly.has_infinite_boundary(sys_base.boundary)
            dummy_box = sys_base.boundary
        else
            dummy_side = cbrt(dummy_vol)  
            dummy_box = CubicBoundary(dummy_side, dummy_side, dummy_side)  
        end
        
        for l in 1:num_lambda  
            evaluate_frame_energy(thread_params[1], thread_templates[1][l], dummy_coords, dummy_box, neighbors[1],   
                                  atom_idxs, pairwise_idxs, specific_idxs, general_idxs)  
        end  
    end
    
    batch_size = n_threads   
    for b in 1:batch_size:num_frames  
        end_idx = min(b + batch_size - 1, num_frames)
        GC.enable(false)
        
        Threads.@threads :static for k in b:end_idx  
            tid = Threads.threadid()  
            coords = ustrip.(logger.coords_history[k])  
            vol = ustrip(logger.volume_history[k])  
            if Molly.has_infinite_boundary(sys_base.boundary)
                box = sys_base.boundary
            else
                side_length = cbrt(vol)  
                box = CubicBoundary(side_length, side_length, side_length)  
            end
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
