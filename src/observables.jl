const G_PER_ML_PER_G_PER_MOL_NM3 = Float64(1.0e21 / ustrip(Unitful.Na))

"""
    MassDensityObservable()

Callable observable that returns the replayed system mass density in `g/mL` by
default. The conversion factor is configurable so callers can choose other
output units if needed. The offline replay path passes a unitless CPU system.
The observable accepts either a unitful or a unitless `Molly.System`; the
"""
Base.@kwdef struct MassDensityObservable{FT <: AbstractFloat}
    conversion_factor::FT = FT(G_PER_ML_PER_G_PER_MOL_NM3)
end

@inline _observable_mass_value(value) = value isa Unitful.AbstractQuantity ? Float64(ustrip(u"g/mol", value)) : Float64(value)
@inline _observable_volume_value(value) = value isa Unitful.AbstractQuantity ? Float64(ustrip(u"nm^3", value)) : Float64(value)

function (obs::MassDensityObservable)(sys::System)
    Molly.has_infinite_boundary(sys.boundary) &&
        throw(ArgumentError("MassDensityObservable requires a finite periodic boundary."))
    volume_nm3 = _observable_volume_value(volume(sys.boundary))
    volume_nm3 > 0 || throw(ArgumentError("MassDensityObservable requires a positive box volume."))
    total_mass = _observable_mass_value(sys.total_mass)
    return obs.conversion_factor * total_mass / volume_nm3
end

@inline function _coerce_observable_value(value, ::Type{FT}) where {FT <: AbstractFloat}
    value isa Real || throw(ArgumentError("Observable callables must return a scalar Real, got $(typeof(value))."))
    coerced = FT(value)
    isfinite(coerced) || throw(ArgumentError("Observable callables must return finite values, got `$value`."))
    return coerced
end

function resolve_target_state(
    selector,
    leg_name::Symbol,
    schedule::ResolvedLegStateSchedule,
)
    if selector isa Integer
        idx = Int(selector)
        1 <= idx <= length(schedule.lambda) ||
            throw(ArgumentError("Observable target on leg `$leg_name` requested state index $idx, but the leg has $(length(schedule.lambda)) states."))
        return idx, Symbol("state_$idx")
    elseif selector isa Symbol
        if selector == :coupled
            return schedule.coupled_state_idx, :coupled
        elseif selector == :decoupled
            return schedule.decoupled_state_idx, :decoupled
        end
    end

    throw(ArgumentError("Observable target on leg `$leg_name` uses unsupported state selector `$selector`. Use an integer state index or one of `:coupled` / `:decoupled`."))
end

function resolve_training_targets(
    sim_cfg::SimulationConfig,
    cycle_cfg::ThermodynamicCycleConfig,
    state_schedules_by_leg::Dict{Symbol, <:ResolvedLegStateSchedule},
)
    raw_targets = sim_cfg.targets
    if isnothing(raw_targets)
        return AbstractTrainingTarget[
            ResolvedCycleFreeEnergyTarget(
                :cycle_free_energy,
                Float64(cycle_cfg.target_dG_kcal_mol),
                nothing,
                1.0,
            ),
        ]
    end

    isempty(raw_targets) &&
        throw(ArgumentError("SimulationConfig.targets must contain at least one target when provided explicitly."))

    seen_names = Set{Symbol}()
    resolved = AbstractTrainingTarget[]

    for raw_target in raw_targets
        raw_target isa AbstractTrainingTarget ||
            throw(ArgumentError("SimulationConfig.targets may only contain `AbstractTrainingTarget` values, got $(typeof(raw_target))."))

        target_name = getfield(raw_target, :name)
        target_name in seen_names &&
            throw(ArgumentError("Duplicate training target name `$target_name`."))
        push!(seen_names, target_name)

        target_weight = Float64(getfield(raw_target, :weight))
        isfinite(target_weight) && target_weight > 0 ||
            throw(ArgumentError("Training target `$target_name` must have a finite positive weight."))

        if raw_target isa CycleFreeEnergyTarget
            target_value = isnothing(raw_target.target_dG_kcal_mol) ?
                cycle_cfg.target_dG_kcal_mol :
                raw_target.target_dG_kcal_mol
            isfinite(target_value) ||
                throw(ArgumentError("Training target `$target_name` has a non-finite cycle free-energy target."))
            target_tolerance = isnothing(raw_target.tolerance_kcal_mol) ? nothing : Float64(raw_target.tolerance_kcal_mol)
            if !isnothing(target_tolerance)
                isfinite(target_tolerance) && target_tolerance > 0 ||
                    throw(ArgumentError("Training target `$target_name` must have a finite positive tolerance when provided."))
            end
            push!(
                resolved,
                ResolvedCycleFreeEnergyTarget(
                    raw_target.name,
                    Float64(target_value),
                    target_tolerance,
                    target_weight,
                ),
            )
        elseif raw_target isa StateObservableTarget
            haskey(state_schedules_by_leg, raw_target.leg) ||
                throw(ArgumentError("Observable target `$target_name` references unknown leg `$(raw_target.leg)`."))

            state_idx, state_label = resolve_target_state(
                raw_target.state,
                raw_target.leg,
                state_schedules_by_leg[raw_target.leg],
            )
            target_value = Float64(raw_target.target_value)
            isfinite(target_value) ||
                throw(ArgumentError("Observable target `$target_name` has a non-finite target value."))
            target_tolerance = isnothing(raw_target.tolerance) ? nothing : Float64(raw_target.tolerance)
            if !isnothing(target_tolerance)
                isfinite(target_tolerance) && target_tolerance > 0 ||
                    throw(ArgumentError("Observable target `$target_name` must have a finite positive tolerance when provided."))
            end

            push!(
                resolved,
                ResolvedStateObservableTarget(
                    raw_target.name,
                    raw_target.leg,
                    state_idx,
                    state_label,
                    raw_target.observable,
                    target_value,
                    target_tolerance,
                    target_weight,
                    String(raw_target.unit_label),
                ),
            )
        else
            throw(ArgumentError("Unsupported training target type $(typeof(raw_target))."))
        end
    end

    return resolved
end

function target_reference_value(
    target::ResolvedCycleFreeEnergyTarget,
    beta_val::BT,
    ::Type{AT},
) where {BT <: AbstractFloat, AT <: AbstractFloat}
    return AT(target.target_dG_kcal_mol) * AT(4.184) * AT(beta_val)
end

function target_reference_value(
    target::ResolvedStateObservableTarget,
    beta_val,
    ::Type{AT},
) where {AT <: AbstractFloat}
    return AT(target.target_value)
end

@inline target_weight(target, ::Type{AT}) where {AT <: AbstractFloat} = AT(target.weight)
@inline target_unit_label(::ResolvedCycleFreeEnergyTarget) = "kT"
@inline target_unit_label(target::ResolvedStateObservableTarget) = target.unit_label
@inline target_configured_tolerance(target::ResolvedCycleFreeEnergyTarget) = target.tolerance_kcal_mol
@inline target_configured_tolerance(target::ResolvedStateObservableTarget) = target.tolerance

function ensemble_eval_template(eval_cache, state_idx::Int)
    1 <= state_idx <= length(eval_cache.awh_sim_prod.state.partition.λ_atoms) ||
        throw(ArgumentError("Requested replay state index $state_idx outside valid range 1:$(length(eval_cache.awh_sim_prod.state.partition.λ_atoms))."))
    if !isnothing(eval_cache.template_cache)
        return eval_cache.template_cache[state_idx]
    end
    return only(build_unitless_lambda_templates(eval_cache.awh_sim_prod, eval_cache.sys_base, [state_idx]))
end

"""
    evaluate_frame_observable(params, observable, sys_ref, coords_nounits, box_nounits,
                              neighbors, atom_idxs, pairwise_idxs, specific_idxs,
                              general_idxs)

Rebuild a single λ-state system with the candidate parameter vector and return
the scalar observable value for one stored frame.
"""
function evaluate_frame_observable(
    params::Vector{FT},
    observable,
    sys_ref::System{D, AT, FT},
    coords_nounits,
    box_nounits,
    neighbors,
    atom_idxs,
    pairwise_idxs,
    specific_idxs,
    general_idxs,
) where {D, AT, FT <: AbstractFloat}
    new_atoms = inject_atom_parameters(sys_ref.atoms, params, atom_idxs)
    new_pairwise = _update_pairwise(sys_ref.pairwise_inters, params, pairwise_idxs)
    new_specific = _update_specific(sys_ref.specific_inter_lists, params, specific_idxs)
    new_general = _update_general(sys_ref.general_inters, params, general_idxs)

    sys_final = _rebuild_system_like(
        sys_ref,
        new_atoms,
        coords_nounits,
        box_nounits,
        new_pairwise,
        new_specific,
        new_general,
    )

    return _coerce_observable_value(observable(sys_final), FT)
end

"""
    evaluate_frame_observable_gradients(observable, sys_ref, coords_nounits,
                                        box_nounits, neighbors, params,
                                        grads_enzyme, atom_idxs, pairwise_idxs,
                                        specific_idxs, general_idxs)

Differentiate `evaluate_frame_observable` with respect to the parameter vector
using Enzyme and return the primal observable value.
"""
function evaluate_frame_observable_gradients(
    observable,
    sys_ref::System{D, AT, FT},
    coords_nounits,
    box_nounits,
    neighbors,
    params::Vector{FT},
    grads_enzyme::Vector{FT},
    atom_idxs,
    pairwise_idxs,
    specific_idxs,
    general_idxs,
) where {D, AT, FT <: AbstractFloat}
    sys_grad = _enzyme_gradient_system(sys_ref)
    return with_compiler_safe_logger() do
        fill!(grads_enzyme, zero(FT))

        result = autodiff(
            set_runtime_activity(ReverseWithPrimal),
            evaluate_frame_observable,
            Active,
            Duplicated(params, grads_enzyme),
            Const(observable),
            Const(sys_grad),
            Const(coords_nounits),
            Const(box_nounits),
            Const(neighbors),
            Const(atom_idxs),
            Const(pairwise_idxs),
            Const(specific_idxs),
            Const(general_idxs),
        )

        return result[2]
    end
end

function evaluate_state_observable(
    eval_cache,
    params::Vector{FT},
    param_names::Vector{String},
    observable,
    state_idx::Int,
    atom_idxs,
    pairwise_idxs,
    specific_idxs,
    general_idxs;
    compute_gradients::Bool=true,
) where {FT <: AbstractFloat}
    logger = eval_cache.logger
    neighbors = eval_cache.neighbors
    num_frames = length(logger.active_idx_history)
    num_params = length(params)

    values = zeros(FT, num_frames)
    gradient_rows = compute_gradients ? [zeros(FT, num_frames) for _ in 1:num_params] : nothing
    num_frames == 0 && return values, Dict{String, Vector{FT}}()

    worker_count = max(1, min(eval_cache.config.threads, num_frames))
    thread_params = [deepcopy(params) for _ in 1:worker_count]
    thread_grads = compute_gradients ? [zeros(FT, num_params) for _ in 1:worker_count] : nothing
    base_template = ensemble_eval_template(eval_cache, state_idx)
    worker_templates = [deepcopy(base_template) for _ in 1:worker_count]

    if compute_gradients
        coords, box = ensemble_eval_frame_state(eval_cache, 1)
        evaluate_frame_observable_gradients(
            observable,
            worker_templates[1],
            coords,
            box,
            neighbors[1],
            thread_params[1],
            thread_grads[1],
            atom_idxs,
            pairwise_idxs,
            specific_idxs,
            general_idxs,
        )
    else
        coords, box = ensemble_eval_frame_state(eval_cache, 1)
        evaluate_frame_observable(
            thread_params[1],
            observable,
            worker_templates[1],
            coords,
            box,
            neighbors[1],
            atom_idxs,
            pairwise_idxs,
            specific_idxs,
            general_idxs,
        )
    end

    run_ensemble_eval_workers!(1:num_frames, worker_count, eval_cache.config.schedule) do frame_idx, worker
        coords, box = ensemble_eval_frame_state(eval_cache, frame_idx)
        nbrs = neighbors[frame_idx]
        params_local = thread_params[worker]
        template = worker_templates[worker]

        if compute_gradients
            grads_local = thread_grads[worker]
            obs_value = evaluate_frame_observable_gradients(
                observable,
                template,
                coords,
                box,
                nbrs,
                params_local,
                grads_local,
                atom_idxs,
                pairwise_idxs,
                specific_idxs,
                general_idxs,
            )
            values[frame_idx] = obs_value
            for p in 1:num_params
                gradient_rows[p][frame_idx] = grads_local[p]
            end
        else
            values[frame_idx] = evaluate_frame_observable(
                params_local,
                observable,
                template,
                coords,
                box,
                nbrs,
                atom_idxs,
                pairwise_idxs,
                specific_idxs,
                general_idxs,
            )
        end
    end

    gradients = Dict{String, Vector{FT}}()
    if compute_gradients
        for (i, p_name) in enumerate(param_names)
            gradients[p_name] = gradient_rows[i]
        end
    end

    return values, gradients
end

function evaluate_leg_state_observable(
    leg::LegArtifacts,
    params::Vector{FT},
    param_names::Vector{String},
    observable,
    state_idx::Int;
    compute_gradients::Bool=true,
) where {FT <: AbstractFloat}
    isnothing(leg.eval_cache) &&
        throw(ArgumentError("Leg $(leg.name) has no ensemble-eval cache available for observable replay."))

    return evaluate_state_observable(
        leg.eval_cache,
        params,
        param_names,
        observable,
        state_idx,
        leg.idxs...;
        compute_gradients=compute_gradients,
    )
end
