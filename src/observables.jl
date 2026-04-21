const G_PER_ML_PER_G_PER_MOL_NM3 = Float64(1.0e21 / ustrip(Unitful.Na))
const COULOMB_CONST_KJ_MOL_NM_PER_E2 = Float64(ustrip(u"kJ * mol^-1 * nm", Molly.coulomb_const))

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

"""
    DielectricConstantObservable()

Marker observable used with `StateObservableTarget` to train against the
relative dielectric constant reconstructed from reweighted dipole fluctuations
and the mean simulation-cell volume. The target is evaluated at the optimizer
level rather than from a single frame, so this observable cannot be called on a
standalone `Molly.System`.
"""
Base.@kwdef struct DielectricConstantObservable{FT <: AbstractFloat}
    coulomb_const::FT = FT(COULOMB_CONST_KJ_MOL_NM_PER_E2)
end

@inline _observable_mass_value(value) = value isa Unitful.AbstractQuantity ? Float64(ustrip(u"g/mol", value)) : Float64(value)
@inline _observable_volume_value(value) = value isa Unitful.AbstractQuantity ? Float64(ustrip(u"nm^3", value)) : Float64(value)
@inline _observable_dipole_component_value(value) = value isa Unitful.AbstractQuantity ? Float64(ustrip(u"nm", value)) : Float64(value)

function (obs::MassDensityObservable)(sys::System)
    Molly.has_infinite_boundary(sys.boundary) &&
        throw(ArgumentError("MassDensityObservable requires a finite periodic boundary."))
    volume_nm3 = _observable_volume_value(volume(sys.boundary))
    volume_nm3 > 0 || throw(ArgumentError("MassDensityObservable requires a positive box volume."))
    total_mass = _observable_mass_value(sys.total_mass)
    return obs.conversion_factor * total_mass / volume_nm3
end

function (::DielectricConstantObservable)(::System)
    throw(ArgumentError("DielectricConstantObservable is an ensemble observable and must be evaluated through the state-observable target path."))
end

"""
    validate_state_observable_target(observable, target_name, leg_cfg)

Observable-specific target validation hook. The default method accepts any
scalar per-frame callable. More structured observables may extend this method
to enforce leg-specific requirements while still using the generic
`StateObservableTarget` pipeline.
"""
validate_state_observable_target(observable, target_name::Symbol, leg_cfg::ThermodynamicLegConfig) = nothing

function validate_state_observable_target(
    ::DielectricConstantObservable,
    target_name::Symbol,
    leg_cfg::ThermodynamicLegConfig,
)
    leg_cfg.is_vacuum &&
        throw(ArgumentError("DielectricConstantObservable target `$target_name` requires a finite periodic leg, but leg `$(leg_cfg.name)` is vacuum."))
    return nothing
end

struct _DipoleComponentObservable{I} end
struct _DipoleMagnitudeSquaredObservable end

const _DIELECTRIC_COMPONENT_OBSERVABLES = (
    _DipoleComponentObservable{1}(),
    _DipoleComponentObservable{2}(),
    _DipoleComponentObservable{3}(),
)

@inline function _observable_dipole_component(sys::System, component_idx::Int)
    dipole = Molly.dipole_moment(sys)
    1 <= component_idx <= length(dipole) ||
        throw(ArgumentError("DielectricConstantObservable requires a dipole moment with at least $component_idx components, got $(length(dipole))."))
    return _observable_dipole_component_value(dipole[component_idx])
end

function (::_DipoleComponentObservable{I})(sys::System) where {I}
    return _observable_dipole_component(sys, I)
end

function (::_DipoleMagnitudeSquaredObservable)(sys::System)
    dipole = Molly.dipole_moment(sys)
    length(dipole) == 3 ||
        throw(ArgumentError("DielectricConstantObservable currently requires a 3D dipole moment, got $(length(dipole)) components."))
    dipole_sq = 0.0
    @inbounds for component in dipole
        component_value = _observable_dipole_component_value(component)
        dipole_sq += component_value * component_value
    end
    return dipole_sq
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
    legs_by_name = Dict(leg.name => leg for leg in cycle_cfg.legs)

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
            validate_state_observable_target(raw_target.observable, target_name, legs_by_name[raw_target.leg])

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
    1 <= state_idx <= eval_cache.num_lambda ||
        throw(ArgumentError("Requested replay state index $state_idx outside valid range 1:$(eval_cache.num_lambda)."))
    if !isnothing(eval_cache.template_cache)
        return eval_cache.template_cache[state_idx]
    end
    isnothing(eval_cache.awh_sim_prod) && throw(ArgumentError("Ensemble-eval cache has no template cache and no source AWH simulation; cannot rebuild observable replay template."))
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
    neighbors = eval_cache.neighbors
    num_frames = eval_cache.frame_count
    num_params = length(params)

    values = zeros(FT, num_frames)
    gradient_rows = compute_gradients ? [zeros(FT, num_frames) for _ in 1:num_params] : nothing
    num_frames == 0 && return values, Dict{String, Vector{FT}}()

    worker_count = max(1, min(eval_cache.config.threads, num_frames))
    thread_params = [deepcopy(params) for _ in 1:worker_count]
    thread_grads = compute_gradients ? [zeros(FT, num_params) for _ in 1:worker_count] : nothing
    base_template = ensemble_eval_template(eval_cache, state_idx)
    worker_templates = [deepcopy(base_template) for _ in 1:worker_count]
    batch_size = _ensemble_eval_batch_size(compute_gradients, worker_count, num_frames)

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

        if compute_gradients
            _run_with_gc_disabled(batch_runner)
        else
            batch_runner()
        end
    end

    if compute_gradients
        GC.gc(true)
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

function evaluate_leg_state_dielectric_observable(
    leg::LegArtifacts,
    params::Vector{FT},
    param_names::Vector{String},
    state_idx::Int;
    compute_gradients::Bool=true,
) where {FT <: AbstractFloat}
    dipole_sq_values, dipole_sq_gradients_theta = evaluate_leg_state_observable(
        leg,
        params,
        param_names,
        _DipoleMagnitudeSquaredObservable(),
        state_idx;
        compute_gradients=compute_gradients,
    )
    dipole_component_results = ntuple(3) do component_idx
        evaluate_leg_state_observable(
            leg,
            params,
            param_names,
            _DIELECTRIC_COMPONENT_OBSERVABLES[component_idx],
            state_idx;
            compute_gradients=compute_gradients,
        )
    end
    dipole_component_values = ntuple(3) do component_idx
        first(dipole_component_results[component_idx])
    end
    dipole_component_gradients_theta = if compute_gradients
        ntuple(3) do component_idx
            last(dipole_component_results[component_idx])
        end
    else
        ntuple(_ -> Dict{String, Vector{FT}}(), 3)
    end

    return (
        dipole_sq_values=dipole_sq_values,
        dipole_sq_gradients_theta=dipole_sq_gradients_theta,
        dipole_component_values=dipole_component_values,
        dipole_component_gradients_theta=dipole_component_gradients_theta,
    )
end
