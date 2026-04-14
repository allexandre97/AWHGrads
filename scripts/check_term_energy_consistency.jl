#!/usr/bin/env julia

using Logging
using Printf
using Statistics
using Unitful

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))

const DEFAULT_LEG = :solvent
const DEFAULT_OUTPUT_FORMAT = :text
const DEFAULT_TOP_K = 5
const DEFAULT_NONBONDED_ENERGY_TYPE = nothing

function usage()
    println(
        """
        Usage:
          julia +1.11 scripts/check_term_energy_consistency.jl [config.jl] [options]
          julia +1.11 scripts/check_term_energy_consistency.jl --full-example [options]

        Options:
          --leg NAME                 Leg name to analyze. Default: solvent
          --stage-a-md-steps N       Unfrozen AWH MD steps before freezing the bias
          --probe-md-steps N         Frozen-bias probe MD steps
          --probe-num-md-steps N     Override frozen probe lambda-sampling cadence
          --frame-indices CSV        Exact raw probe frames to inspect, e.g. 1,10,25
          --frame-limit N            Max analyzed frames after stride thinning (0 = all)
          --frame-stride N           Keep every Nth raw frame before optional cap
          --lambda-indices CSV       Extra lambda states to inspect on every selected frame
          --array-type cpu|gpu       Override SimulationConfig.AT
          --nonbonded-energy-type T  nothing|Float32|Float64 override for Molly pairwise energy evaluation
          --top-k N                  Number of worst checks to include in the report
          --output-format text|json  Report format. Default: text
          --full-example             Recreate scripts/run_alch_full_example.jl settings
          --help                     Show this message
        """
    )
end

function load_configs(config_path::AbstractString)
    cfg_obj = include(config_path)
    sim_cfg = AWHGrads.default_simulation_config()
    opt_cfg = AWHGrads.default_optimization_config()

    if cfg_obj isa AWHGrads.SimulationConfig
        sim_cfg = cfg_obj
    elseif cfg_obj isa AWHGrads.OptimizationConfig
        opt_cfg = cfg_obj
    elseif cfg_obj isa NamedTuple
        if haskey(cfg_obj, :sim_cfg)
            sim_cfg = cfg_obj.sim_cfg
        end
        if haskey(cfg_obj, :opt_cfg)
            opt_cfg = cfg_obj.opt_cfg
        end
    else
        throw(ArgumentError(
            "Config file must return `SimulationConfig`, `OptimizationConfig`, or a NamedTuple with `sim_cfg` and/or `opt_cfg`."
        ))
    end

    return sim_cfg, opt_cfg
end

function build_full_example_configs()
    base_sim = AWHGrads.default_simulation_config()
    base_opt = AWHGrads.default_optimization_config()

    lambda_schedule = Float32.(range(1.0, stop=0.0, length=21))
    dense_solvent_lambda_schedule = AWHGrads.dense_solvent_leg_lambda_schedule(
        base_sim.FT;
        lambda_scheduler=:ele_scaled,
    )
    cycle_cfg = AWHGrads.default_cycle_config(
        ;
        target_dG_kcal_mol=base_sim.dG_exp_kcal_mol,
        FT=base_sim.FT,
    )

    for leg in cycle_cfg.legs
        if leg.name == :solvent
            new_leg = AWHGrads.ThermodynamicLegConfig(
                name=leg.name,
                pdb=leg.pdb,
                coefficient=leg.coefficient,
                is_vacuum=leg.is_vacuum,
                include_pv=leg.include_pv,
                lambda_schedule=dense_solvent_lambda_schedule,
                ensemble=leg.ensemble,
                probe_awh_seed_num_md_steps=1000,
                electrostatics_method=:cutoff,
                lambda_scheduler=:ele_scaled,
                coulomb_softcore_model=:gapsys,
                lj_softcore_model=:gapsys,
                readiness_policy=leg.readiness_policy,
            )
            idx = findfirst(l -> l.name == :solvent, cycle_cfg.legs)
            cycle_cfg.legs[idx] = new_leg
        end
    end

    sim_cfg = AWHGrads.simulation_config_with(
        base_sim;
        device_id=1,
        solute_idx=1:9,
        lambda_schedule=lambda_schedule,
        awh_budget_time=base_sim.FT(60)u"ns",
        awh_probe_reweight_stride_solv=2,
        awh_probe_reweight_min_frames_solv=2000,
        awh_probe_reweight_max_frames_solv=6000,
        awh_probe_discard_fraction=0.1,
        force_field=AWHGrads.ForceFieldConfig(
            xml_files=["tip3p_standard.xml", "gaff.xml", "ethanol.xml"],
        ),
        awh_control=AWHGrads.AWHControlConfig(
            lj_softcore_alpha=0.85,
            coul_softcore_alpha=0.3,
            seed_num_md_steps=250,
            bias_update_interval_md_steps=25_000,
            stats_log_every_updates=1,
            coverage_threshold=1.0,
            significant_weight=0.1,
            initial_n_bias=100,
            well_tempered_factor=Inf,
            coverage_type=:physical,
        ),
        cycle=cycle_cfg,
        parameter_reference_leg=:solvent,
    )

    opt_cfg = AWHGrads.optimization_config_with(
        base_opt;
        max_macro_epochs=30,
        optimize_solvent=false,
        awh_split_tol_kT=1.0,
        awh_parity_tol_kT=0.5,
        awh_convergence_tol=5e-3,
    )

    return sim_cfg, opt_cfg
end

function parse_csv_ints(value::AbstractString)
    isempty(strip(value)) && return Int[]
    vals = Int[]
    for item in split(value, ",")
        stripped = strip(item)
        isempty(stripped) && continue
        push!(vals, parse(Int, stripped))
    end
    return sort(unique(vals))
end

function parse_nonbonded_energy_type(value::AbstractString)
    lowered = lowercase(strip(value))
    if lowered in ("nothing", "native", "system", "default")
        return nothing
    elseif lowered == "float32"
        return Float32
    elseif lowered == "float64"
        return Float64
    end
    throw(ArgumentError("Unsupported --nonbonded-energy-type=$value"))
end

function apply_nonbonded_energy_type(sys, override)
    override === nothing && return sys
    return AWHGrads.Molly.System(sys; nonbonded_energy_type=override)
end

function parse_cli(args::Vector{String})
    options = Dict{Symbol, Any}(
        :config_path => nothing,
        :full_example => false,
        :leg_name => DEFAULT_LEG,
        :stage_a_md_steps => nothing,
        :probe_md_steps => nothing,
        :probe_num_md_steps => nothing,
        :frame_indices => Int[],
        :frame_limit => 0,
        :frame_stride => 1,
        :lambda_indices => Int[],
        :output_format => DEFAULT_OUTPUT_FORMAT,
        :array_type_override => nothing,
        :top_k => DEFAULT_TOP_K,
        :nonbonded_energy_type => DEFAULT_NONBONDED_ENERGY_TYPE,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            usage()
            exit(0)
        elseif arg == "--full-example"
            options[:full_example] = true
            i += 1
        elseif arg == "--leg"
            i += 1
            i <= length(args) || throw(ArgumentError("--leg requires a value"))
            options[:leg_name] = Symbol(strip(args[i]))
            i += 1
        elseif arg == "--stage-a-md-steps"
            i += 1
            i <= length(args) || throw(ArgumentError("--stage-a-md-steps requires a value"))
            options[:stage_a_md_steps] = parse(Int, args[i])
            i += 1
        elseif arg == "--probe-md-steps"
            i += 1
            i <= length(args) || throw(ArgumentError("--probe-md-steps requires a value"))
            options[:probe_md_steps] = parse(Int, args[i])
            i += 1
        elseif arg == "--probe-num-md-steps"
            i += 1
            i <= length(args) || throw(ArgumentError("--probe-num-md-steps requires a value"))
            options[:probe_num_md_steps] = parse(Int, args[i])
            i += 1
        elseif arg == "--frame-indices"
            i += 1
            i <= length(args) || throw(ArgumentError("--frame-indices requires a value"))
            options[:frame_indices] = parse_csv_ints(args[i])
            i += 1
        elseif arg == "--frame-limit"
            i += 1
            i <= length(args) || throw(ArgumentError("--frame-limit requires a value"))
            options[:frame_limit] = parse(Int, args[i])
            i += 1
        elseif arg == "--frame-stride"
            i += 1
            i <= length(args) || throw(ArgumentError("--frame-stride requires a value"))
            options[:frame_stride] = parse(Int, args[i])
            i += 1
        elseif arg == "--lambda-indices"
            i += 1
            i <= length(args) || throw(ArgumentError("--lambda-indices requires a value"))
            options[:lambda_indices] = parse_csv_ints(args[i])
            i += 1
        elseif arg == "--output-format"
            i += 1
            i <= length(args) || throw(ArgumentError("--output-format requires a value"))
            fmt = Symbol(lowercase(strip(args[i])))
            fmt in (:text, :json) || throw(ArgumentError("Unsupported --output-format=$fmt"))
            options[:output_format] = fmt
            i += 1
        elseif arg == "--array-type"
            i += 1
            i <= length(args) || throw(ArgumentError("--array-type requires a value"))
            val = Symbol(lowercase(strip(args[i])))
            val in (:cpu, :gpu) || throw(ArgumentError("Unsupported --array-type=$val"))
            options[:array_type_override] = val
            i += 1
        elseif arg == "--nonbonded-energy-type"
            i += 1
            i <= length(args) || throw(ArgumentError("--nonbonded-energy-type requires a value"))
            options[:nonbonded_energy_type] = parse_nonbonded_energy_type(args[i])
            i += 1
        elseif arg == "--top-k"
            i += 1
            i <= length(args) || throw(ArgumentError("--top-k requires a value"))
            options[:top_k] = parse(Int, args[i])
            i += 1
        elseif startswith(arg, "--")
            throw(ArgumentError("Unknown option: $arg"))
        else
            isnothing(options[:config_path]) || throw(ArgumentError("Only one positional config path is allowed"))
            options[:config_path] = abspath(arg)
            i += 1
        end
    end

    if options[:full_example] && !isnothing(options[:config_path])
        throw(ArgumentError("Pass either a config path or --full-example, not both"))
    end

    options[:frame_limit] >= 0 || throw(ArgumentError("--frame-limit must be >= 0"))
    options[:frame_stride] >= 1 || throw(ArgumentError("--frame-stride must be >= 1"))
    options[:top_k] >= 0 || throw(ArgumentError("--top-k must be >= 0"))
    if !isnothing(options[:stage_a_md_steps])
        options[:stage_a_md_steps] >= 0 || throw(ArgumentError("--stage-a-md-steps must be >= 0"))
    end
    if !isnothing(options[:probe_md_steps])
        options[:probe_md_steps] > 0 || throw(ArgumentError("--probe-md-steps must be > 0"))
    end
    if !isnothing(options[:probe_num_md_steps])
        options[:probe_num_md_steps] > 0 || throw(ArgumentError("--probe-num-md-steps must be > 0"))
    end

    return options
end

function maybe_override_array_type(sim_cfg::AWHGrads.SimulationConfig, override)
    if override === nothing
        return sim_cfg
    elseif override == :cpu
        return AWHGrads.simulation_config_with(sim_cfg; AT=Array)
    else
        return AWHGrads.simulation_config_with(sim_cfg; AT=AWHGrads.CUDA.CuArray)
    end
end

function ensure_mixed_precision_energy_type(
    sim_cfg::AWHGrads.SimulationConfig,
    override,
)
    desired_type = if override !== nothing
        override
    elseif sim_cfg.FT == Float32
        Float64
    else
        AWHGrads.effective_nonbonded_energy_type(sim_cfg)
    end
    return AWHGrads.simulation_config_with(sim_cfg; nonbonded_energy_type=desired_type)
end

function load_selected_configs(options)
    if options[:full_example]
        sim_cfg, opt_cfg = build_full_example_configs()
        source = "full_example"
    elseif !isnothing(options[:config_path])
        sim_cfg, opt_cfg = load_configs(options[:config_path])
        source = options[:config_path]
    else
        sim_cfg = AWHGrads.default_simulation_config()
        opt_cfg = AWHGrads.default_optimization_config()
        source = "default"
    end
    sim_cfg = maybe_override_array_type(sim_cfg, options[:array_type_override])
    sim_cfg = ensure_mixed_precision_energy_type(sim_cfg, options[:nonbonded_energy_type])
    return sim_cfg, opt_cfg, source
end

diagnostic_stage_a_steps(awh_sim) = max(awh_sim.n_md_steps * awh_sim.update_freq, 10 * awh_sim.n_md_steps)
diagnostic_probe_steps(awh_sim) = max(awh_sim.n_md_steps * awh_sim.update_freq, 20 * awh_sim.n_md_steps)

strip_energy_value(value, ::Type{FT}) where {FT <: AbstractFloat} = FT(ustrip(value))

function validate_energy_storage(awh_sim, sys_base, ::Type{ET}) where {ET <: AbstractFloat}
    expected_logger_type = typeof(ET(1.0)u"kJ * mol^-1")
    actual_energy_type = AWHGrads.Molly.nonbonded_energy_type(awh_sim.state.active_sys)
    actual_energy_type == ET || throw(ArgumentError(
        "Expected active AWH system to use nonbonded_energy_type=$ET, got $actual_energy_type.",
    ))
    base_energy_type = AWHGrads.Molly.nonbonded_energy_type(sys_base)
    base_energy_type == ET || throw(ArgumentError(
        "Expected base system to use nonbonded_energy_type=$ET, got $base_energy_type.",
    ))

    logger = awh_sim.state.active_sys.loggers.awh_logger
    eltype(logger.potential_energy_history) == expected_logger_type || throw(ArgumentError(
        "Expected AWHEnsembleLogger potential_energy_history element type $expected_logger_type, got $(eltype(logger.potential_energy_history)).",
    ))

    for state_loggers in awh_sim.state.state_loggers
        hasproperty(state_loggers, :awh_logger) || continue
        state_logger_type = eltype(state_loggers.awh_logger.potential_energy_history)
        state_logger_type == expected_logger_type || throw(ArgumentError(
            "Expected per-state AWHEnsembleLogger energy element type $expected_logger_type, got $state_logger_type.",
        ))
    end
    return expected_logger_type
end

function reconstruct_logged_box_native(boundary, volume_entry)
    if AWHGrads.Molly.has_infinite_boundary(boundary)
        return boundary
    end
    side = cbrt(ustrip(volume_entry)) * unit(volume_entry)^(1 // 3)
    return AWHGrads.Molly.CubicBoundary(side, side, side)
end

function reconstruct_logged_box_unitless(boundary, volume_entry)
    if AWHGrads.Molly.has_infinite_boundary(boundary)
        return boundary
    end
    side = cbrt(ustrip(volume_entry))
    return AWHGrads.Molly.CubicBoundary(side, side, side)
end

function select_frame_indices(raw_frame_count::Int, explicit::Vector{Int}, stride::Int, limit::Int)
    if !isempty(explicit)
        any(idx -> idx < 1 || idx > raw_frame_count, explicit) && throw(ArgumentError(
            "--frame-indices must lie in 1:$raw_frame_count",
        ))
        return explicit
    end

    idxs = collect(1:stride:raw_frame_count)
    if limit > 0 && length(idxs) > limit
        idxs = idxs[1:limit]
    end
    isempty(idxs) && throw(ArgumentError("Selected zero frames. Increase --frame-limit or lower --frame-stride."))
    return idxs
end

function build_native_state_templates(probe_sim, sys_base)
    num_lambda = length(probe_sim.state.partition.λ_atoms)
    templates = Vector{Any}(undef, num_lambda)
    for lambda_idx in 1:num_lambda
        templates[lambda_idx] = AWHGrads.Molly.System(
            sys_base;
            atoms=probe_sim.state.partition.λ_atoms[lambda_idx],
            pairwise_inters=probe_sim.state.state_pairwise_inters[lambda_idx],
            specific_inter_lists=probe_sim.state.state_specific_inter_lists[lambda_idx],
            general_inters=probe_sim.state.state_general_inters[lambda_idx],
        )
    end
    return templates
end

function build_replay_cpu_state_templates(probe_sim, sys_base)
    num_lambda = length(probe_sim.state.partition.λ_atoms)
    templates = Vector{Any}(undef, num_lambda)
    for lambda_idx in 1:num_lambda
        raw_template = AWHGrads.Molly.System(
            sys_base;
            atoms=probe_sim.state.partition.λ_atoms[lambda_idx],
            pairwise_inters=probe_sim.state.state_pairwise_inters[lambda_idx],
            specific_inter_lists=probe_sim.state.state_specific_inter_lists[lambda_idx],
            general_inters=probe_sim.state.state_general_inters[lambda_idx],
        )
        sys_cpu = AWHGrads.Molly.from_device(raw_template)
        sys_fixed = AWHGrads._fix_interactions_for_cpu(sys_cpu)
        templates[lambda_idx] = ustrip(sys_fixed)
    end
    return templates
end

function build_native_frame_system(sys_template, coords_entry, volume_entry)
    AT = AWHGrads.Molly.array_type(sys_template.coords)
    coords_native = AWHGrads.Molly.to_device(coords_entry, AT)
    box_native = reconstruct_logged_box_native(sys_template.boundary, volume_entry)
    return AWHGrads.Molly.System(
        sys_template;
        coords=coords_native,
        boundary=box_native,
        neighbor_finder=sys_template.neighbor_finder,
    )
end

function build_replay_frame_system(sys_template, coords_entry, volume_entry)
    coords_cpu = ustrip.(coords_entry)
    box_cpu = reconstruct_logged_box_unitless(sys_template.boundary, volume_entry)
    return AWHGrads.Molly.System(
        sys_template;
        coords=coords_cpu,
        boundary=box_cpu,
        neighbor_finder=sys_template.neighbor_finder,
    )
end

_component_values(items::NamedTuple) = collect(values(items))
_component_values(items::Tuple) = collect(items)
_component_names(items::NamedTuple) = String.(keys(items))
_component_names(items::Tuple) = [string(i) for i in eachindex(items)]

function build_component_specs(sys)
    specs = NamedTuple[]

    if !isempty(sys.pairwise_inters)
        push!(specs, (label="family:pairwise_total", category="family", kind=:pairwise_family, index=0))
        names = _component_names(sys.pairwise_inters)
        vals = _component_values(sys.pairwise_inters)
        for i in eachindex(vals)
            push!(
                specs,
                (
                    label="pairwise:$(names[i]):$(nameof(typeof(vals[i])))",
                    category="pairwise",
                    kind=:pairwise_component,
                    index=i,
                ),
            )
        end
    end

    if !isempty(sys.specific_inter_lists)
        push!(specs, (label="family:specific_total", category="family", kind=:specific_family, index=0))
        names = _component_names(sys.specific_inter_lists)
        vals = _component_values(sys.specific_inter_lists)
        for i in eachindex(vals)
            push!(
                specs,
                (
                    label="specific:$(names[i]):$(nameof(typeof(vals[i])))",
                    category="specific",
                    kind=:specific_component,
                    index=i,
                ),
            )
        end
    end

    if !isempty(sys.general_inters)
        push!(specs, (label="family:general_total", category="family", kind=:general_family, index=0))
        names = _component_names(sys.general_inters)
        vals = _component_values(sys.general_inters)
        for i in eachindex(vals)
            push!(
                specs,
                (
                    label="general:$(names[i]):$(nameof(typeof(vals[i])))",
                    category="general",
                    kind=:general_component,
                    index=i,
                ),
            )
        end
    end

    return specs
end

function system_energy(sys, neighbors, ::Type{FT}) where {FT <: AbstractFloat}
    if sys.neighbor_finder isa AWHGrads.Molly.GPUNeighborFinder
        # Use Molly's production tiled CUDA path for native GPU systems.
        return FT(ustrip(AWHGrads.Molly.potential_energy(sys; n_threads=1)))
    end
    return FT(ustrip(AWHGrads.Molly.potential_energy(sys, neighbors; n_threads=1)))
end

function component_energy(sys, spec, neighbors, ::Type{FT}) where {FT <: AbstractFloat}
    subsys = if spec.kind == :pairwise_family
        AWHGrads.Molly.System(sys; specific_inter_lists=(), general_inters=())
    elseif spec.kind == :specific_family
        AWHGrads.Molly.System(sys; pairwise_inters=(), general_inters=())
    elseif spec.kind == :general_family
        AWHGrads.Molly.System(sys; pairwise_inters=(), specific_inter_lists=())
    elseif spec.kind == :pairwise_component
        inter = _component_values(sys.pairwise_inters)[spec.index]
        AWHGrads.Molly.System(sys; pairwise_inters=(inter,), specific_inter_lists=(), general_inters=())
    elseif spec.kind == :specific_component
        inter_list = _component_values(sys.specific_inter_lists)[spec.index]
        AWHGrads.Molly.System(sys; pairwise_inters=(), specific_inter_lists=(inter_list,), general_inters=())
    elseif spec.kind == :general_component
        inter = _component_values(sys.general_inters)[spec.index]
        AWHGrads.Molly.System(sys; pairwise_inters=(), specific_inter_lists=(), general_inters=(inter,))
    else
        throw(ArgumentError("Unsupported component kind $(spec.kind)"))
    end

    return system_energy(subsys, neighbors, FT)
end

function materialize_neighbors(sys)
    neighbors = AWHGrads.Molly.find_neighbors(sys)
    if !isnothing(neighbors)
        return neighbors, length(neighbors)
    end

    nf = sys.neighbor_finder
    if nf isa AWHGrads.Molly.GPUNeighborFinder
        eligible_cpu, special_cpu = AWHGrads.Molly.neighbor_finder_masks(nf)
        AT = AWHGrads.Molly.array_type(sys.coords)
        nf_explicit = AWHGrads.Molly.DistanceNeighborFinder(
            eligible=AWHGrads.Molly.to_device(eligible_cpu, AT),
            dist_cutoff=nf.dist_cutoff,
            special=AWHGrads.Molly.to_device(special_cpu, AT),
            n_steps=nf.n_steps_reorder,
        )
        explicit_neighbors = AWHGrads.Molly.find_neighbors(sys, nf_explicit, nothing, 0, true)
        return explicit_neighbors, length(explicit_neighbors)
    end

    return neighbors, 0
end

function evaluate_component_breakdown(sys, neighbors, neighbor_count, component_specs, ::Type{FT}) where {FT <: AbstractFloat}
    total = system_energy(sys, neighbors, FT)
    rows = NamedTuple[]
    family_sum = zero(FT)
    for spec in component_specs
        energy = component_energy(sys, spec, neighbors, FT)
        if spec.category == "family"
            family_sum += energy
        end
        push!(
            rows,
            (
                label=spec.label,
                category=spec.category,
                energy=energy,
            ),
        )
    end
    return (
        total=total,
        rows=rows,
        family_sum=family_sum,
        family_sum_residual=total - family_sum,
        neighbor_count=neighbor_count,
    )
end

function summarize_deltas(deltas::AbstractVector{<:Real})
    if isempty(deltas)
        return (
            count=0,
            mean_signed=NaN,
            mean_abs=NaN,
            rms=NaN,
            max_abs=NaN,
            min_signed=NaN,
            max_signed=NaN,
        )
    end
    abs_deltas = abs.(deltas)
    return (
        count=length(deltas),
        mean_signed=mean(deltas),
        mean_abs=mean(abs_deltas),
        rms=sqrt(mean(deltas .^ 2)),
        max_abs=maximum(abs_deltas),
        min_signed=minimum(deltas),
        max_signed=maximum(deltas),
    )
end

function summarize_named_deltas(delta_map::Dict{String, Vector{FT}}) where {FT <: AbstractFloat}
    rows = NamedTuple[]
    for label in sort(collect(keys(delta_map)))
        push!(rows, (label=label, summary=summarize_deltas(delta_map[label])))
    end
    sort!(rows; by=row -> row.summary.max_abs, rev=true)
    return rows
end

function summarize_named_deltas_by_lambda(delta_map::Dict{Int, Dict{String, Vector{FT}}}) where {FT <: AbstractFloat}
    rows = NamedTuple[]
    for lambda_idx in sort(collect(keys(delta_map)))
        push!(
            rows,
            (
                lambda_idx=lambda_idx,
                components=summarize_named_deltas(delta_map[lambda_idx]),
            ),
        )
    end
    return rows
end

function summarize_total_by_lambda(checks)
    lambda_values = sort(unique(getfield.(checks, :lambda_idx)))
    rows = NamedTuple[]
    for lambda_idx in lambda_values
        deltas = [row.delta_native_replay for row in checks if row.lambda_idx == lambda_idx]
        push!(rows, (lambda_idx=lambda_idx, summary=summarize_deltas(deltas)))
    end
    return rows
end

function top_worst_checks(checks, top_k::Int)
    isempty(checks) && return NamedTuple[]
    order = sortperm([abs(row.delta_native_replay) for row in checks]; rev=true)
    return checks[first(order, min(top_k, length(order)))]
end

function json_escape(value::AbstractString)
    return replace(
        value,
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\n" => "\\n",
        "\r" => "\\r",
        "\t" => "\\t",
    )
end

function to_json(value)
    if value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Symbol
        return to_json(String(value))
    elseif value isa AbstractString
        return "\"" * json_escape(value) * "\""
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return isfinite(value) ? string(value) : to_json(string(value))
    elseif value isa NamedTuple
        parts = String[]
        for name in keys(value)
            push!(parts, to_json(String(name)) * ":" * to_json(getfield(value, name)))
        end
        return "{" * join(parts, ",") * "}"
    elseif value isa AbstractDict
        parts = String[]
        for key in sort!(collect(keys(value)); by=string)
            push!(parts, to_json(string(key)) * ":" * to_json(value[key]))
        end
        return "{" * join(parts, ",") * "}"
    elseif value isa Tuple
        return "[" * join(to_json.(collect(value)), ",") * "]"
    elseif value isa AbstractVector
        return "[" * join(to_json.(value), ",") * "]"
    else
        return to_json(string(value))
    end
end

function print_component_summary(io::IO, title::AbstractString, rows; limit::Int=10)
    println(io, title)
    isempty(rows) && return println(io, "  (none)")
    for row in first(rows, min(limit, length(rows)))
        summary = row.summary
        @printf(
            io,
            "  %s count=%d mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
            row.label,
            summary.count,
            summary.mean_abs,
            summary.rms,
            summary.max_abs,
        )
    end
end

function print_lambda_total_summary(io::IO, rows)
    println(io, "Total native-vs-replay mismatch by lambda:")
    isempty(rows) && return println(io, "  (none)")
    for row in rows
        summary = row.summary
        @printf(
            io,
            "  lambda=%d count=%d mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
            row.lambda_idx,
            summary.count,
            summary.mean_abs,
            summary.rms,
            summary.max_abs,
        )
    end
end

function print_text_report(result)
    println("Per-Term Energy Consistency Report")
    println("==================================")
    println("Config source: ", result.config_source)
    println("Leg: ", result.leg_name)
    println("Array type: ", result.array_type)
    println("Simulation float type: ", result.simulation_float_type)
    println("Energy float type: ", result.energy_float_type)
    println("Logger energy storage type: ", result.logger_energy_storage_type)
    println("Nonbonded energy type: ", result.nonbonded_energy_type)
    println("Boundary type: ", result.boundary_type)
    println("Native neighbor finder: ", result.native_neighbor_finder_type)
    println("Replay neighbor finder: ", result.replay_neighbor_finder_type)
    println("Stage A MD steps: ", result.stage_a_md_steps)
    println("Probe MD steps: ", result.probe_md_steps)
    println("Probe num_md_steps: ", result.probe_num_md_steps)
    println("Raw probe frames: ", result.raw_frame_count)
    println("Analyzed frames: ", result.analyzed_frame_count)
    println("Evaluated checks: ", result.check_count)
    println("Evaluated lambdas: ", join(result.evaluated_lambda_indices, ", "))
    println()

    total_summary = result.total_native_vs_replay_summary
    @printf(
        "Native vs replay total: mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        total_summary.mean_abs,
        total_summary.rms,
        total_summary.max_abs,
    )
    @printf(
        "Native family-sum residual: mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        result.native_family_sum_residual_summary.mean_abs,
        result.native_family_sum_residual_summary.rms,
        result.native_family_sum_residual_summary.max_abs,
    )
    @printf(
        "Replay family-sum residual: mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        result.replay_family_sum_residual_summary.mean_abs,
        result.replay_family_sum_residual_summary.rms,
        result.replay_family_sum_residual_summary.max_abs,
    )
    @printf(
        "Neighbor-count delta (native - replay): mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        result.neighbor_count_delta_summary.mean_abs,
        result.neighbor_count_delta_summary.rms,
        result.neighbor_count_delta_summary.max_abs,
    )
    if result.active_logged_vs_native_summary.count > 0
        @printf(
            "Logged vs native(active): mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
            result.active_logged_vs_native_summary.mean_abs,
            result.active_logged_vs_native_summary.rms,
            result.active_logged_vs_native_summary.max_abs,
        )
        @printf(
            "Logged vs replay(active): mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
            result.active_logged_vs_replay_summary.mean_abs,
            result.active_logged_vs_replay_summary.rms,
            result.active_logged_vs_replay_summary.max_abs,
        )
    end
    println()

    print_lambda_total_summary(stdout, result.total_native_vs_replay_by_lambda)
    println()
    print_component_summary(stdout, "Top component deltas:", result.component_delta_summary; limit=12)

    println()
    println("Worst checks by |native - replay total|:")
    if isempty(result.worst_checks)
        println("  (none)")
        return
    end

    for row in result.worst_checks
        @printf(
            "  raw_frame=%d local_frame=%d lambda=%d active_lambda=%d native_total=%.6f replay_total=%.6f delta=%.6f native_neighbors=%d replay_neighbors=%d\n",
            row.raw_frame_idx,
            row.local_frame_idx,
            row.lambda_idx,
            row.active_lambda_idx,
            row.native_total,
            row.replay_total,
            row.delta_native_replay,
            row.native_neighbor_count,
            row.replay_neighbor_count,
        )
        if row.logged_energy !== nothing
            @printf(
                "    logged=%.6f d(log-native)=%.6f d(log-replay)=%.6f\n",
                row.logged_energy,
                row.delta_logged_native,
                row.delta_logged_replay,
            )
        end
        for term in row.term_rows
            @printf(
                "    %s native=%.6f replay=%.6f delta=%.6f\n",
                term.label,
                term.native_energy,
                term.replay_energy,
                term.delta_native_replay,
            )
        end
    end
end

function main()
    options = parse_cli(copy(ARGS))
    sim_cfg, opt_cfg, config_source = load_selected_configs(options)

    with_logger(NullLogger()) do
        AWHGrads.apply_simulation_config!(sim_cfg)
    end
    FT = sim_cfg.FT
    ET = AWHGrads.effective_nonbonded_energy_type(sim_cfg)

    cycle_cfg = AWHGrads.validate_cycle_config(
        AWHGrads.resolved_cycle_config(sim_cfg);
        default_lambda_schedule=sim_cfg.lambda_schedule,
        FT=FT,
    )
    leg_names = [leg.name for leg in cycle_cfg.legs]
    options[:leg_name] in leg_names || throw(ArgumentError(
        "Unknown leg `$(options[:leg_name])`. Available legs: $(join(string.(leg_names), ", "))."
    ))

    state_schedules_by_leg = Dict(
        leg.name => AWHGrads.resolve_leg_state_schedule(leg, sim_cfg.lambda_schedule, FT)
        for leg in cycle_cfg.legs
    )
    runtime = AWHGrads.RuntimeState()
    pstate = AWHGrads.initialize_parameter_state(sim_cfg, opt_cfg, cycle_cfg)

    T_coord, T_vol, T_en = AWHGrads.awh_logger_value_types(sim_cfg)

    awh_by_leg, sys_by_leg = with_logger(NullLogger()) do
        AWHGrads.setup_macro_legs(
            cycle_cfg,
            sim_cfg,
            state_schedules_by_leg,
            runtime,
            pstate.theta_active,
            pstate.idxs_by_leg,
            T_coord,
            T_vol,
            T_en,
            1,
            opt_cfg.restart_rmsd_tol_nm,
        )
    end

    leg_name = options[:leg_name]
    awh_leg = awh_by_leg[leg_name]
    sys_base = apply_nonbonded_energy_type(sys_by_leg[leg_name], options[:nonbonded_energy_type])
    logger_energy_storage_type = validate_energy_storage(awh_leg, sys_base, ET)

    stage_a_md_steps = isnothing(options[:stage_a_md_steps]) ?
        diagnostic_stage_a_steps(awh_leg) :
        options[:stage_a_md_steps]
    probe_md_steps = isnothing(options[:probe_md_steps]) ?
        diagnostic_probe_steps(awh_leg) :
        options[:probe_md_steps]
    probe_num_md_steps = isnothing(options[:probe_num_md_steps]) ?
        awh_leg.n_md_steps :
        options[:probe_num_md_steps]

    if stage_a_md_steps > 0
        with_logger(NullLogger()) do
            AWHGrads.simulate!(awh_leg, stage_a_md_steps)
        end
    end

    probe_sim, _ = AWHGrads.build_stage_b_probe_sim(
        awh_leg,
        probe_md_steps;
        probe_num_md_steps=probe_num_md_steps,
    )
    probe_sim.state.active_sys = apply_nonbonded_energy_type(
        probe_sim.state.active_sys,
        options[:nonbonded_energy_type],
    )
    with_logger(NullLogger()) do
        AWHGrads.simulate!(probe_sim, probe_md_steps)
    end

    logger_probe_raw = AWHGrads.get_production_logger(probe_sim, String(leg_name))
    logger_energy_storage_type = validate_energy_storage(probe_sim, sys_base, ET)
    raw_frame_count = length(logger_probe_raw.active_idx_history)
    raw_frame_count >= 1 || throw(ArgumentError(
        "Probe produced zero logged frames. Increase --probe-md-steps or lower production_log_interval."
    ))

    selected_frame_indices = select_frame_indices(
        raw_frame_count,
        options[:frame_indices],
        options[:frame_stride],
        options[:frame_limit],
    )
    logger_probe = AWHGrads.subset_awh_logger_frames(logger_probe_raw, selected_frame_indices)
    analyzed_frame_count = length(logger_probe.active_idx_history)

    active_lambda_idx = copy(logger_probe.active_idx_history)
    num_lambda = length(probe_sim.state.partition.λ_atoms)
    if any(idx -> idx < 1 || idx > num_lambda, active_lambda_idx)
        throw(ArgumentError("Active lambda history contains indices outside 1:$num_lambda"))
    end
    if any(idx -> idx < 1 || idx > num_lambda, options[:lambda_indices])
        throw(ArgumentError("Requested --lambda-indices must lie in 1:$num_lambda"))
    end

    evaluated_lambda_indices = sort(unique(vcat(options[:lambda_indices], active_lambda_idx)))
    native_templates = build_native_state_templates(probe_sim, sys_base)
    replay_templates = build_replay_cpu_state_templates(probe_sim, sys_base)

    component_specs = build_component_specs(native_templates[first(evaluated_lambda_indices)])
    replay_component_specs = build_component_specs(replay_templates[first(evaluated_lambda_indices)])
    [row.label for row in component_specs] == [row.label for row in replay_component_specs] || throw(ArgumentError(
        "Native and replay component layouts differ on the first evaluated lambda state.",
    ))

    total_deltas = ET[]
    neighbor_count_deltas = ET[]
    native_family_sum_residuals = ET[]
    replay_family_sum_residuals = ET[]
    active_logged_native_deltas = ET[]
    active_logged_replay_deltas = ET[]
    component_delta_map = Dict{String, Vector{ET}}()
    component_delta_map_by_lambda = Dict{Int, Dict{String, Vector{ET}}}()
    check_rows = NamedTuple[]

    for local_frame_idx in 1:analyzed_frame_count
        coords_entry = logger_probe.coords_history[local_frame_idx]
        volume_entry = logger_probe.volume_history[local_frame_idx]
        active_lambda = active_lambda_idx[local_frame_idx]
        raw_frame_idx = selected_frame_indices[local_frame_idx]
        logged_energy = strip_energy_value(logger_probe.potential_energy_history[local_frame_idx], ET)

        for lambda_idx in evaluated_lambda_indices
            native_sys = build_native_frame_system(native_templates[lambda_idx], coords_entry, volume_entry)
            replay_sys = build_replay_frame_system(replay_templates[lambda_idx], coords_entry, volume_entry)

            native_neighbors, native_neighbor_count = materialize_neighbors(native_sys)
            replay_neighbors, replay_neighbor_count = materialize_neighbors(replay_sys)

            native_breakdown = evaluate_component_breakdown(native_sys, native_neighbors, native_neighbor_count, component_specs, ET)
            replay_breakdown = evaluate_component_breakdown(replay_sys, replay_neighbors, replay_neighbor_count, component_specs, ET)

            native_labels = getfield.(native_breakdown.rows, :label)
            replay_labels = getfield.(replay_breakdown.rows, :label)
            native_labels == replay_labels || throw(ArgumentError(
                "Component label mismatch at raw_frame=$raw_frame_idx lambda=$lambda_idx.",
            ))

            push!(total_deltas, native_breakdown.total - replay_breakdown.total)
            push!(neighbor_count_deltas, ET(native_breakdown.neighbor_count - replay_breakdown.neighbor_count))
            push!(native_family_sum_residuals, native_breakdown.family_sum_residual)
            push!(replay_family_sum_residuals, replay_breakdown.family_sum_residual)

            term_rows = NamedTuple[]
            for row_idx in eachindex(native_breakdown.rows)
                native_row = native_breakdown.rows[row_idx]
                replay_row = replay_breakdown.rows[row_idx]
                delta = native_row.energy - replay_row.energy
                get!(component_delta_map, native_row.label, ET[])
                push!(component_delta_map[native_row.label], delta)

                lambda_map = get!(component_delta_map_by_lambda, lambda_idx, Dict{String, Vector{ET}}())
                get!(lambda_map, native_row.label, ET[])
                push!(lambda_map[native_row.label], delta)

                push!(
                    term_rows,
                    (
                        label=native_row.label,
                        category=native_row.category,
                        native_energy=native_row.energy,
                        replay_energy=replay_row.energy,
                        delta_native_replay=delta,
                    ),
                )
            end
            sort!(term_rows; by=row -> abs(row.delta_native_replay), rev=true)

            logged_energy_value = nothing
            delta_logged_native = nothing
            delta_logged_replay = nothing
            if lambda_idx == active_lambda
                logged_energy_value = logged_energy
                delta_logged_native = logged_energy - native_breakdown.total
                delta_logged_replay = logged_energy - replay_breakdown.total
                push!(active_logged_native_deltas, delta_logged_native)
                push!(active_logged_replay_deltas, delta_logged_replay)
            end

            push!(
                check_rows,
                (
                    local_frame_idx=local_frame_idx,
                    raw_frame_idx=raw_frame_idx,
                    lambda_idx=lambda_idx,
                    active_lambda_idx=active_lambda,
                    is_active_lambda=lambda_idx == active_lambda,
                    native_total=native_breakdown.total,
                    replay_total=replay_breakdown.total,
                    delta_native_replay=native_breakdown.total - replay_breakdown.total,
                    native_neighbor_count=native_breakdown.neighbor_count,
                    replay_neighbor_count=replay_breakdown.neighbor_count,
                    native_family_sum=native_breakdown.family_sum,
                    replay_family_sum=replay_breakdown.family_sum,
                    native_family_sum_residual=native_breakdown.family_sum_residual,
                    replay_family_sum_residual=replay_breakdown.family_sum_residual,
                    logged_energy=logged_energy_value,
                    delta_logged_native=delta_logged_native,
                    delta_logged_replay=delta_logged_replay,
                    term_rows=term_rows,
                ),
            )
        end
    end

    result = (
        config_source=config_source,
        leg_name=String(leg_name),
        array_type=string(typeof(probe_sim.state.active_sys.coords)),
        simulation_float_type=string(FT),
        energy_float_type=string(ET),
        logger_energy_storage_type=string(logger_energy_storage_type),
        nonbonded_energy_type=string(AWHGrads.Molly.nonbonded_energy_type(probe_sim.state.active_sys)),
        boundary_type=string(typeof(sys_base.boundary)),
        native_neighbor_finder_type=string(typeof(native_templates[first(evaluated_lambda_indices)].neighbor_finder)),
        replay_neighbor_finder_type=string(typeof(replay_templates[first(evaluated_lambda_indices)].neighbor_finder)),
        stage_a_md_steps=stage_a_md_steps,
        probe_md_steps=probe_md_steps,
        probe_num_md_steps=probe_num_md_steps,
        raw_frame_count=raw_frame_count,
        analyzed_frame_count=analyzed_frame_count,
        check_count=length(check_rows),
        selected_frame_indices=selected_frame_indices,
        requested_lambda_indices=options[:lambda_indices],
        evaluated_lambda_indices=evaluated_lambda_indices,
        total_native_vs_replay_summary=summarize_deltas(total_deltas),
        total_native_vs_replay_by_lambda=summarize_total_by_lambda(check_rows),
        neighbor_count_delta_summary=summarize_deltas(neighbor_count_deltas),
        native_family_sum_residual_summary=summarize_deltas(native_family_sum_residuals),
        replay_family_sum_residual_summary=summarize_deltas(replay_family_sum_residuals),
        active_logged_vs_native_summary=summarize_deltas(active_logged_native_deltas),
        active_logged_vs_replay_summary=summarize_deltas(active_logged_replay_deltas),
        component_delta_summary=summarize_named_deltas(component_delta_map),
        component_delta_summary_by_lambda=summarize_named_deltas_by_lambda(component_delta_map_by_lambda),
        worst_checks=top_worst_checks(check_rows, options[:top_k]),
    )

    if options[:output_format] == :json
        println(to_json(result))
    else
        print_text_report(result)
    end
end

main()
