#!/usr/bin/env julia

using Logging
using Printf
using Statistics
using Unitful

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))

const DEFAULT_LEG = :solvent
const DEFAULT_OUTPUT_FORMAT = :text
const DEFAULT_NONBONDED_ENERGY_TYPE = nothing

function usage()
    println(
        """
        Usage:
          julia +1.11 scripts/check_energy_consistency.jl [config.jl] [options]
          julia +1.11 scripts/check_energy_consistency.jl --full-example [options]

        Options:
          --leg NAME                 Leg name to analyze. Default: solvent
          --stage-a-md-steps N       Unfrozen AWH MD steps before freezing the bias
          --probe-md-steps N         Frozen-bias probe MD steps
          --probe-num-md-steps N     Override frozen probe lambda-sampling cadence
          --frame-limit N            Max analyzed frames after stride thinning (0 = all)
          --frame-stride N           Keep every Nth logged frame before optional cap
          --lambda-indices CSV       Extra lambda states for direct spot checks, e.g. 1,15,21
          --array-type cpu|gpu       Override SimulationConfig.AT
          --nonbonded-energy-type T  nothing|Float32|Float64 override for Molly pairwise energy evaluation
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

function parse_lambda_indices(value::AbstractString)
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
        :frame_limit => 0,
        :frame_stride => 1,
        :lambda_indices => Int[],
        :output_format => DEFAULT_OUTPUT_FORMAT,
        :array_type_override => nothing,
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
            options[:lambda_indices] = parse_lambda_indices(args[i])
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

function diagnostic_stage_a_steps(awh_sim)
    return max(awh_sim.n_md_steps * awh_sim.update_freq, 10 * awh_sim.n_md_steps)
end

function diagnostic_probe_steps(awh_sim)
    return max(awh_sim.n_md_steps * awh_sim.update_freq, 20 * awh_sim.n_md_steps)
end

function strip_energy_value(value, ::Type{FT}) where {FT <: AbstractFloat}
    return FT(ustrip(value))
end

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

function reconstruct_logged_box(sys_base, volume_entry)
    if AWHGrads.Molly.has_infinite_boundary(sys_base.boundary)
        return sys_base.boundary
    end
    side = cbrt(ustrip(volume_entry))
    return AWHGrads.Molly.CubicBoundary(side, side, side)
end

function build_cpu_state_templates(probe_sim, sys_base)
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

function group_active_lambda_summary(active_lambda_idx::Vector{Int}, deltas::Vector{<:Real})
    lambda_values = sort(unique(active_lambda_idx))
    rows = NamedTuple[]
    for lambda_idx in lambda_values
        frame_idxs = findall(==(lambda_idx), active_lambda_idx)
        push!(
            rows,
            (
                lambda_idx=lambda_idx,
                count=length(frame_idxs),
                summary=summarize_deltas(deltas[frame_idxs]),
            ),
        )
    end
    return rows
end

function top_worst_frames(
    selected_frame_indices::Vector{Int},
    active_lambda_idx::Vector{Int},
    logged_energy::Vector{<:Real},
    eval_active::Vector{<:Real},
    direct_active::Vector{<:Real},
    delta_logged_eval::Vector{<:Real},
    delta_logged_direct::Vector{<:Real},
    delta_direct_eval::Vector{<:Real};
    top_k::Int=5,
)
    order = sortperm(abs.(delta_logged_eval); rev=true)
    rows = NamedTuple[]
    for idx in first(order, min(top_k, length(order)))
        push!(
            rows,
            (
                local_frame_idx=idx,
                raw_frame_idx=selected_frame_indices[idx],
                active_lambda_idx=active_lambda_idx[idx],
                logged_energy=logged_energy[idx],
                eval_active_energy=eval_active[idx],
                direct_active_energy=direct_active[idx],
                delta_logged_eval=delta_logged_eval[idx],
                delta_logged_direct=delta_logged_direct[idx],
                delta_direct_eval=delta_direct_eval[idx],
            ),
        )
    end
    return rows
end

function direct_lambda_summary(entries)
    isempty(entries) && return NamedTuple[]
    lambda_values = sort(unique(getfield.(entries, :lambda_idx)))
    rows = NamedTuple[]
    for lambda_idx in lambda_values
        lambda_entries = filter(entry -> entry.lambda_idx == lambda_idx, entries)
        deltas = [entry.delta_direct_eval for entry in lambda_entries]
        push!(
            rows,
            (
                lambda_idx=lambda_idx,
                count=length(lambda_entries),
                summary=summarize_deltas(deltas),
            ),
        )
    end
    return rows
end

function infer_mismatch_hint(active_summary, direct_summary)
    if active_summary.count == 0
        return "No analyzed frames."
    end
    if !isfinite(active_summary.max_abs)
        return "Active-state mismatch summary is not finite."
    end
    if active_summary.max_abs <= 1e-8 && direct_summary.count > 0 && direct_summary.max_abs <= 1e-8
        return "No significant mismatch detected between logged, direct, and ensemble-eval energies."
    end
    if direct_summary.count == 0
        return "Direct CPU recomputation was not run."
    end
    if !isfinite(direct_summary.max_abs)
        return "Direct recomputation summary is not finite."
    end
    if direct_summary.max_abs <= max(1e-6, active_summary.max_abs * 1e-2)
        return "Matrix and direct CPU energies agree much better than either agrees with the logged energies."
    end
    if direct_summary.max_abs >= max(1e-6, active_summary.max_abs * 0.1)
        return "Matrix evaluation and direct CPU recomputation differ materially; inspect template rebuild or evaluate_ensemble."
    end
    return "Logged, matrix, and direct energies should be inspected together; no single mismatch dominates strongly."
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

function print_summary_table(io::IO, title::AbstractString, rows)
    println(io, title)
    isempty(rows) && return println(io, "  (none)")
    for row in rows
        summary = row.summary
        @printf(
            io,
            "  lambda=%d count=%d mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
            row.lambda_idx,
            row.count,
            summary.mean_abs,
            summary.rms,
            summary.max_abs,
        )
    end
end

function print_text_report(result)
    println("Energy Consistency Report")
    println("=========================")
    println("Config source: ", result.config_source)
    println("Leg: ", result.leg_name)
    println("Array type: ", result.array_type)
    println("Simulation float type: ", result.simulation_float_type)
    println("Energy float type: ", result.energy_float_type)
    println("Logger energy storage type: ", result.logger_energy_storage_type)
    println("Nonbonded energy type: ", result.nonbonded_energy_type)
    println("Boundary type: ", result.boundary_type)
    println("Neighbor finder: ", result.neighbor_finder_type)
    println("Stage A MD steps: ", result.stage_a_md_steps)
    println("Probe MD steps: ", result.probe_md_steps)
    println("Probe num_md_steps: ", result.probe_num_md_steps)
    println("Raw probe frames: ", result.raw_frame_count)
    println("Analyzed frames: ", result.analyzed_frame_count)
    println()

    active_summary = result.active_logged_vs_eval_summary
    direct_summary = result.direct_active_vs_eval_summary
    @printf(
        "Logged vs eval(active): mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        active_summary.mean_abs,
        active_summary.rms,
        active_summary.max_abs,
    )
    @printf(
        "Logged vs direct(active): mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        result.logged_vs_direct_active_summary.mean_abs,
        result.logged_vs_direct_active_summary.rms,
        result.logged_vs_direct_active_summary.max_abs,
    )
    @printf(
        "Direct vs eval(active): mean_abs=%.6f rms=%.6f max_abs=%.6f\n",
        direct_summary.mean_abs,
        direct_summary.rms,
        direct_summary.max_abs,
    )
    println("Hint: ", result.mismatch_hint)
    println()

    print_summary_table(stdout, "Active lambda mismatch summary:", result.active_lambda_summary)
    if !isempty(result.direct_lambda_summary)
        println()
        print_summary_table(stdout, "Direct lambda spot-check summary:", result.direct_lambda_summary)
    end

    println()
    println("Worst frames by |logged - eval(active)|:")
    if isempty(result.worst_frames)
        println("  (none)")
    else
        for row in result.worst_frames
            @printf(
                "  raw_frame=%d local_frame=%d active_lambda=%d logged=%.6f eval=%.6f direct=%.6f d(log-eval)=%.6f d(log-dir)=%.6f d(dir-eval)=%.6f\n",
                row.raw_frame_idx,
                row.local_frame_idx,
                row.active_lambda_idx,
                row.logged_energy,
                row.eval_active_energy,
                row.direct_active_energy,
                row.delta_logged_eval,
                row.delta_logged_direct,
                row.delta_direct_eval,
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
    raw_frame_count >= 2 || throw(ArgumentError(
        "Probe produced only $raw_frame_count logged frame(s). Increase --probe-md-steps or lower production_log_interval."
    ))

    selected_frame_indices = AWHGrads.probe_frame_indices(
        raw_frame_count;
        frame_stride=options[:frame_stride],
        min_frames=2,
        max_frames=options[:frame_limit],
    )
    logger_probe = AWHGrads.subset_awh_logger_frames(logger_probe_raw, selected_frame_indices)
    analyzed_frame_count = length(logger_probe.active_idx_history)

    neighbors = AWHGrads.precompute_neighbors(logger_probe, probe_sim.state.active_sys)
    u_eval, _ = AWHGrads.evaluate_ensemble(
        logger_probe,
        neighbors,
        probe_sim,
        sys_base,
        pstate.theta_active,
        pstate.param_names,
        pstate.idxs_by_leg[leg_name]...;
        compute_gradients=false,
    )

    active_lambda_idx = copy(logger_probe.active_idx_history)
    num_lambda = size(u_eval, 2)
    if any(idx -> idx < 1 || idx > num_lambda, active_lambda_idx)
        throw(ArgumentError("Active lambda history contains indices outside 1:$num_lambda"))
    end

    lambda_indices = copy(options[:lambda_indices])
    if any(idx -> idx < 1 || idx > num_lambda, lambda_indices)
        throw(ArgumentError("Requested --lambda-indices must lie in 1:$num_lambda"))
    end

    logged_energy = [strip_energy_value(val, ET) for val in logger_probe.potential_energy_history]
    eval_active = [u_eval[k, active_lambda_idx[k]] for k in eachindex(active_lambda_idx)]

    templates = build_cpu_state_templates(probe_sim, sys_base)
    direct_active = zeros(ET, analyzed_frame_count)
    direct_lambda_entries = NamedTuple[]

    for local_frame_idx in 1:analyzed_frame_count
        coords = ustrip.(logger_probe.coords_history[local_frame_idx])
        box = reconstruct_logged_box(sys_base, logger_probe.volume_history[local_frame_idx])
        neighbor_entry = neighbors[local_frame_idx]
        active_lambda = active_lambda_idx[local_frame_idx]

        direct_active[local_frame_idx] = AWHGrads.evaluate_frame_energy(
            pstate.theta_active,
            templates[active_lambda],
            coords,
            box,
            neighbor_entry,
            pstate.idxs_by_leg[leg_name]...,
        )

        for lambda_idx in lambda_indices
            direct_energy = lambda_idx == active_lambda ? direct_active[local_frame_idx] :
                AWHGrads.evaluate_frame_energy(
                    pstate.theta_active,
                    templates[lambda_idx],
                    coords,
                    box,
                    neighbor_entry,
                    pstate.idxs_by_leg[leg_name]...,
                )
            push!(
                direct_lambda_entries,
                (
                    lambda_idx=lambda_idx,
                    local_frame_idx=local_frame_idx,
                    raw_frame_idx=selected_frame_indices[local_frame_idx],
                    direct_energy=direct_energy,
                    eval_energy=u_eval[local_frame_idx, lambda_idx],
                    delta_direct_eval=direct_energy - u_eval[local_frame_idx, lambda_idx],
                ),
            )
        end
    end

    delta_logged_eval = logged_energy .- eval_active
    delta_logged_direct = logged_energy .- direct_active
    delta_direct_eval = direct_active .- eval_active

    active_logged_vs_eval_summary = summarize_deltas(delta_logged_eval)
    logged_vs_direct_active_summary = summarize_deltas(delta_logged_direct)
    direct_active_vs_eval_summary = summarize_deltas(delta_direct_eval)

    result = (
        config_source=config_source,
        leg_name=String(leg_name),
        array_type=string(typeof(probe_sim.state.active_sys.coords)),
        simulation_float_type=string(FT),
        energy_float_type=string(ET),
        logger_energy_storage_type=string(logger_energy_storage_type),
        nonbonded_energy_type=string(AWHGrads.Molly.nonbonded_energy_type(probe_sim.state.active_sys)),
        boundary_type=string(typeof(sys_base.boundary)),
        neighbor_finder_type=string(typeof(probe_sim.state.active_sys.neighbor_finder)),
        stage_a_md_steps=stage_a_md_steps,
        probe_md_steps=probe_md_steps,
        probe_num_md_steps=probe_num_md_steps,
        raw_frame_count=raw_frame_count,
        analyzed_frame_count=analyzed_frame_count,
        selected_frame_indices=selected_frame_indices,
        active_logged_vs_eval_summary=active_logged_vs_eval_summary,
        logged_vs_direct_active_summary=logged_vs_direct_active_summary,
        direct_active_vs_eval_summary=direct_active_vs_eval_summary,
        active_lambda_summary=group_active_lambda_summary(active_lambda_idx, delta_logged_eval),
        direct_lambda_summary=direct_lambda_summary(direct_lambda_entries),
        worst_frames=top_worst_frames(
            selected_frame_indices,
            active_lambda_idx,
            logged_energy,
            eval_active,
            direct_active,
            delta_logged_eval,
            delta_logged_direct,
            delta_direct_eval,
        ),
        mismatch_hint=infer_mismatch_hint(
            active_logged_vs_eval_summary,
            direct_active_vs_eval_summary,
        ),
    )

    if options[:output_format] == :json
        println(to_json(result))
    else
        print_text_report(result)
    end
end

main()
