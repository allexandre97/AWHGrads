#!/usr/bin/env julia

using Printf
using CUDA
using Molly
using Unitful

include(joinpath(@__DIR__, "src", "AWHGrads.jl"))

const DEFAULT_PDB = "ethanol_solv.pdb"
const DEFAULT_CASE = :softcore
const DEFAULT_FLOAT_TYPES = (Float32, Float64)
const DEFAULT_ARRAY_TYPES = CUDA.functional() ? (:cpu, :gpu) : (:cpu,)
const DEFAULT_LAMBDAS = (1.0, 0.5, 0.15, 0.0)
const DEFAULT_SOLUTE = 1:9
const DEFAULT_SCHEDULER = :ele_scaled
const DEFAULT_TOP_COMPONENTS = 4

function usage()
    println(
        """
        Usage:
          julia +1.11 ./probas.jl [options]

        Options:
          --pdb PATH                  Input structure. Default: ethanol_solv.pdb
          --case base|softcore|both   Which system family to inspect. Default: softcore
          --float-types CSV           Float32,Float64 or a subset. Default: Float32,Float64
          --array-types CSV           cpu,gpu or a subset. Default: cpu plus gpu if CUDA is available
          --lambda-values CSV         Soft-core lambda values. Default: 1.0,0.5,0.15,0.0
          --solute-indices SPEC       Solute atoms, e.g. 1:9 or 1,2,3. Default: 1:9
          --scheduler NAME            Lambda scheduler. Default: ele_scaled
          --top-components N          Number of largest component deltas to print. Default: 4
          --help                      Show this message

        This script isolates where the energy mismatch first appears in the transform chain:
          native -> from_device -> ustrip -> _fix_interactions_for_cpu
        """
    )
end

function parse_csv_strings(value::AbstractString)
    parts = String[]
    for item in split(value, ",")
        stripped = strip(item)
        isempty(stripped) && continue
        push!(parts, stripped)
    end
    return unique(parts)
end

function parse_range_or_csv(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && return Int[]
    if occursin(":", stripped)
        parts = split(stripped, ":")
        length(parts) == 2 || throw(ArgumentError("Invalid range `$value`; expected `start:stop`."))
        return collect(parse(Int, parts[1]):parse(Int, parts[2]))
    end
    return parse.(Int, parse_csv_strings(stripped))
end

function parse_float_types(value::AbstractString)
    mapping = Dict("float32" => Float32, "float64" => Float64)
    types = DataType[]
    for item in parse_csv_strings(lowercase(value))
        haskey(mapping, item) || throw(ArgumentError("Unsupported float type `$item`."))
        push!(types, mapping[item])
    end
    isempty(types) && throw(ArgumentError("--float-types cannot be empty."))
    return Tuple(types)
end

function parse_array_types(value::AbstractString)
    allowed = Set(("cpu", "gpu"))
    syms = Symbol[]
    for item in parse_csv_strings(lowercase(value))
        item in allowed || throw(ArgumentError("Unsupported array type `$item`."))
        push!(syms, Symbol(item))
    end
    isempty(syms) && throw(ArgumentError("--array-types cannot be empty."))
    return Tuple(unique(syms))
end

function parse_float_values(value::AbstractString)
    vals = Float64[]
    for item in parse_csv_strings(value)
        push!(vals, parse(Float64, item))
    end
    isempty(vals) && throw(ArgumentError("--lambda-values cannot be empty."))
    return Tuple(vals)
end

function parse_case(value::AbstractString)
    sym = Symbol(lowercase(strip(value)))
    sym in (:base, :softcore, :both) || throw(ArgumentError("Unsupported --case=$value"))
    return sym
end

function parse_cli(args::Vector{String})
    opts = Dict{Symbol, Any}(
        :pdb => joinpath(@__DIR__, DEFAULT_PDB),
        :case => DEFAULT_CASE,
        :float_types => DEFAULT_FLOAT_TYPES,
        :array_types => DEFAULT_ARRAY_TYPES,
        :lambda_values => DEFAULT_LAMBDAS,
        :solute_indices => collect(DEFAULT_SOLUTE),
        :scheduler => DEFAULT_SCHEDULER,
        :top_components => DEFAULT_TOP_COMPONENTS,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            usage()
            exit(0)
        elseif arg == "--pdb"
            i += 1
            i <= length(args) || throw(ArgumentError("--pdb requires a value"))
            opts[:pdb] = abspath(args[i])
            i += 1
        elseif arg == "--case"
            i += 1
            i <= length(args) || throw(ArgumentError("--case requires a value"))
            opts[:case] = parse_case(args[i])
            i += 1
        elseif arg == "--float-types"
            i += 1
            i <= length(args) || throw(ArgumentError("--float-types requires a value"))
            opts[:float_types] = parse_float_types(args[i])
            i += 1
        elseif arg == "--array-types"
            i += 1
            i <= length(args) || throw(ArgumentError("--array-types requires a value"))
            opts[:array_types] = parse_array_types(args[i])
            i += 1
        elseif arg == "--lambda-values"
            i += 1
            i <= length(args) || throw(ArgumentError("--lambda-values requires a value"))
            opts[:lambda_values] = parse_float_values(args[i])
            i += 1
        elseif arg == "--solute-indices"
            i += 1
            i <= length(args) || throw(ArgumentError("--solute-indices requires a value"))
            opts[:solute_indices] = parse_range_or_csv(args[i])
            i += 1
        elseif arg == "--scheduler"
            i += 1
            i <= length(args) || throw(ArgumentError("--scheduler requires a value"))
            opts[:scheduler] = Symbol(lowercase(strip(args[i])))
            i += 1
        elseif arg == "--top-components"
            i += 1
            i <= length(args) || throw(ArgumentError("--top-components requires a value"))
            opts[:top_components] = parse(Int, args[i])
            i += 1
        else
            throw(ArgumentError("Unknown option: $arg"))
        end
    end

    opts[:top_components] >= 1 || throw(ArgumentError("--top-components must be >= 1"))
    isfile(opts[:pdb]) || throw(ArgumentError("PDB file not found: $(opts[:pdb])"))
    if :gpu in opts[:array_types] && !CUDA.functional()
        throw(ArgumentError("CUDA is not available, but --array-types requested gpu"))
    end
    return opts
end

function format_num(value)
    if value isa Bool
        return string(value)
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return @sprintf("%.8f", value)
    else
        return string(value)
    end
end

function strip_scalar(value)
    if value isa Bool || value isa Integer
        return value
    elseif value isa Unitful.AbstractQuantity
        return float(ustrip(value))
    elseif value isa AbstractFloat
        return float(value)
    elseif value isa Molly.DistanceCutoff
        return strip_scalar(value.dist_cutoff)
    elseif value isa Molly.NoCutoff
        return "NoCutoff"
    else
        return string(value)
    end
end

function interaction_signature(inter)
    fields = Pair{String, Any}[]
    push!(fields, "type" => string(nameof(typeof(inter))))
    if hasproperty(inter, :use_neighbors)
        push!(fields, "use_neighbors" => getproperty(inter, :use_neighbors))
    end
    for name in (:cutoff, :dist_cutoff, :solvent_dielectric, :α, :σQ, :weight_special, :coulomb_const)
        if hasproperty(inter, name)
            push!(fields, string(name) => strip_scalar(getproperty(inter, name)))
        end
    end
    if hasproperty(inter, :scheduler)
        push!(fields, "scheduler" => string(nameof(typeof(getproperty(inter, :scheduler)))))
    end
    return fields
end

function _component_values(items::NamedTuple)
    return collect(values(items))
end

function _component_values(items::Tuple)
    return collect(items)
end

function _component_names(items::NamedTuple)
    return String.(keys(items))
end

function _component_names(items::Tuple)
    return [string(i) for i in eachindex(items)]
end

function build_component_specs(sys)
    specs = NamedTuple[]

    if !isempty(sys.pairwise_inters)
        push!(specs, (label="family:pairwise_total", kind=:pairwise_family, index=0))
        names = _component_names(sys.pairwise_inters)
        vals = _component_values(sys.pairwise_inters)
        for i in eachindex(vals)
            push!(specs, (label="pairwise:$(names[i]):$(nameof(typeof(vals[i])))", kind=:pairwise_component, index=i))
        end
    end

    if !isempty(sys.specific_inter_lists)
        push!(specs, (label="family:specific_total", kind=:specific_family, index=0))
        names = _component_names(sys.specific_inter_lists)
        vals = _component_values(sys.specific_inter_lists)
        for i in eachindex(vals)
            push!(specs, (label="specific:$(names[i]):$(nameof(typeof(vals[i])))", kind=:specific_component, index=i))
        end
    end

    if !isempty(sys.general_inters)
        push!(specs, (label="family:general_total", kind=:general_family, index=0))
        names = _component_names(sys.general_inters)
        vals = _component_values(sys.general_inters)
        for i in eachindex(vals)
            push!(specs, (label="general:$(names[i]):$(nameof(typeof(vals[i])))", kind=:general_component, index=i))
        end
    end

    return specs
end

function materialize_neighbors(sys)
    neighbors = Molly.find_neighbors(sys)
    if !isnothing(neighbors)
        return neighbors, length(neighbors)
    end

    nf = sys.neighbor_finder
    if nf isa Molly.GPUNeighborFinder
        eligible_cpu, special_cpu = Molly.neighbor_finder_masks(nf)
        AT = Molly.array_type(sys.coords)
        explicit_nf = Molly.DistanceNeighborFinder(
            eligible=Molly.to_device(eligible_cpu, AT),
            dist_cutoff=nf.dist_cutoff,
            special=Molly.to_device(special_cpu, AT),
            n_steps=nf.n_steps_reorder,
        )
        explicit_neighbors = Molly.find_neighbors(sys, explicit_nf, nothing, 0, true)
        return explicit_neighbors, length(explicit_neighbors)
    end

    return neighbors, 0
end

function energy_value(value, ::Type{FT}) where {FT <: AbstractFloat}
    return FT(ustrip(value))
end

function total_energy(sys, neighbors, ::Type{FT}) where {FT <: AbstractFloat}
    return energy_value(Molly.potential_energy(sys, neighbors; n_threads=1), FT)
end

function component_energy(sys, spec, neighbors, ::Type{FT}) where {FT <: AbstractFloat}
    subsys = if spec.kind == :pairwise_family
        Molly.System(sys; specific_inter_lists=(), general_inters=())
    elseif spec.kind == :specific_family
        Molly.System(sys; pairwise_inters=(), general_inters=())
    elseif spec.kind == :general_family
        Molly.System(sys; pairwise_inters=(), specific_inter_lists=())
    elseif spec.kind == :pairwise_component
        Molly.System(sys; pairwise_inters=(_component_values(sys.pairwise_inters)[spec.index],), specific_inter_lists=(), general_inters=())
    elseif spec.kind == :specific_component
        Molly.System(sys; pairwise_inters=(), specific_inter_lists=(_component_values(sys.specific_inter_lists)[spec.index],), general_inters=())
    elseif spec.kind == :general_component
        Molly.System(sys; pairwise_inters=(), specific_inter_lists=(), general_inters=(_component_values(sys.general_inters)[spec.index],))
    else
        throw(ArgumentError("Unsupported component kind $(spec.kind)"))
    end
    return total_energy(subsys, neighbors, FT)
end

function evaluate_system(sys, specs, ::Type{FT}) where {FT <: AbstractFloat}
    neighbors, neighbor_count = materialize_neighbors(sys)
    components = Dict{String, FT}()
    for spec in specs
        components[spec.label] = component_energy(sys, spec, neighbors, FT)
    end
    return (
        total=total_energy(sys, neighbors, FT),
        components=components,
        neighbor_count=neighbor_count,
        coords_type=string(typeof(sys.coords)),
        neighbor_finder_type=string(typeof(sys.neighbor_finder)),
        pairwise_signatures=[interaction_signature(inter) for inter in _component_values(sys.pairwise_inters)],
    )
end

function compare_variants(left_name::AbstractString, left, right_name::AbstractString, right, top_k::Int)
    deltas = Pair{String, Float64}[]
    all_labels = sort(collect(intersect(keys(left.components), keys(right.components))))
    for label in all_labels
        push!(deltas, label => (Float64(left.components[label]) - Float64(right.components[label])))
    end
    sort!(deltas; by=item -> abs(last(item)), rev=true)
    return (
        left=left_name,
        right=right_name,
        total_delta=Float64(left.total) - Float64(right.total),
        neighbor_count_delta=left.neighbor_count - right.neighbor_count,
        top_components=deltas[1:min(end, top_k)],
    )
end

function format_signature(fields::Vector{Pair{String, Any}})
    inner = join(["$(k)=$(format_num(v))" for (k, v) in fields], ", ")
    return "[$inner]"
end

function print_variant_block(io::IO, name::AbstractString, eval_data)
    println(io, "  $name")
    println(io, "    total = ", format_num(Float64(eval_data.total)))
    println(io, "    coords = ", eval_data.coords_type)
    println(io, "    neighbor_finder = ", eval_data.neighbor_finder_type)
    println(io, "    neighbor_count = ", eval_data.neighbor_count)
    for (idx, sig) in enumerate(eval_data.pairwise_signatures)
        println(io, "    pairwise[$idx] = ", format_signature(sig))
    end
end

function print_edge_block(io::IO, edge)
    println(io, "  $(edge.left) -> $(edge.right)")
    println(io, "    total_delta = ", format_num(edge.total_delta))
    println(io, "    neighbor_count_delta = ", edge.neighbor_count_delta)
    for (label, delta) in edge.top_components
        println(io, "    ", label, " = ", format_num(delta))
    end
end

function array_type_from_symbol(sym::Symbol)
    if sym == :cpu
        return Array
    elseif sym == :gpu
        CUDA.functional() || throw(ArgumentError("CUDA requested but unavailable"))
        return CuArray
    end
    throw(ArgumentError("Unsupported array type symbol `$sym`"))
end

function build_simulation_config(::Type{FT}, array_sym::Symbol) where {FT <: AbstractFloat}
    AT = array_type_from_symbol(array_sym)
    cfg = AWHGrads.default_simulation_config(; FT=FT, AT=AT)
    AWHGrads.apply_simulation_config!(cfg)
    return cfg
end

function build_base_system(pdb_path::AbstractString, ::Type{FT}, array_sym::Symbol) where {FT <: AbstractFloat}
    cfg = build_simulation_config(FT, array_sym)
    return Molly.System(
        pdb_path,
        AWHGrads.ff;
        array_type=cfg.AT,
        units=true,
        nonbonded_method=:cutoff,
    )
end

function clone_solute_atoms(base_atoms_cpu, solute_index_set::Set{Int}, lambda_value, ::Type{FT}) where {FT <: AbstractFloat}
    AtomT = typeof(first(base_atoms_cpu))
    atoms_new = AtomT[]
    λ = FT(lambda_value)
    for atom in base_atoms_cpu
        if atom.index in solute_index_set
            push!(atoms_new, Molly.Atom(atom.index, atom.atom_type, atom.mass, atom.charge, atom.σ, atom.ϵ, λ, Molly.InsertRole))
        else
            push!(atoms_new, Molly.Atom(atom.index, atom.atom_type, atom.mass, atom.charge, atom.σ, atom.ϵ, atom.λ, atom.alch_role))
        end
    end
    return atoms_new
end

function only_interaction(sys, T)
    matches = [inter for inter in _component_values(sys.pairwise_inters) if inter isa T]
    length(matches) == 1 || throw(ArgumentError("Expected exactly one $(T), found $(length(matches))"))
    return only(matches)
end

function build_softcore_system(base_sys, lambda_value, solute_indices, scheduler_name::Symbol, ::Type{FT}) where {FT <: AbstractFloat}
    scheduler = AWHGrads.resolve_lambda_scheduler(scheduler_name)
    awh_control = AWHGrads.AWHControlConfig()
    lj_0 = only_interaction(base_sys, Molly.LennardJones)
    coul_0 = only_interaction(base_sys, Molly.CoulombReactionField)
    lj_sc = AWHGrads.build_lj_softcore_interaction(lj_0, awh_control, scheduler, :gapsys, false)
    coul_sc = AWHGrads.build_coulomb_softcore_interaction(coul_0, awh_control, scheduler, :gapsys_rf, false)

    atoms_cpu = collect(Molly.from_device(base_sys.atoms))
    atoms_sc = clone_solute_atoms(atoms_cpu, Set(solute_indices), lambda_value, FT)
    return Molly.System(
        base_sys;
        atoms=Molly.to_device(atoms_sc, Molly.array_type(base_sys.coords)),
        pairwise_inters=(coul_sc, lj_sc),
    )
end

function build_transform_variants(sys)
    native_unitless_device = ustrip(sys)
    host = Molly.from_device(sys)
    host_unitless = ustrip(host)
    native_unitless_host = Molly.from_device(native_unitless_device)
    replay_host = AWHGrads._fix_interactions_for_cpu(host)
    replay = ustrip(replay_host)

    return [
        "native" => sys,
        "host" => host,
        "host_unitless" => host_unitless,
        "native_unitless_host" => native_unitless_host,
        "replay_host" => replay_host,
        "replay" => replay,
    ]
end

function print_header(io::IO, title::AbstractString)
    println(io)
    println(io, title)
    println(io, repeat("=", length(title)))
end

function analyze_system(io::IO, sys, label::AbstractString, ::Type{FT}, top_k::Int) where {FT <: AbstractFloat}
    specs = build_component_specs(Molly.from_device(sys))
    variants = Dict{String, Any}()
    for (name, variant_sys) in build_transform_variants(sys)
        variants[name] = evaluate_system(variant_sys, specs, FT)
    end

    print_header(io, label)
    println(io, "Variants")
    println(io, "--------")
    for name in ("native", "host", "host_unitless", "native_unitless_host", "replay_host", "replay")
        print_variant_block(io, name, variants[name])
    end

    println(io)
    println(io, "Edges")
    println(io, "-----")
    edge_order = [
        ("native", "host"),
        ("host", "host_unitless"),
        ("host_unitless", "native_unitless_host"),
        ("host", "replay_host"),
        ("host_unitless", "replay"),
    ]
    for (lhs, rhs) in edge_order
        print_edge_block(io, compare_variants(lhs, variants[lhs], rhs, variants[rhs], top_k))
    end
end

function main()
    opts = parse_cli(ARGS)

    cases = opts[:case] == :both ? (:base, :softcore) : (opts[:case],)

    for FT in opts[:float_types]
        for array_sym in opts[:array_types]
            base_sys = build_base_system(opts[:pdb], FT, array_sym)

            if :base in cases
                label = "case=base FT=$(FT) array=$(array_sym)"
                analyze_system(stdout, base_sys, label, FT, opts[:top_components])
            end

            if :softcore in cases
                for lambda_value in opts[:lambda_values]
                    softcore_sys = build_softcore_system(base_sys, lambda_value, opts[:solute_indices], opts[:scheduler], FT)
                    label = "case=softcore FT=$(FT) array=$(array_sym) lambda=$(lambda_value) scheduler=$(opts[:scheduler])"
                    analyze_system(stdout, softcore_sys, label, FT, opts[:top_components])
                end
            end
        end
    end
end

main()
