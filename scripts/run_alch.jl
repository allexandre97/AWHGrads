#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))
# Run with Julia 1.11:
#   julia +1.11 --project=. scripts/run_alch.jl [optional_config.jl]

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
        throw(ArgumentError("Config file must return `SimulationConfig`, `OptimizationConfig`, or a NamedTuple with `sim_cfg` and/or `opt_cfg`."))
    end

    return sim_cfg, opt_cfg
end

sim_cfg = AWHGrads.default_simulation_config()
opt_cfg = AWHGrads.default_optimization_config()

if !isempty(ARGS)
    cfg_path = abspath(ARGS[1])
    sim_cfg, opt_cfg = load_configs(cfg_path)
end

AWHGrads.run_pipeline(; sim_cfg=sim_cfg, opt_cfg=opt_cfg)
