#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))
AWHGrads.run_pipeline()
