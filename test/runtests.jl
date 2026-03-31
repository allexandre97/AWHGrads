using Test

include(joinpath(@__DIR__, "..", "src", "AWHGrads.jl"))
using .AWHGrads

include("test_transforms.jl")
include("test_readiness.jl")
include("test_free_energy_estimators.jl")
include("test_pipeline.jl")
