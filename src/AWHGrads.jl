module AWHGrads

using Molly
using CUDA
using Unitful
using StatsBase
using LinearAlgebra

# Defaults are overwritten by `apply_simulation_config!`.
FT = Float32
AT = CuArray
Δt = FT(1)u"fs"
T0 = FT(310)u"K"
P0 = FT(1)u"bar"
lambda_schedule = FT.(range(1.0, stop=0.0, length=21))
num_lambda_states = length(lambda_schedule)
target_rho = FT(1.0 / num_lambda_states)

_data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
_ff_dir = joinpath(_data_dir, "force_fields")
ff = MolecularForceField(FT, joinpath.(_ff_dir, ["tip3p_standard.xml", "gaff.xml", "ethanol.xml"])...; units=true)

include("types.jl")
include("config.jl")
include("gradients_core.jl")
include("setup.jl")
include("logging_utils.jl")
include("ensemble_eval.jl")
include("readiness.jl")
include("transforms.jl")
include("index_maps.jl")
include("optimization.jl")
include("pipeline.jl")

export SimulationConfig
export OptimizationConfig
export RuntimeState
export LegArtifacts
export StageAStats
export StageBStats
export default_simulation_config
export default_optimization_config
export run_pipeline

end
