"""
    AWHGrads

Utilities for alternating between AWH sampling and parameter optimization for a
thermodynamic cycle. The package is organized around three phases:

1. Configure the cycle, force field, and optimization settings.
2. Run AWH until each leg passes readiness checks.
3. Reweight the production data and update the trainable Lennard-Jones
   parameters.
"""
module AWHGrads

using Molly
using CUDA
using Unitful
using StatsBase
using LinearAlgebra
using Logging
using Enzyme

"""
    default_nonbonded_energy_type(FT=Float32)

Return the default precision Molly should use when accumulating nonbonded
potential energies for simulations running with scalar type `FT`.
"""
default_nonbonded_energy_type(::Type{FT}=Float32) where {FT <: AbstractFloat} = promote_type(FT, Float64)

# Defaults are overwritten by `apply_simulation_config!` so helper functions can
# share the active precision/backend without threading config objects through
# every call.
FT = Float32
AT = CuArray
nonbonded_energy_type = default_nonbonded_energy_type(FT)
Δt = FT(1)u"fs"
T0 = FT(293.15)u"K"
P0 = FT(1)u"bar"
lambda_schedule = FT.(range(1.0, stop=0.0, length=21))
num_lambda_states = length(lambda_schedule)
target_rho = FT(1.0 / num_lambda_states)

include("types.jl")
include("config.jl")
include("parameter_pools.jl")
ff = MolecularForceField(FT, resolve_force_field_paths(default_simulation_config())...; units=true)
include("charges.jl")
include("gradients_core.jl")
include("setup.jl")
include("logging_utils.jl")
include("ensemble_eval.jl")
include("observables.jl")
include("readiness.jl")
include("transforms.jl")
include("index_maps.jl")
include("optimization.jl")
include("pipeline.jl")

export SimulationConfig
export OptimizationConfig
export ForceFieldConfig
export AWHControlConfig
export EnsembleEvalConfig
export StageAReadinessPolicyConfig
export ThermodynamicLegConfig
export ThermodynamicCycleConfig
export AbstractTrainingTarget
export CycleFreeEnergyTarget
export StateObservableTarget
export ResolvedCycleFreeEnergyTarget
export ResolvedStateObservableTarget
export ParameterPoolConfig
export ChargeTrainingConfig
export ParameterBoundsConfig
export RuntimeState
export LegArtifacts
export StageAStats
export StageBStats
export MassDensityObservable
export DielectricConstantObservable
export default_simulation_config
export default_optimization_config
export default_cycle_config
export default_solvent_leg_lambda_schedule
export dense_solvent_leg_lambda_schedule
export default_nonbonded_energy_type
export resolved_cycle_config
export validate_cycle_config
export validate_lambda_schedule
export effective_nonbonded_energy_type
export default_energy_analysis_type
export awh_logger_value_types
export resolve_parameter_reference_leg
export resolve_training_targets
export resolve_force_field_paths
export simulation_config_with
export optimization_config_with
export run_pipeline
export setup_logging

end
