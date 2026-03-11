# Refactor Outline: `AWH_Grads`

## Goal
Split `main_alch.jl` and `gradients.jl` into a small Julia package-style layout with:
- clear module boundaries,
- one lightweight run script,
- testable pure functions separated from simulation orchestration.

## Current Pain Points
- `main_alch.jl` currently mixes constants/config, setup, logging helpers, readiness checks, ensemble evaluation, optimization math, and top-level execution loop.
- Global mutable state (`active_bias_*`, `phi_active`, `theta_active`, restart caches) is shared through a long script-level scope.
- `gradients.jl` is already reasonably cohesive, but is tightly coupled to how `main_alch.jl` builds index maps and calls it.

## Target Layout
```text
AWH_Grads/
  Project.toml
  src/
    AWHGrads.jl
    config.jl
    setup.jl
    logging_utils.jl
    readiness.jl
    ensemble_eval.jl
    gradients_core.jl
    transforms.jl
    optimization.jl
    pipeline.jl
    types.jl
  scripts/
    run_alch.jl
  test/
    runtests.jl
    test_transforms.jl
    test_readiness.jl
    test_free_energy_estimators.jl
```

## Module Responsibilities

### `src/types.jl`
- Define stable data containers to reduce global state:
  - `SimulationConfig`
  - `OptimizationConfig`
  - `RuntimeState` (current `phi/theta`, biases, restart caches)
  - `LegArtifacts` (logger, neighbors, reference energies, etc.)
  - `StageAStats` and `StageBStats`

### `src/config.jl`
- Move constants and defaults currently at top of `main_alch.jl`:
  - `FT`, `AT`, `Δt`, `T0`, `P0`
  - schedules, tolerances, budget/probe durations, optimizer hyperparameters.
- Build and return config structs instead of mutating globals.

### `src/setup.jl`
- System construction and parameter-index mapping:
  - `setup_alchemical_awh`
  - `capture_restart_state`
  - `build_index_maps`
  - helper for trainable parameter initialization (`theta_ref`, bounds, `phi_0`, index maps).

### `src/logging_utils.jl`
- Logger and history utilities:
  - `get_production_logger`
  - `clear_awh_logger_history!`
  - `clear_awh_logger_histories!`
  - `get_awh_active_idx_history`
  - `steps_to_ns`

### `src/readiness.jl`
- Convergence/readiness logic:
  - `awh_linear_stage_stats`
  - `count_full_round_trips`
  - `endpoint_occupancy_fractions`
  - `evaluate_stage_a_readiness`
  - `run_stage_b_probe`
  - `split_half_ranges`
  - `estimate_leg_dg_from_reference`

### `src/ensemble_eval.jl`
- Frame-level evaluation orchestration:
  - `precompute_neighbors`
  - `evaluate_ensemble`
- Keep threading details localized to this module.

### `src/gradients_core.jl`
- Move current `gradients.jl` here with minimal edits:
  - Enzyme/Molly functional injectors
  - `evaluate_frame_energy`, `evaluate_frame_gradients`
  - MBAR/FIM estimators (`compute_weights_and_ess`, `compute_empirical_gradients_and_fim`, `compute_global_endpoint_gradients`, `compute_full_mbar_profile`, `compute_parity_gap`).

### `src/transforms.jl`
- Parameter transform utilities:
  - `map_phi_to_theta`
  - `get_chain_rule_multiplier`
  - any clipping/bounds helper functions.

### `src/optimization.jl`
- Inner-loop natural-gradient optimization step:
  - active block selection (solute/solvent),
  - FIM preconditioning and pseudo-inverse,
  - KL trust region scaling,
  - infinity-norm clipping,
  - line search and acceptance criteria.
- Output updated parameters + diagnostics, without running MD.

### `src/pipeline.jl`
- High-level orchestration:
  - macro-epoch loop,
  - Stage A/B gating,
  - production runs,
  - call optimization step,
  - convergence/failure exit reasons.
- This should be the only module that coordinates both legs end-to-end.

### `src/AWHGrads.jl`
- Main package module that `include`s and exports public entrypoints:
  - `run_pipeline`
  - config constructors
  - diagnostics output helpers.

### `scripts/run_alch.jl`
- Minimal executable script:
  - parse run options (or hard-coded defaults initially),
  - instantiate configs,
  - call `AWHGrads.run_pipeline`.

## Refactor Sequence (Low Risk)

1. **Create package skeleton** (`src/`, `scripts/`, `test/`) and move files without behavior changes.
2. **Extract pure utilities first** (`transforms`, readiness scalar helpers, MBAR helpers) and add unit tests.
3. **Extract evaluation layer** (`ensemble_eval`, `gradients_core`) and keep interfaces identical to current code.
4. **Wrap global state** into `RuntimeState`; remove script-level globals.
5. **Extract optimization loop** into `optimization.jl`, returning explicit state deltas/metrics.
6. **Implement `run_pipeline`** in `pipeline.jl`; keep script as thin wrapper.
7. **Only then simplify** duplicated orchestration blocks for solvent/vacuum via shared leg runner function.

## Immediate Next Commits

1. `chore: scaffold src module tree and run script`
2. `refactor: move gradients.jl to src/gradients_core.jl with same API`
3. `refactor: extract readiness and logging helpers`
4. `refactor: move orchestration into pipeline.run_pipeline`
5. `test: add unit tests for transforms/readiness/mbar estimators`

## Guardrails During Refactor
- Keep all numerics in `Float32` unless intentionally changed.
- Preserve current AWH acceptance thresholds and readiness criteria exactly in early commits.
- Add regression checks for:
  - split-gap/parity calculations,
  - ESS thresholds,
  - line-search acceptance decisions.
- Do not combine logic changes with structural moves in the same commit.
