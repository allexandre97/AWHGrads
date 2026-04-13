# Charge Training Support Plan

## Summary
Extend AWHGrads and Molly so the trainable parameter vector can include
charge-model parameters in addition to LJ parameters, with final partial
charges obtained from the analytic constrained charge-equilibration solve
from `d2sc02739a.pdf`. V1 will use a transferable but simple model: predict
per-atom electronegativity `e_i` and positive hardness `s_i` from Molly atom
classes, then solve charges exactly under a per-molecule net-charge
constraint. Joint LJ + charge optimization will be supported from the start.
Molly should gain first-class APIs for charge-aware atom injection and
charge-sensitive system rebuilding rather than relying on AWHGrads-only
shims.

## Key Changes

### AWHGrads parameterization and transforms
- Generalize the parameter state built in `initialize_parameter_state` so
`theta_ref`, `param_names`, trainable subsets, and per-leg index maps can
include:
- LJ `σ` and `ϵ` parameters, still shared by atom type as today.
- Charge-model parameters keyed by atom class: one electronegativity
parameter and one hardness parameter per class.
- Replace the current single “bounded sigmoid for every parameter”
assumption with parameter-family transforms:
- LJ keeps the existing bounded sigmoid map.
- Electronegativity is unbounded or lightly centered and does not use the
LJ box transform.
- Hardness uses a strictly positive map such as `softplus(h_raw) +
hardness_floor`.
- Introduce parameter metadata so optimization and logging know each
parameter family, its transform, and whether it is shared by type or class.
- Add a charge-model config section under simulation/parameter config with
explicit defaults:
- `enable_charge_training::Bool`
- `charge_typing_basis::Symbol = :class`
- `hardness_floor`
- `formal_charge_source`
- `joint_charge_lj_optimization::Bool = true`
- Keep the current LJ-only path intact when charge training is disabled.

### Charge-equilibration model and charge constraints
- Add a charge-model module that, for each system being evaluated, computes
per-atom `e_i` and `s_i` from shared class-level parameters and then solves
the analytic constrained problem
- `min Σ_i (e_i q_i + 1/2 s_i q_i^2)`
- subject to `Σ_{i in molecule m} q_i = Q_m`
- Use one independent linear constraint per molecule, with `Q_m` taken from
the molecule’s reference formal/net charge as loaded by Molly or inferred
from the reference force-field charges.
- Implement the solve in a batched, differentiable way so Enzyme sees a
smooth map from charge-model parameters to atom charges and energies.
- Return both solved charges and the Jacobian support needed by the
optimization code, but keep the public interface simple:
- “given system metadata and parameter vector, return atom charges”
- Do not add residue-group or arbitrary linear constraints in v1.

### AWHGrads system setup, replay, and optimization
- Extend per-leg index maps so AWHGrads can project the global parameter
vector onto:
- per-atom LJ indices
- per-atom charge-model indices
- per-atom molecule membership and net-charge constraints
- Update setup/rebuild paths so both live simulation systems and offline
replay systems receive recomputed atom charges whenever the parameter vector
changes.
- Make replay/evaluation construct the full atom set from the parameter
vector, not just swap `σ` and `ϵ`; this should feed both Stage B and
production-artifact evaluation.
- Keep the current Stage B and MBAR denominator semantics unchanged; charge
training only changes the Hamiltonian used in replay and simulation.
- Extend optimization blocks so joint LJ + charge training works from day
one:
- add a charge block kind and a mixed block kind
- allow confidence scaling, KL control, and line search to operate on the
combined parameter vector
- Keep minimal regularization in v1:
- no explicit reference-charge penalty
- rely on transform safety, positivity of hardness, Fisher
preconditioning, trust region, and line search
- Because this is intentionally lightly regularized, add hard monitoring
diagnostics:
- per-molecule charge sums
- max per-atom charge drift from reference
- hardness minima
- largest class-level `e`/`s` updates

### Molly.jl API changes
- Extend Molly’s parameter-injection API so atom charge can be injected from
a dense parameter vector, not only mass/`σ`/`ϵ`.
- Add a public atom-injection form that supports at least:
- `mass`
- `charge`
- `σ`
- `ϵ`
- Add matching parameter-index extraction helpers so downstream code can
build charge-aware atom maps without custom private assumptions.
- Ensure `System` rebuilding remains correct for Coulomb cutoff, reaction
field, Ewald, and PME when atom charges change:
- rebuilding must refresh any charge-dependent buffers or state
- PME/Ewald cached structures may be reused only where they are charge-
independent
- Add Molly-level tests for:
- charge injection on CPU and GPU-backed systems
- differentiable energy evaluation with charge updates
- correctness of Ewald/PME energy changes after charge updates
- Prefer small public API additions over AWHGrads-specific hooks so this
- agreement of replayed energies when only charge parameters change
- Regression tests in AWHGrads:
- a tiny system with known molecular net charges where joint LJ + charge evaluation runs end-to-end
- Stage B frozen-bias probe still behaves identically when charge training is disabled
- joint optimization updates both LJ and charge-model parameters without violating charge
constraints
- Unit tests in Molly:
- `inject_atom` or successor API updates charge correctly
- rebuilt systems with changed charges produce expected Coulomb energy deltas
- PME/Ewald paths do not retain stale charge-grid state after charge updates
- Acceptance criteria:
- disabling charge training reproduces current behavior
- enabling charge training preserves exact molecular net charges at every evaluation
- mixed-parameter gradients are finite and usable by the existing optimizer
- no code path assumes every trainable parameter uses the LJ sigmoid bounds

## Assumptions and Defaults
- V1 charge predictors are shared by Molly atom class, not by exact atom type.
- V1 constraints are per-molecule net-charge constraints only.
- V1 supports all molecules that are trainable under the current optimization scope, not only the
solute.
- Joint LJ + charge optimization is part of the initial implementation.
- Minimal regularization is intentional for v1, but the implementation should leave a clean insertion
point for later priors on charges or on `e_i`/`s_i`.