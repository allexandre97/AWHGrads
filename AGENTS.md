# Repository Guidelines

## Project Structure & Module Organization
`src/AWHGrads.jl` is the module entry point and export list. Keep it thin; add implementation to focused files such as `config.jl`, `readiness.jl`, `optimization.jl`, and `pipeline.jl`.

`scripts/` contains runnable workflows and configs, including `run_alch.jl`, `run_alch_full_example.jl`, and benchmark/example config files. `test/` contains regression suites loaded by `test/runtests.jl`; add new suites there and include them from the test harness.

The repository root also holds sample inputs like `ethanol_vac.pdb` and `ethanol_solv.pdb`, plus generated logs and reference PDFs. Do not commit large generated outputs unless they are intentionally part of the repo.

## Build, Test, and Development Commands
Use Julia 1.11 for all local work.

`julia +1.11 -e 'using Pkg; Pkg.instantiate()'` installs dependencies from `Project.toml` and `Manifest.toml`.

`julia +1.11 test/runtests.jl` runs the full regression suite.

`julia +1.11 scripts/run_alch.jl` runs the default AWH pipeline.

`julia +1.11 scripts/run_alch.jl scripts/example_config.jl` runs the pipeline with explicit `sim_cfg` and `opt_cfg` overrides.

`julia +1.11 scripts/run_alch_full_example.jl` exercises the longer example and writes `logs.log`.

## Coding Style & Naming Conventions
Use 4-space indentation and standard Julia style. Follow existing naming patterns: `CamelCase` for structs and config types such as `SimulationConfig`, `snake_case` for functions such as `run_pipeline`, and descriptive `Symbol` names for cycle legs such as `:vacuum` and `:solvent`.

Preserve the current `Base.@kwdef` config style and add short docstrings when touching nontrivial scientific logic. Keep shared defaults and exports centralized, but move new behavior into the specialized files under `src/`.

## Testing Guidelines
Name test files `test_<feature>.jl`. Register each new suite in `test/runtests.jl`.

Prefer deterministic unit and regression coverage over long GPU-heavy runs. When changing readiness checks, transforms, or optimization logic, add tests for both nominal behavior and numerical edge cases.

## AWH and Reweighting Notes
Treat Stage B solvent parity failures as a high-signal diagnostic. In current examples, the main discrepancy appears in the late Lennard-Jones decoupling tail rather than in the vacuum leg, so inspect the solvent `λ` profile before relaxing readiness thresholds.

Keep AWH/MBAR math aligned with Molly’s expanded-ensemble implementation. The reference mixture denominator must use the fixed Gibbs log-weights `g_ref = f_ref + logρ_ref`, matching Molly’s `process_sample`; use sampled `active_lambda_idx` only when reweighting the full extended state at the physically visited `λ`.

Stage B probes must keep the AWH bias frozen while collecting frames. If bias updates are allowed during a probe, MBAR/AWH parity checks stop being interpretable because the probe no longer samples a single reference mixture.

Do not assume a large parity gap is caused by CPU versus GPU neighbor handling. Simulation runs use Molly’s `GPUNeighborFinder`, while offline ensemble evaluation converts that state to a CPU `DistanceNeighborFinder` with the same buffered cutoff and exception masks. The physical LJ/Coulomb cutoff is still applied by the interaction itself, so investigate frozen-bias quality, λ-tail sampling, and state-profile consistency before blaming neighbor-list drift.

## Code Map and Mental Model
When reasoning about this repository, treat it as a three-layer system.

`src/setup.jl` builds the thermodynamic states and the Molly/AWH simulations. It resolves per-leg lambda schedules, electrostatics choices, soft-core models, thermostat/barostat coupling, restart reuse, and warm-started bias injection.

`src/readiness.jl`, `src/gradients_core.jl`, and `src/ensemble_eval.jl` contain the scientific logic. `readiness.jl` decides whether a leg is trustworthy enough to freeze and optimize against. `ensemble_eval.jl` replays logged frames against every lambda state on CPU, unitless templates. `gradients_core.jl` implements the actual reweighting, MBAR-style profile reconstruction, gradient accumulation, Fisher matrix assembly, and parity metrics.

`src/optimization.jl` and `src/pipeline.jl` are the control layer. `optimization.jl` turns replayed energies and gradients into bounded parameter updates. `pipeline.jl` orchestrates macro epochs, readiness, production artifact capture, optimization, and warm-started resimulation.

The tests are important here. `test/test_readiness.jl` is effectively the executable specification for Stage A, Stage B, support-aware parity gating, and frozen-bias semantics. `test/test_free_energy_estimators.jl` pins down the fixed-Gibbs-weight MBAR denominator. `test/test_ensemble_eval.jl` checks that cached replay, templating, and Enzyme differentiation are consistent. The long scripts under `scripts/` are not just runners; several of them are diagnostics for logged-vs-replayed energy consistency and GPU-tile replay equivalence.

## Detailed Algorithm Walkthrough
The pipeline optimizes Lennard-Jones parameters for an alchemical thermodynamic cycle, usually solvent plus vacuum, by alternating between sampling and reweighting. The default cycle is assembled in `config.jl`, while `pipeline.jl` drives the repeated macro-epoch workflow.

Each thermodynamic leg is a lambda-expanded ensemble. A state index `k` corresponds to a lambda value and therefore to a Hamiltonian `U_k(x; θ)` with current physical parameters `θ`. The solvent leg normally uses a staged path: electrostatics are removed first, then Lennard-Jones interactions are decoupled with extra density in the late LJ tail. The default dense solvent schedule is deliberately concentrated around the problematic low-LJ region rather than being uniformly spaced.

At the start of a run, the code extracts trainable LJ parameters from the chosen reference leg, typically the solvent leg. The optimization variables are stored in an unconstrained latent vector `ϕ`, then mapped into physically bounded parameters `θ` by a shifted sigmoid transform in `transforms.jl`. The chain-rule factor `dθ/dϕ` is always needed later when converting physical gradients back to optimization-space gradients.

For each macro epoch, `pipeline.jl` calls `setup_macro_legs`, which builds a fresh AWH simulation for each leg. On the first macro epoch this is a cold start. On later epochs the code can reuse restart coordinates and velocities, and it can also inject the previous AWH bias if and only if the run is a true warm start. Tests explicitly guard this behavior: a stored bias must not be reused on a cold restart with no restart state.

Readiness happens before any optimization. The readiness loop alternates between Stage A blocks and occasional Stage B probes.

Stage A is a cheap AWH-only screening phase. It does not replay energies and does not claim physical correctness by itself. Instead it asks whether the expanded ensemble looks mixed and statistically stable. The metrics are: recent mean bias-change magnitude `df_mean`, lambda-history ESS, Molly's linear-stage `N_eff`, full round trips between the coupled and decoupled ends, recent endpoint occupancy, and for solvent legs a focused occupancy floor on the late LJ-tail states. A leg is Stage-A-ready only when all configured thresholds are satisfied on the recent history window. Stage A is therefore a necessary but not sufficient condition for readiness.

Stage B is the decisive consistency check. The current leg is cloned, the AWH bias is frozen, all bias-update accumulators are cleared, and a short frozen-bias probe simulation is run. The probe frames are then thinned and optionally partially discarded. For each retained frame, `ensemble_eval.jl` rebuilds CPU, unitless templates for every lambda state and evaluates the frame under every state Hamiltonian. The result is an energy matrix `u_ref[n, k]` giving the replayed reduced energies of frame `n` under lambda state `k`.

Stage B then reconstructs the reference expanded-ensemble mixture using the frozen AWH Gibbs log-weights `g_ref(k) = f_ref(k) + logρ_ref(k)`. The crucial denominator is

`log_mix_ref(n) = logsumexp_k(g_ref(k) - β * u_ref(n, k) - β * P0 * V_n)`

for NPT solvent legs, with the `P0 V` term omitted where not applicable. This is the central mathematical contract in the codebase. It must match Molly's expanded-ensemble semantics, and the tests explicitly distinguish it from incorrect variants that use only `f_ref`.

From this frozen reference mixture, the code computes two independent Stage B checks.

The split-half check compares endpoint free energies computed from the first and second halves of the same frozen-bias probe. This measures whether the probe itself is internally converged.

The parity check compares the MBAR-style free-energy profile reconstructed from replayed probe frames against the frozen AWH profile that generated those frames. The comparison is alignment-invariant because both profiles are shifted to a common reference state before taking residuals. The main parity statistic is a max-absolute residual over lambda states.

The default gate is not a naive raw max. It is support-aware. `compute_state_reweighting_ess_from_log_mixture_denom` estimates the per-state reweighting support, then Stage B computes a supported parity gap over only those states whose ESS exceeds the configured support threshold. This prevents a single essentially unsampled state from dominating the diagnostic. However, the gate still enforces minimum support coverage and a separate endpoint parity requirement, so a run is not allowed to pass simply because the worst state was masked away.

Stage B therefore produces several distinct outcomes. A leg can pass, fail on split-half convergence, fail on endpoint parity, fail on supported parity, or fail on low support. It can also be marked as a near-pass, which means the probe is close enough that the next retry policy may shorten the probe instead of growing it. The retry logic lives in `stage_b_next_probe_policy` and `stage_b_retry_controls`, and it adjusts probe length, Stage A streak length, and cooldown based on repeated failures. Bias softening can also be triggered after repeated failures by reducing the effective `N_eff` target to make future AWH adaptation less rigid.

An important invariant is that Stage B statistics are single-probe diagnostics. The code and tests explicitly reject the idea of mixing frames from different frozen biases as if they came from one stationary reference distribution. If the bias changes, the old probe and the new probe are not one ensemble and must not be merged for parity purposes.

Only after all legs pass readiness does the pipeline collect production artifacts. This uses another frozen-bias clone, now for a longer production segment. The logged trajectory, volumes, active lambda history, replay neighbors, and replay energy matrix are packaged into `LegArtifacts`. Optimization is done only from these frozen-bias production artifacts, never from an actively adapting bias.

## Optimization Math
The optimization target is the cycle free energy. Each leg contributes a coefficient-weighted endpoint free energy, and an optional standard-state correction can be added for the cycle. The code computes a predicted cycle free energy from the current parameters and compares it against the experimental target `dG_exp`.

For a given leg, the endpoint free energy and its parameter gradient are computed by reweighting the frozen-bias production frames. The replayed frames are always interpreted relative to the same frozen reference mixture that generated them. For full extended-state weights, the sampled active lambda index is used where appropriate, because the frame was physically visited at that lambda. For endpoint free energies and full-profile MBAR reconstruction, the fixed reference Gibbs weights must remain in the denominator. This distinction is deliberate and tested.

The code accumulates weighted score-function gradients and a weighted Fisher information matrix from the replayed frames. These are physical-space gradients with respect to `θ`, then mapped back to latent-space gradients with the chain-rule multiplier from `transforms.jl`. The optimization never steps directly in unconstrained raw LJ parameter space without respecting bounds.

The scalar residual is the difference between predicted and target cycle free energies, optionally passed through a Huber loss derivative so that very large residuals do not create unstable updates. Multi-parameter updates are preconditioned by a regularized Fisher matrix. The implementation rescales by the Fisher diagonal, truncates weak eigenmodes, and then chooses a step size using a KL-style trust-region target. A hard infinity-norm cap on the latent step prevents any single parameter from moving too far in one epoch.

The optimizer is conservative about accepting steps. It line-searches on the residual while enforcing ESS thresholds. A proposed update is accepted only if it improves the residual enough relative to the configured noise tolerance and does not destroy the reweighting support. If no acceptable step is found, the code restores the best inner-loop state rather than forcing a poor update. By default the optimization block is solute-only, so solvent atom-type parameters remain invariant unless `optimize_solvent=true`.

## Practical Interpretation Rules
Do not treat Stage A success as evidence that the physics is correct. Stage A only says that the adaptive bias seems statistically settled. Stage B is the actual self-consistency test against a frozen reference mixture.

If vacuum behaves well while solvent fails Stage B, that usually points to a solvent-path or solvent-sampling problem rather than a general implementation bug. In the current examples, repeated failures tend to localize in the late Lennard-Jones tail.

If optimization dramatically improves the cycle residual but the next macro epoch fails solvent Stage B, the monitoring logic is doing its job. That pattern means the reweighting objective found a parameter move that fits the target on frozen-bias artifacts, but fresh solvent resimulation does not reproduce a self-consistent free-energy profile there.

Before blaming neighbor lists or CPU-vs-GPU replay drift, use the diagnostic scripts. `scripts/check_energy_consistency.jl` compares logged active-state energies, direct recomputation, and replay-matrix values. `scripts/check_term_energy_consistency.jl` decomposes native-vs-replay deltas by interaction family and lambda. `scripts/check_gpu_tile_replay.jl` isolates whether disagreement first appears in the GPU tiled pairwise path, in `from_device`, or in the CPU replay templates. `probas.jl` explores the transform chain `native -> from_device -> ustrip -> _fix_interactions_for_cpu`.

When changing this project, preserve these invariants unless you are intentionally redesigning the method and updating the tests accordingly:

The frozen-bias Stage B probe must remain frozen.

The MBAR denominator must use `g_ref = f_ref + logρ_ref`.

Support-aware parity must still enforce support coverage and endpoint sanity, not just supported-state agreement.

Optimization must consume frozen-bias production artifacts, not actively adapting AWH data.

Warm-start bias reuse must depend on restart continuity, not just the existence of cached bias values.

## Prioritized Improvements
These are the current implementation priorities based on the present `logs.log` behavior and should be treated as the active short-term roadmap.

### High Priority
Do not add a hard outer-loop rule that rejects a parameter set simply because the next fresh solvent macro has a poor Stage B result. Inner-loop progress can still move parameters in the correct long-term direction even when the next fresh resimulation is temporarily fragile.

Make the optimization trust region depend on solvent Stage B health. A barely passing solvent Stage B result should reduce how aggressively the next optimization phase can move `ϕ`, even if ESS and the artifact residual look good. The practical targets here are `kl_target`, `max_phi_step_solute`, and related step-size controls in `optimization.jl`.

Remove the current solvent policy that shortens the next probe after a Stage B near-pass. In the observed run, a near-pass followed by a shorter `1.5 ns` probe immediately failed on split convergence. For solvent, near-pass retries should keep the same probe length or grow modestly rather than shrink.

When the only Stage B failure mode is split-gap failure under a frozen bias, continue sampling that same frozen probe instead of returning to Stage A and allowing the bias to evolve. This must preserve the single-frozen-reference semantics of Stage B: appending frames is valid only while the bias remains unchanged.

Add confidence estimation to the optimization objective and gradients. The optimizer currently reacts to replayed artifacts as though they were noiseless. Add uncertainty estimates for endpoint `ΔG`, residual improvement, and gradient quality, for example by block/bootstrap or split-half methods on frozen-bias production frames, and use them to moderate step acceptance and step size.

### Secondary
Increase solvent-side trust-region conservatism further if the last validated Stage B pass is only marginal, but keep this as a softer scaling effect rather than a hard accept/reject gate.

Consider longer or asymmetric production artifact collection for solvent relative to vacuum. Vacuum is already cheap and stable, while solvent dominates the uncertainty and fragility.

Add more targeted Stage A or pre-Stage-B solvent diagnostics for repeatedly problematic endpoint states, especially `λ20-λ21`, so that obvious endpoint-support weakness is detected earlier.

Consider adaptive lambda refinement driven by supported Stage B residuals. The current dense solvent schedule is tail-focused, but the final passing Macro 3 probe had its worst supported residual in the internal charge/LJ-transition region rather than at the endpoint.

Consider adding a profile-shape regularization term or related safeguard so the optimizer does not improve only the scalar cycle `ΔG` while pushing the solvent profile into a less robust resimulation regime.

Keep using the existing replay diagnostic scripts before attributing failures to CPU/GPU or neighbor-list drift. The current evidence still points first to frozen-bias profile quality and solvent-path consistency.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects, for example `Retune optimization trust region and block scheduling`. Follow that pattern.

Pull requests should explain the scientific or behavioral change, list the Julia commands you ran, and call out any new config assumptions or generated artifacts. Include representative log snippets when pipeline behavior, convergence thresholds, or staging logic changes.
