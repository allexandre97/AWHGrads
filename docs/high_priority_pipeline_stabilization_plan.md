# High-Priority Pipeline Stabilization And Logging Plan

## Summary

Implement the active pipeline fixes plus a logging cleanup pass:

- Make optimization steps softer when the latest validated solvent Stage B pass is marginal.
- Remove solvent near-pass probe shortening.
- Extend frozen Stage B probes in-place when the only failure is split-gap convergence.
- Add low-overhead confidence estimation to optimization using split-half statistics from frozen production artifacts.
- Reduce logging clutter by separating fresh events from cached status and suppressing repeated Stage B summaries.

This keeps the current macro workflow intact: no hard reject of parameter sets based on the next macro's Stage B result.

## Key Changes

### 1. Confidence-aware trust region from solvent Stage B health

Use the latest validated solvent Stage B result as a soft aggressiveness signal for the next optimization phase.

Implementation:

- Add a compact solvent Stage B health summary from the last validated pass:
  - supported parity margin
  - endpoint parity margin
  - split-gap margin
  - support coverage margin
- Collapse those into a bounded `stage_b_health_score in [0, 1]`.
- Feed that score into optimization before each epoch to scale:
  - effective `kl_target`
  - effective `max_phi_step_solute`
- Do not block optimization when health is poor; only reduce allowed move size.
- Use only the latest validated solvent pass, not failed probes and not vacuum.

Defaults:

- Full trust region when solvent margins are comfortably inside tolerance.
- Monotone shrink as the worst solvent margin approaches zero.
- Nonzero floor so optimization still progresses.

### 2. Remove near-pass probe shortening

Change the Stage B retry policy so solvent near-pass does not shorten the next probe.

Implementation:

- Update `stage_b_next_probe_policy` so solvent `near_pass` keeps the same probe length by default.
- Preserve existing cooldown behavior unless split-only extension takes over.
- Keep non-near-pass growth behavior unchanged.

Default:

- Solvent `near_pass` uses `next_probe_steps = current_probe_steps`.
- No `0.5x` retry path.

### 3. Extend Stage B in place on split-only failures

If a Stage B probe passes parity/support criteria but fails only on split-gap convergence, continue the same frozen-bias probe instead of returning to Stage A.

Implementation:

- Add an explicit split-only extension path in the readiness loop.
- Preserve the exact frozen reference:
  - same frozen AWH bias
  - same replay template definition
  - same probe identity
- Append new frames from a continuation segment to the existing probe dataset and recompute Stage B on the accumulated probe.
- Do not allow Stage A updates, bias evolution, or bias softening before the extension completes or is abandoned.
- Keep accumulation legal only while the bias is unchanged.
- Record in Stage B stats/logging:
  - accumulated frame count
  - number of probe segments
  - accumulation mode
- Exit the extension path and return to normal Stage A only if:
  - parity/support now fail materially, or
  - a configured max extension budget is exhausted.

Defaults:

- Trigger extension only for pure split-gap failure.
- Extension uses additive segment growth from the current probe length.
- Recompute Stage B on all accumulated frames, not only the newest segment.

### 4. Add split-half confidence estimation to optimization

Use split-half uncertainty from frozen production artifacts to temper optimization confidence.

Implementation:

- For each production artifact set, compute optimization diagnostics on:
  - full artifact set
  - first half
  - second half
- Derive uncertainty signals from half-vs-half disagreement for:
  - leg endpoint `dG`
  - cycle residual contribution
  - parameter gradient vector
- Use those signals to:
  - downscale effective trust-region size when disagreement is large
  - require stronger residual improvement before accepting a line-search proposal
  - log confidence metrics per epoch
- Keep the current ESS gates; confidence is an extra moderation layer, not a replacement.

Defaults:

- Split by time-ordered retained production frames after thinning/discard.
- Use deterministic halves, not bootstrap, for v1.
- If artifacts are too short for a meaningful split, fall back to current behavior and log reduced-confidence handling.

### 5. Improve logging and suppress cached-result spam

Refactor readiness and optimization logging so fresh work is obvious and repeated cached summaries are minimized.

Implementation:

- Separate logs into two classes:
  - event logs: probe started, probe extended, probe completed, probe failed, bias softened, optimization epoch accepted/rejected
  - status logs: lightweight periodic readiness state
- Stop printing the full cached Stage B summary every AWH block.
- Emit the full Stage B summary only when one of these happens:
  - a fresh Stage B probe completes
  - an accumulated split-only extension completes
  - the Stage B result changes materially
- For blocks with no new Stage B work, replace the verbose cached summary with a compact one-line marker such as:
  - last outcome
  - age in blocks since last fresh probe
  - cooldown / streak / pending extension state
- Track and compare a Stage B summary fingerprint so unchanged diagnostics are not re-logged.
- Make the optimization log reflect both:
  - current pre-step residual
  - accepted post-step residual
  so the log no longer looks like it stalled when a proposal actually improved.
- Keep the detailed diagnostics available behind a verbosity/config flag for debugging.

Defaults:

- Concise readiness logging on by default.
- Full Stage B diagnostics printed only on fresh probe events or changed outcomes.
- Optimization epoch summary prints accepted post-step residual when a step is taken.

## Public Interfaces / Config Additions

Add explicit config knobs so behavior is transparent and testable:

- Stage B health scaling:
  - enable/disable
  - minimum trust-region scale
  - margin-to-scale mapping parameters
- Near-pass retry behavior:
  - keep-same-length default for solvent
- Split-only extension:
  - enable/disable
  - max extension segments or total step budget
  - segment growth factor
- Optimization confidence:
  - mode: `:split_half`
  - minimum frames for confidence gating
  - confidence-to-step scaling strength
  - extra residual-improvement requirement from disagreement
- Logging cleanup:
  - readiness verbosity mode
  - Stage B repeat-suppression toggle
  - optimization summary verbosity

Keep defaults aligned with the current roadmap and avoid breaking existing scripts beyond changed retry behavior and cleaner logs.

## Test Plan

Add or update tests for the exact behavioral contracts:

- Readiness policy tests:
  - solvent near-pass no longer shortens probes
  - pure split-gap failure enters extension path
  - split-only extension appends frames under one frozen bias
  - extension path does not permit Stage A or bias evolution in between
  - parity/support failures still return to normal retry logic
- Optimization tests:
  - poorer solvent Stage B health shrinks effective `kl_target` / max step
  - strong solvent Stage B health leaves the trust region near baseline
  - split-half disagreement reduces accepted step size
  - proposals with marginal improvement are rejected when confidence is poor
  - fallback behavior works when production artifacts are too short to split
  - epoch logging includes accepted post-step residual when applicable
- Logging tests:
  - unchanged cached Stage B results are not reprinted every block
  - fresh probe completion still emits the full Stage B summary
  - materially changed Stage B outcomes re-log diagnostics
  - concise status logs still retain cooldown/streak visibility
- Regression coverage:
  - fixed-Gibbs-weight MBAR denominator behavior remains unchanged
  - no hard outer-loop parameter rejection is introduced

## Assumptions

- No hard accept/reject gate will be added at the macro boundary.
- The first confidence implementation is deterministic split-half only.
- For solvent near-pass retries, "keep same probe length" is the default policy.
- Split-only extension is valid only when the frozen bias is unchanged and the accumulated probe remains one stationary reference ensemble.
- Logging should optimize for human readability of long runs, with detailed cached diagnostics available only when explicitly requested or when results change.
