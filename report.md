The proposed plan provides a rigorous, physically sound architecture for managing extended-ensemble convergence prior to optimization. The separation of online heuristics (Stage A) from offline statistical mechanics validation (Stage B) is the correct approach to prevent unconverged FIM evaluations.

Here is a technical evaluation of the specific implementation details, along with potential edge cases that require attention.

### 1. Two-Stage Readiness Gate Validation
The division into Stage A and Stage B optimally balances computational cost with statistical rigor.
* Stage A (Online): Evaluating the trailing mean of ΔF, round-trips, and endpoint occupancy operates directly on the AWH logger data. This adds effectively zero overhead to the MD loop while guaranteeing that the system is topologically mixing across the full λ-coordinate.
* Stage B (Offline Probe): Computing the MBAR parity and split-half gap requires evaluating the reference Hamiltonian across all frames, which is computationally expensive. Gating this behind Stage A ensures the expensive Enzyme/autodiff or energy evaluation machinery is only invoked when the system is highly likely to be converged.

### 2. Operational Vulnerabilities to Address

A. The Length of the Stage B Probe (`awh_probe_time`)
You proposed a frozen-bias probe of `0.1 ns`. While 0.1 ns is computationally cheap, it is likely too short to generate a statistically reliable MBAR estimate for the solvent leg.
* Issue: If the probe is too short, the empirical MBAR free energy profile will be dominated by high-variance local solvent fluctuations. This will cause the `max|F_MBAR(λ) - F_AWH(λ)| <= awh_parity_tol_kT` check to fail due to finite-sampling noise, not because the underlying AWH bias is unconverged.
* Solution: Decouple the probe length from the AWH update block length. If Stage A passes, run a slightly longer probe (e.g., 0.5 ns - 1.0 ns). If Stage B fails, this data is discarded, the bias is unfrozen, and the system returns to Stage A. 

B. Lockstep Advancement Inefficiency
The plan states: "Keep both legs advancing together until both pass (no one-leg freeze policy)."
* Issue: The vacuum leg typically converges an order of magnitude faster than the explicit solvent leg (e.g., 385 iterations vs. 3436 iterations in your previous logs). Forcing the vacuum leg to continuously simulate while waiting for the solvent leg to pass Stage B consumes unnecessary GPU cycles. While vacuum is cheap, continuous AWH updating on an already converged vacuum system can induce random walk drift in its bias.
* Solution: If the vacuum leg passes Stage A and Stage B, you should freeze its parameters and bias, and halt its simulation loop entirely. Only resume it if the solvent leg eventually exhausts its 50 ns budget and the macro-epoch resets.

C. Parity Alignment Constraint
When implementing the `max|F_MBAR(λ) - F_AWH(λ)|` parity check, the absolute values of the free energies are arbitrary; only the relative differences matter.
* Implementation detail: Before computing the maximum deviation, both profiles must be strictly zeroed at a reference state (e.g., `F_MBAR[1] = 0.0` and `F_AWH[1] = 0.0`). Furthermore, the MBAR profile must be computed using the exact same thermodynamic definition as the AWH update (incorporating the PV terms for NPT solvent legs).

### 3. Budget Exhaustion Fallback
Skipping the optimization phase if the 50 ns budget is exhausted (`phase2_exit_reason = :awh_not_ready_budget`) is the correct mathematical fail-safe. If the system cannot converge the AWH bias within 50 ns, the proposed force field parameters ($\theta_n$) have likely pushed the system into a glassy or severely clash-prone region of phase space where ergodic sampling is impossible. Retaining the parameters and attempting a new macro-epoch with a fresh velocity distribution is the safest recovery mechanism.