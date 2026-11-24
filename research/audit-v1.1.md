# AIR-002 V1.1 scientific audit

## Scope and evidence

Baseline: `766978bd7da226e6ba34103e9cc972aa9f596e3c`. The original [hosted CI](https://github.com/MarkRizam/mathematics-for-ai/actions/runs/45441371359) completed successfully. Reviewed core methods, experiment runner, protocol and committed-result verification. Baseline local suite: 26 passed.

## Observed defects and corrections

- Complex NumPy inputs were cast to real, silently discarding imaginary values after a warning. Explicitly reject complex arrays at the input boundary.
- A step of 1e-5 at x=1e20 did not change the represented floating-point input and falsely returned a zero numerical derivative. Reject steps that cannot perturb either side.
- Non-scalar and complex objective values were not explicitly rejected; a NaN initial objective could survive optimization. Require a finite real scalar for all objective evaluations.
- Adam could overflow its squared-gradient state, returning an unchanged solution without reporting failure. Raise on overflow or invalid moment operations, and validate the update and resulting state.

Fourteen regression cases added. Local results: 40/40 tests passed and Python compilation succeeded. All 69 committed per-seed records matched exactly in a local re-execution; a second fresh execution also matched (maximum absolute numerical difference 0.0 for both comparisons). The previously documented cross-platform tolerances remain unchanged.

## Scientific review

LU uses the identity P A = L U; QR uses modified Gram–Schmidt, with separate reconstruction and orthogonality metrics. The conditioning experiment measures relative solution error against a generated known solution. Finite differences measure error against analytic derivatives, not another finite-difference implementation. Optimizer objective gaps use the known quadratic minimum of zero; equal iterations are not equal compute budgets. Scalar reverse-mode differentiation is compared with analytic and finite-difference gradients. SVD and symmetric eigenpairs use NumPy solvers and are reference-backed demonstrations, not independently implemented solvers.

Three seeds and synthetic dense real inputs are descriptive only. The record comparison tolerances for the most sensitive configurations are deliberately wider across platforms; successful comparison does not imply universal accuracy or bitwise reproducibility. The original numerical evidence has not been changed.

## Outstanding limits

Fresh hosted CI is required for the audit revision. Docker, linting, runtime/memory benchmarks, extreme-magnitude matrix robustness, deep/cyclic autodiff graphs and tensor autodiff remain outside tested scope. No peer-review or novelty claim.

Run: `python -m pytest -q` and `python scripts/verify_committed_results.py`.
