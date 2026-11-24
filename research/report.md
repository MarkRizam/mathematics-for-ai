# AIR-002 — Numerical mathematics for AI

## Abstract
Eight controlled synthetic experiments examine factorization residuals, conditioning, numerical derivatives, optimization, reverse-mode differentiation, SVD compression and symmetric eigenpair residuals. The local package passed 26 tests and seven notebooks executed. Two runs matched within absolute tolerance 1e-10 across 69 records (23 configurations, three seeds). This is an educational reproduction.

## Method
Seeds 11, 22, 33; independent generated small dense real matrices and scalar/vector functions. No external datasets. Compare pivoted LU against its mathematical identity, QR against NumPy QR, linear solves against known solutions and NumPy solve, gradients against analytic derivatives, optimizer objective gaps against a known optimum, autodiff against analytic and finite-difference gradients, NumPy SVD reconstruction against singular-value tails, and NumPy symmetric eigenpairs against their defining identity. Refer to protocol and exact executable configuration.

## Observations
- LU mean relative residual 1.01e-16; QR reconstruction 7.13e-17. These values indicate close reconstruction on small random matrices only.
- Relative solution error increased from 1.82e-16 at condition 1 to 9.72e-10 at condition 1e8. Ill-conditioning amplifies numerical error.
- Central finite-difference gradient error fell from 2.40e-3 at h=0.1 to 2.63e-11 at h=1e-5, then increased to 6.75e-8 at h=1e-9. This illustrates truncation versus roundoff for this function and scale.
- At rate 0.01 after 200 steps, the mean quadratic objective gaps were 0.00448 (GD), 9.66e-12 (momentum) and 0.00132 (Adam). This is not a general optimizer ranking; only one quadratic and two rates were tested.
- Reverse-mode mean gradient error 7.40e-17 versus 2.96e-11 for central differences in the tested graph.
- SVD rank-one mean reconstruction error 0.6593 versus 8.75e-16 at rank four. Eigenpair relative residual mean 2.99e-16. Both use NumPy solvers, not novel implementations.

All configurations and sample SDs are in results/benchmarks.json and the eight figures. The SD is descriptive, not a confidence interval. References in the plots can be zero by mathematical definition. Errors with different metrics cannot be compared across experiments.

## Reproducibility and limitations
Local same-environment reproducibility: absolute tolerance 1e-10, relative tolerance zero. The committed-result cross-platform test allows 1e-8 absolute for the condition-1e8 solve to account for BLAS variation, and 3e-9 / 3e-7 for finite differences at h=1e-7 / 1e-9 to account for rounding amplification; this is a deliberate tolerance choice, not a claim of cross-platform execution. See limitations. No claims of general superiority or clinical benefit. No runtime/memory benchmarks.

## References
- NumPy linear algebra documentation: https://numpy.org/doc/stable/reference/routines.linalg.html
- Mathematics for Machine Learning (Deisenroth, Faisal, Ong): https://mml-book.github.io/
- Kingma and Ba, Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
