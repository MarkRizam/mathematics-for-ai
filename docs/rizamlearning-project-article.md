# AIR-002 — Mathematics behind AI

The project explores what happens when mathematical ideas are implemented with finite-precision arithmetic. It implements pivoted LU, QR, triangular solving, numerical gradients, gradient descent, momentum, Adam and scalar reverse-mode autodiff. Eigen/SVD exercises use NumPy reference routines, not new solvers.

Eight synthetic experiments, three fixed seeds and 23 configurations generated 69 records. The central-difference example illustrates why making a step ever smaller can increase error. On the tested sinusoidal function, mean gradient error was 2.63e-11 at step 1e-5 but 6.75e-8 at step 1e-9. Ill-conditioned linear systems likewise amplified error. These are bounded demonstrations, not universal performance claims.

Study related mathematics lessons in RizamLearning, then use the seven notebooks and reproduce the measurements using commands in the project README. See the research report for all observations and limitations. The portfolio snapshot records implementation status, not peer-reviewed publication.
