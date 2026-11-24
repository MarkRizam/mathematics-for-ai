# Mathematics for AI — AIR-002

**Status:** implemented educational numerical experiments, awaiting release audit and hosted CI. Not a novel or peer-reviewed study.

## Purpose
Understand numerical foundations of AI through transparent Python implementations, mathematical references and reproducible experiments. This project complements [Machine Learning From Scratch](https://github.com/MarkRizam/machine-learning-from-scratch) and [RizamLearning](https://github.com/MarkRizam/RizamLearning).

## Implemented
- Partial-pivoted LU and triangular linear-system solving.
- Modified Gram–Schmidt QR.
- Central finite differences.
- Gradient descent, heavy-ball momentum and Adam.
- Scalar reverse-mode automatic differentiation with shared graph nodes.
- NumPy eigenvalue and SVD demonstrations (reference-backed, not from-scratch solvers).
- Eight synthetic experiments, 23 configurations and 69 per-seed records; seven executed teaching notebooks.

## Reproduce
Python 3.10+; tested locally with Python 3.13 and NumPy 2.3.5. CPU only. Install in a virtual environment.

```sh
python -m pip install -e ".[research]"
python -m pytest -q
python scripts/verify_committed_results.py
python -m math_for_ai.experiments
python scripts/notebooks.py
```

For PowerShell set `$env:OPENBLAS_NUM_THREADS='1'` and `$env:OMP_NUM_THREADS='1'` before running. For Bash use `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The experiment runner regenerates results and figures; verify committed records before regenerating.

## Findings
The experiments are descriptive across three seeds. See [report](research/report.md), [protocol](research/protocol.md), [raw records](results/benchmarks.json), and [limitations](research/limitations.md). The reference may be an analytical ideal of zero error, rather than an independent numerical solver. No real-world dataset or participant data is used.

## License and contributions
MIT for original software; dependencies retain their respective licenses. See [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md) and [CITATION.cff](CITATION.cff). Scientific claims must be supported by reproducible evidence.
