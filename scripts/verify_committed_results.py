"""Compare fresh outputs with committed experiment records."""
import json
from pathlib import Path
import numpy as np
from math_for_ai.experiments import run


def main():
    baseline=json.loads((Path(__file__).resolve().parents[1]/'results/benchmarks.json').read_text())['records']
    fresh=run()
    assert len(baseline)==len(fresh)==69
    for a,b in zip(baseline,fresh):
        assert a.keys()==b.keys()
        for key in a:
            if isinstance(a[key],float):
                # Ill-conditioned solve errors can vary with BLAS and architecture.
                atol=1e-10
                if a['experiment']=='EXP-003' and a['configuration']=='condition=1e+08':
                    atol=1e-8
                if a['experiment']=='EXP-004' and a['configuration'] in ('h=1e-07','h=1e-09'):
                    # Finite-difference rounding is amplified by 1/h.
                    atol=3e-9 if a['configuration']=='h=1e-07' else 3e-7
                np.testing.assert_allclose(a[key],b[key],rtol=0,atol=atol)
            else: assert a[key]==b[key]
    print('69 committed records reproduced within declared cross-platform tolerances')

if __name__=='__main__': main()
