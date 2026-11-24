"""Regression cases identified during the AIR-002 V1.1 audit."""
import numpy as np
import pytest
from math_for_ai import gradient, lu, optimize, qr, solve


@pytest.mark.parametrize('fn', [
    lambda: lu(np.array([[1+2j]])),
    lambda: qr(np.array([[1+2j]])),
    lambda: solve([[1]], np.array([1+2j])),
    lambda: gradient(sum, np.array([1+2j])),
    lambda: optimize(sum, lambda x: x, np.array([1+2j])),
    lambda: optimize(sum, lambda x: np.array([1+2j]), [1.], steps=1),
])
def test_complex_rejected(fn):
    with pytest.raises(ValueError):
        fn()


@pytest.mark.parametrize('fn', [
    lambda: gradient(lambda x: x[0], [1e20], 1e-5),
    lambda: gradient(lambda x: np.array([x[0]]), [1.]),
    lambda: gradient(lambda x: 1+2j, [1.]),
    lambda: optimize(lambda x: np.array([x[0]]), lambda x: x, [1.], steps=1),
])
def test_invalid_scalar_or_step(fn):
    with pytest.raises(ValueError):
        fn()


@pytest.mark.parametrize('fn', [
    lambda: gradient(lambda x: np.nan, [1.]),
    lambda: gradient(lambda x: np.inf, [1.]),
    lambda: optimize(lambda x: np.nan if x[0] == 1 else x@x, lambda x: x, [1.], steps=1),
    lambda: optimize(lambda x: 0., lambda x: np.array([1e200]), [1.], method='adam', steps=1),
])
def test_nonfinite_rejected(fn):
    with pytest.raises(FloatingPointError):
        fn()
