from math import factorial

import numpy as np
import pytest

from ..taylor import pade as wpade


def _exp_pade(m, n):
    """
    The closed form Pade approximation to :math:`e^x` with orders :code:`m`
    and :code:`n` of the denominator and numerator, highest order first.
    """
    norm = factorial(m + n)
    p = [
        factorial(m + n - j) * factorial(n) / (norm * factorial(j) * factorial(n - j))
        for j in range(n + 1)
    ]
    q = [
        (-1) ** j
        * factorial(m + n - j)
        * factorial(m)
        / (norm * factorial(j) * factorial(m - j))
        for j in range(m + 1)
    ]
    return np.array(p[::-1]), np.array(q[::-1])


def _exp_taylor(n_terms):
    return np.array([1 / factorial(k) for k in range(n_terms)])


def test_pade(backend):
    taylor = _exp_taylor(10)
    expected = _exp_pade(4, 3)
    wcoeffs = wpade(backend.array(taylor), 4, 3)
    assert max(abs(backend.array(expected[0]) - wcoeffs[0])) < 1e-13
    assert max(abs(backend.array(expected[1]) - wcoeffs[1])) < 1e-13
    assert isinstance(wcoeffs[0], backend.ndarray)


def test_pade_raises_n_less_0():
    taylor = np.random.uniform(0, 1, 10)
    with pytest.raises(ValueError):
        wpade(taylor, 4, -3)


def test_pade_n_is_none(backend):
    taylor = _exp_taylor(10)
    expected = _exp_pade(4, 5)
    wcoeffs = wpade(backend.array(taylor), 4)
    assert max(abs(backend.array(expected[0]) - wcoeffs[0])) < 1e-13
    assert max(abs(backend.array(expected[1]) - wcoeffs[1])) < 1e-13
    with pytest.raises(ValueError):
        wpade(taylor, 20)


def test_pade_raises_m_n_too_large():
    taylor = np.random.uniform(0, 1, 10)
    with pytest.raises(ValueError):
        wpade(taylor, 20, 20)


def test_strip_units_unwraps_zero_dimensional_arrays():
    from ..utils import strip_units

    value = strip_units(np.array(67.0))
    assert isinstance(value, float)
    assert value == 67.0
    array = np.array([67.0])
    assert strip_units(array) is array
