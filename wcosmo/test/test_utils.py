import numpy as np
import pytest
from scipy.interpolate import pade as spade

from ..taylor import pade as wpade


def test_pade(backend):
    taylor = np.random.uniform(0, 1, 10)
    scoeffs = spade(taylor, 4, 3)
    wcoeffs = wpade(backend.array(taylor), 4, 3)
    assert max(abs(backend.array(scoeffs[0]) - wcoeffs[0])) < 1e-13
    assert max(abs(backend.array(scoeffs[1]) - wcoeffs[1])) < 1e-13
    assert isinstance(wcoeffs[0], backend.ndarray)


def test_pade_raises_n_less_0():
    taylor = np.random.uniform(0, 1, 10)
    with pytest.raises(ValueError):
        wpade(taylor, 4, -3)


def test_pade_n_is_none(backend):
    taylor = np.random.uniform(0, 1, 10)
    scoeffs = spade(taylor, 4)
    wcoeffs = wpade(backend.array(taylor), 4)
    assert max(abs(backend.array(scoeffs[0]) - wcoeffs[0])) < 1e-13
    assert max(abs(backend.array(scoeffs[1]) - wcoeffs[1])) < 1e-13
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
