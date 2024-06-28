import numpy as np
import pytest
from gwpopulation.utils import to_numpy
from scipy.interpolate import pade as spade

from ..taylor import pade as wpade


def test_pade(backend):
    taylor = np.random.uniform(0, 1, 10)
    scoeffs = spade(taylor, 4, 3)
    wcoeffs = wpade(taylor, 4, 3)
    assert max(abs(np.array(scoeffs[0]) - to_numpy(wcoeffs[0]))) < 1e-13
    assert max(abs(np.array(scoeffs[1]) - to_numpy(wcoeffs[1]))) < 1e-13


def test_pade_raises_n_less_0():
    taylor = np.random.uniform(0, 1, 10)
    with pytest.raises(ValueError):
        wpade(taylor, 4, -3)


def test_pade_n_is_none():
    taylor = np.random.uniform(0, 1, 10)
    scoeffs = spade(taylor, 4)
    wcoeffs = wpade(taylor, 4)
    assert max(abs(np.array(scoeffs[0]) - to_numpy(wcoeffs[0]))) < 1e-13
    assert max(abs(np.array(scoeffs[1]) - to_numpy(wcoeffs[1]))) < 1e-13
    with pytest.raises(ValueError):
        wpade(taylor, 20)


def test_pade_raises_m_n_too_large():
    taylor = np.random.uniform(0, 1, 10)
    with pytest.raises(ValueError):
        wpade(taylor, 20, 20)
