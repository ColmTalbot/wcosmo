import numpy as np
from gwpopulation.utils import to_numpy
from scipy.interpolate import pade as spade

from ..utils import pade as wpade


def test_pade(backend):
    taylor = np.random.uniform(0, 1, 10)
    scoeffs = spade(taylor, 4, 3)
    wcoeffs = wpade(taylor, 4, 3)
    assert max(abs(np.array(scoeffs[0]) - to_numpy(wcoeffs[0]))) < 1e-13
    assert max(abs(np.array(scoeffs[1]) - to_numpy(wcoeffs[1]))) < 1e-13
