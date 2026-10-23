import numpy as np


def test_strip_units_unwraps_zero_dimensional_arrays():
    from ..utils import strip_units

    value = strip_units(np.array(67.0))
    assert isinstance(value, float)
    assert value == 67.0
    array = np.array([67.0])
    assert strip_units(array) is array
