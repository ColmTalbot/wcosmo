import os
import subprocess
import sys

import numpy as np


def test_strip_units_unwraps_zero_dimensional_arrays():
    from ..utils import strip_units

    value = strip_units(np.array(67.0))
    assert isinstance(value, float)
    assert value == 67.0
    array = np.array([67.0])
    assert strip_units(array) is array


def test_import_does_not_import_jax():
    """
    The JAX implementations are only imported when they are needed, unless
    :code:`WCOSMO_ARRAY_API=jax` makes JAX the default.
    """
    env = {key: val for key, val in os.environ.items() if key != "WCOSMO_ARRAY_API"}
    code = "import sys, wcosmo; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
