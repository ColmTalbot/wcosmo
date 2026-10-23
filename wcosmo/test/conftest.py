import os
from importlib.util import find_spec

import pytest
from astropy.cosmology import available

from ..utils import disable_units, enable_units


def _available_backends():
    """
    The installed array backends, :code:`cupy` also needs a GPU.
    """
    backends = ["numpy"]
    if find_spec("jax") is not None:
        backends.append("jax")
    if find_spec("cupy") is not None:
        import cupy
        from cupy_backends.cuda.api.runtime import CUDARuntimeError

        try:
            cupy.cuda.Device()
            backends.append("cupy")
        except CUDARuntimeError:
            pass
    return backends


AVAILABLE_BACKENDS = _available_backends()

test_points = [
    dict(H0=70.0, Om0=0.0, w0=-1),
    dict(H0=70.0, Om0=0.1, w0=-1),
    dict(H0=70.0, Om0=0.9, w0=-1),
    dict(H0=70.0, Om0=1.0, w0=-1),
    dict(H0=70.0, Om0=0.0, w0=-1.1),
    dict(H0=70.0, Om0=0.1, w0=-1.1),
    dict(H0=70.0, Om0=0.9, w0=-1.1),
    dict(H0=70.0, Om0=1.0, w0=-1.1),
    dict(H0=70.0, Om0=0.0, w0=0.0),
    dict(H0=70.0, Om0=0.1, w0=0.0),
    dict(H0=70.0, Om0=0.9, w0=0.0),
    dict(H0=70.0, Om0=1.0, w0=0.0),
    "Planck15_LAL",
]


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend(request):
    pytest.importorskip(request.param)
    os.environ["WCOSMO_ARRAY_API"] = request.param
    match request.param:
        case "numpy":
            import numpy as xp
        case "jax":
            import jax
            import jax.numpy as xp

            jax.config.update("jax_enable_x64", True)
        case "cupy":
            import cupy as xp
    return xp


@pytest.fixture(params=list(available) + test_points)
def cosmo(request):
    return request.param


@pytest.fixture(params=[True, False])
def units(request):
    if request.param:
        enable_units()
    else:
        disable_units()
    return request.param


@pytest.fixture(params=["pade", "analytic"])
def method(request):
    return request.param
