import gwpopulation
import pytest
from astropy import units as u

from ..astropy import FlatLambdaCDM, available
from ..utils import disable_units, enable_units

h_unit = u.km / u.s / u.Mpc

# FIXME: make the commented test cases work
test_points = [
    dict(H0=70 * h_unit, Om0=0.0, w0=-1),
    dict(H0=70 * h_unit, Om0=0.1, w0=-1),
    dict(H0=70 * h_unit, Om0=0.9, w0=-1),
    dict(H0=70 * h_unit, Om0=1.0, w0=-1),
    dict(H0=70 * h_unit, Om0=0.0, w0=-1.1),
    dict(H0=70 * h_unit, Om0=0.1, w0=-1.1),
    dict(H0=70 * h_unit, Om0=0.9, w0=-1.1),
    dict(H0=70 * h_unit, Om0=1.0, w0=-1.1),
    dict(H0=70 * h_unit, Om0=0.0, w0=0.0),
    dict(H0=70 * h_unit, Om0=0.1, w0=0.0),
    dict(H0=70 * h_unit, Om0=0.9, w0=0.0),
    dict(H0=70 * h_unit, Om0=1.0, w0=0.0),
]


@pytest.fixture(params=["numpy", "jax", "cupy"])
def backend(request):
    pytest.importorskip(request.param)
    gwpopulation.set_backend(request.param)
    return request.param


@pytest.fixture
def npy():
    return gwpopulation.set_backend("numpy")


@pytest.fixture(params=list(available.keys()) + test_points)
def cosmo(request):
    if isinstance(request.param, str):
        ours = available[request.param]
        if not isinstance(ours, FlatLambdaCDM):
            pytest.skip()
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
