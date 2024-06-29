import gwpopulation
import pytest

from ..astropy import FlatLambdaCDM, available
from ..utils import disable_units, enable_units


@pytest.fixture(params=["numpy", "jax", "cupy"])
def backend(request):
    pytest.importorskip(request.param)
    gwpopulation.set_backend(request.param)
    return request.param


@pytest.fixture
def npy():
    return gwpopulation.set_backend("numpy")


@pytest.fixture(params=available.keys())
def cosmo(request):
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
