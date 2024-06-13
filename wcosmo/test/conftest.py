import gwpopulation
import pytest

from ..wcosmo import FlatLambdaCDM, available


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
