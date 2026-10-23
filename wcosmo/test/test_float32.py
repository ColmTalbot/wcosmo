"""
Test that single precision calculations with :code:`JAX` (with x64 disabled)
agree with double precision calculations with :code:`numpy`.
"""

import numpy as np
import pytest

from .. import wcosmo
from ..utils import disable_units, enable_units

EPS = 2e-5

# include points that were previously close to spurious poles in the
# single precision Pade approximant
REDSHIFTS = np.sort(
    np.concatenate([np.logspace(-5, 1, 2000), [0.0036, 0.070, 0.178]])
).astype(np.float32)


@pytest.fixture
def jnp():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    disable_units()
    yield jnp
    jax.config.update("jax_enable_x64", x64)
    enable_units()


def _max_relative_error(test, reference):
    return np.max(np.abs(np.asarray(test, dtype=np.float64) / reference - 1))


@pytest.mark.parametrize(
    "func", ["comoving_distance", "differential_comoving_volume", "luminosity_distance"]
)
@pytest.mark.parametrize("Om0", [0.25, 0.3075, 0.35, 0.5])
@pytest.mark.parametrize("w0", [-1.0, -0.8, -1.3])
def test_float32_matches_float64(jnp, func, Om0, w0):
    func = getattr(wcosmo, func)
    reference = func(REDSHIFTS.astype(np.float64), 67.7, Om0, w0)
    test = func(jnp.asarray(REDSHIFTS), 67.7, Om0, w0)
    assert test.dtype == jnp.float32
    assert _max_relative_error(test, reference) < EPS


@pytest.mark.parametrize("Om0", [0.25, 0.3075, 0.35])
@pytest.mark.parametrize("w0", [-1.0, -0.8])
def test_float32_z_at_value(jnp, Om0, w0):
    redshifts = np.logspace(-3.9, 1.5, 500)
    distances = wcosmo.luminosity_distance(redshifts, 67.7, Om0, w0)
    kwargs = dict(H0=67.7, Om0=Om0, w0=w0)
    reference = wcosmo.z_at_value(wcosmo.luminosity_distance, distances, **kwargs)
    test = wcosmo.z_at_value(
        wcosmo.luminosity_distance, jnp.asarray(distances, dtype=jnp.float32), **kwargs
    )
    assert test.dtype == jnp.float32
    assert _max_relative_error(test, reference) < EPS


def test_float32_jit_with_traced_parameters(jnp):
    import jax

    func = jax.jit(wcosmo.differential_comoving_volume)
    for Om0, w0 in [(0.3075, -1.0), (0.25, -0.8), (0.35, -1.3)]:
        reference = wcosmo.differential_comoving_volume(
            REDSHIFTS.astype(np.float64), 67.7, Om0, w0
        )
        test = func(
            jnp.asarray(REDSHIFTS), jnp.asarray(67.7), jnp.asarray(Om0), jnp.asarray(w0)
        )
        assert test.dtype == jnp.float32
        assert _max_relative_error(test, reference) < EPS


def test_float32_gradients(jnp):
    import jax

    redshifts = np.logspace(-1, 1, 100).astype(np.float32)
    Om0, w0 = 0.3075, -1.0

    def func(z, Om0, w0):
        return wcosmo.luminosity_distance(z, 67.7, Om0, w0)

    grads = jax.vmap(jax.grad(func, argnums=(0, 1, 2)), in_axes=(0, None, None))(
        jnp.asarray(redshifts), jnp.asarray(Om0), jnp.asarray(w0)
    )
    for grad in grads:
        assert grad.dtype == jnp.float32

    zs = redshifts.astype(np.float64)
    step = 1e-6
    references = [
        wcosmo.dDLdz(zs, 67.7, Om0, w0),
        (func(zs, Om0 + step, w0) - func(zs, Om0 - step, w0)) / 2 / step,
        (func(zs, Om0, w0 + step) - func(zs, Om0, w0 - step)) / 2 / step,
    ]
    for grad, reference in zip(grads, references):
        assert _max_relative_error(grad, reference) < 1e-4
