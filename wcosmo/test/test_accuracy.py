r"""
Test distances against :code:`astropy` in double and single precision over
a wide range of cosmological parameters.

The expansion used for the distance integrals converges more slowly for
small :math:`\Omega_{m, 0}`, so we include matter densities down to 0.05.
"""

from functools import cache, partial

import numpy as np
import pytest
from astropy import cosmology

from .. import wcosmo
from ..astropy import FlatwCDM
from ..utils import disable_units, enable_units

OM0S = [0.05, 0.1, 0.2, 0.3075, 0.5, 0.8, 1.0]
W0S = [-2.0, -1.45, -1.0, -0.55, -0.3]
REDSHIFTS = np.logspace(-5, 1, 500)
FUNCS = [
    "comoving_distance",
    "luminosity_distance",
    "differential_comoving_volume",
    "lookback_time",
]
# astropy agrees with high-precision quadrature to ~1e-11
TOLERANCES = dict(float64=1e-10, float32=5e-5)


@pytest.fixture(params=["numpy", "jax", "jax-float32"])
def precision(request):
    """
    Yield a function to convert inputs to the tested array type and precision
    and the name of the precision.
    """
    disable_units()
    if request.param == "numpy":
        yield np.asarray, "float64"
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        x64 = jax.config.jax_enable_x64
        single = request.param == "jax-float32"
        jax.config.update("jax_enable_x64", not single)
        dtype = jnp.float32 if single else jnp.float64
        yield partial(jnp.asarray, dtype=dtype), dtype.__name__
        jax.config.update("jax_enable_x64", x64)
    enable_units()


@cache
def _astropy_cosmology(Om0, w0):
    return cosmology.FlatwCDM(H0=70.0, Om0=Om0, w0=w0, Tcmb0=0)


@cache
def _astropy_value(func, Om0, w0):
    return getattr(_astropy_cosmology(Om0, w0), func)(REDSHIFTS).value


def _max_relative_error(test, reference):
    return np.max(np.abs(np.asarray(test, dtype=np.float64) / reference - 1))


@pytest.mark.parametrize("func", FUNCS)
@pytest.mark.parametrize("Om0", OM0S)
@pytest.mark.parametrize("w0", W0S)
def test_distances_match_astropy(precision, func, Om0, w0):
    convert, dtype = precision
    ours = getattr(FlatwCDM(H0=70.0, Om0=Om0, w0=w0), func)(convert(REDSHIFTS))
    assert ours.dtype == dtype
    theirs = _astropy_value(func, Om0, w0)
    assert _max_relative_error(ours, theirs) < TOLERANCES[dtype]


@pytest.mark.parametrize("Om0", OM0S)
@pytest.mark.parametrize("w0", W0S)
def test_z_at_value_inverts_astropy_distances(precision, Om0, w0):
    """
    :func:`wcosmo.z_at_value` interpolates on a grid of 1000 redshifts,
    which limits the accuracy to ~1e-5.
    """
    convert, dtype = precision
    redshifts = np.logspace(-3, 1, 30)
    distances = _astropy_cosmology(Om0, w0).luminosity_distance(redshifts).value
    ours = wcosmo.z_at_value(
        wcosmo.luminosity_distance, convert(distances), H0=70.0, Om0=Om0, w0=w0
    )
    assert ours.dtype == dtype
    assert _max_relative_error(ours, redshifts) < 2e-5


def _integral_derivatives(z, Om0, w0, zpower):
    r"""
    Derivatives of :math:`\int_0^z dz' (1 + z')^k / E(z')` with respect to
    :math:`\Omega_{m, 0}` and :math:`w_0` by numerical quadrature.
    """
    from scipy.integrate import quad

    def integrand(zz, parameter):
        zp1 = 1 + zz
        de_scale = zp1 ** (3 * (1 + w0))
        efunc_cubed = (Om0 * zp1**3 + (1 - Om0) * de_scale) ** 1.5
        if parameter == "Om0":
            derivative = zp1**3 - de_scale
        else:
            derivative = 3 * (1 - Om0) * np.log(zp1) * de_scale
        return -(zp1**zpower) * derivative / efunc_cubed / 2

    return [
        np.array(
            [
                quad(integrand, 0, zz, args=(name,), epsabs=0, epsrel=1e-13)[0]
                for zz in z
            ]
        )
        for name in ["Om0", "w0"]
    ]


@pytest.mark.parametrize("Om0, w0", [(0.05, -1.45), (0.1, -0.55), (0.3075, -1.0)])
def test_gradients_match_quadrature(precision, Om0, w0):
    r"""
    Check gradients of the luminosity distance and lookback time with respect
    to redshift against :func:`wcosmo.dDLdz` and the integrand, and with
    respect to :math:`\Omega_{m, 0}` and :math:`w_0` against numerical
    quadrature.

    At low redshift the derivatives with respect to the parameters are
    suppressed by a factor of :math:`z` relative to the function, so we test
    the error in the derivative of the logarithm of the function.
    """
    convert, dtype = precision
    if dtype == "float64" and convert is np.asarray:
        pytest.skip("Gradients require JAX")
    import jax

    eps_z, eps_parameters = dict(float64=(1e-12, 1e-9), float32=(1e-5, 1e-3))[dtype]

    redshifts = np.logspace(-4, 1, 30)
    d_H = wcosmo.hubble_distance(70.0)
    t_H = wcosmo.hubble_time(70.0)
    distance_derivatives = _integral_derivatives(redshifts, Om0, w0, zpower=0)
    time_derivatives = _integral_derivatives(redshifts, Om0, w0, zpower=-1)

    for func, references in [
        (
            wcosmo.luminosity_distance,
            [wcosmo.dDLdz(redshifts, 70.0, Om0, w0)]
            + [(1 + redshifts) * d_H * value for value in distance_derivatives],
        ),
        (
            wcosmo.lookback_time,
            [t_H / (1 + redshifts) / wcosmo.efunc(redshifts, Om0, w0)]
            + [t_H * value for value in time_derivatives],
        ),
    ]:
        values = func(redshifts, 70.0, Om0, w0)
        grads = jax.vmap(
            jax.grad(lambda z, Om0, w0: func(z, 70.0, Om0, w0), argnums=(0, 1, 2)),
            in_axes=(0, None, None),
        )(convert(redshifts), convert(Om0), convert(w0))
        for grad in grads:
            assert grad.dtype == dtype
        grads = [np.asarray(grad, dtype=np.float64) for grad in grads]
        assert _max_relative_error(grads[0], references[0]) < eps_z
        for grad, reference in zip(grads[1:], references[1:]):
            assert np.max(np.abs(grad - reference) / values) < eps_parameters
