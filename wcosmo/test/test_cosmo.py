import os
import warnings

import numpy as np
import pytest
from astropy import cosmology
from scipy.integrate import IntegrationWarning

from .. import astropy, wcosmo
from ..utils import array_namespace, disable_units, enable_units, strip_units

funcs = [
    "luminosity_distance",
    "comoving_volume",
    "comoving_distance",
    "efunc",
    "inv_efunc",
    "differential_comoving_volume",
    "absorption_distance",
    "lookback_time",
    "de_density_scale",
    "distmod",
    "H",
    "age",
]

EPS = 1e-2


def to_numpy(arr):
    xp = array_namespace(arr)
    if "cupy" in xp.__name__:
        return arr.get()
    else:
        return np.asarray(arr)


def get_equivalent_cosmologies(cosmo):
    if isinstance(cosmo, str):
        ours = astropy.available[cosmo]
        theirs = getattr(cosmology, cosmo)
    else:
        ours = astropy.FlatwCDM(**cosmo)
        theirs = cosmology.FlatwCDM(**cosmo)
    return ours, theirs


@pytest.mark.parametrize("func", funcs)
def test_redshift_function(cosmo, func, backend, units, method):
    if (units or method == "analytic") and ("cupy" in backend.__name__):
        pytest.skip()
    xp = backend

    instance, alt = get_equivalent_cosmologies(cosmo)

    if func in ["age", "lookback_time"] and instance.Om0 == 0.0 and instance.w0 == -1:
        pytest.skip("Age is infinite for de Sitter cosmologies")

    redshifts = np.linspace(1e-3, 10, 1000)
    xredshifts = xp.linspace(1e-3, 10, 1000)

    object.__setattr__(instance, "method", method)

    ours = getattr(instance, func)(xredshifts)
    with warnings.catch_warnings():
        # filter warnings from astropy about integration not converging for age
        warnings.simplefilter("ignore", category=IntegrationWarning)
        theirs = getattr(alt, func)(redshifts)

    object.__setattr__(instance, "method", "pade")

    if not units:
        theirs = strip_units(theirs)
    elif hasattr(theirs, "unit") and hasattr(ours, "unit"):
        assert ours.unit == theirs.unit
        ours = ours.value
        theirs = theirs.value
    elif hasattr(theirs, "unit"):
        theirs = theirs.value
    if func == "absorption_distance":
        # The absorption distance calculation is not super
        # accurate, I think this is either an issue with
        # astropy or there's a conceptual issue in the wcosmo
        # implementation.
        eps = 1e-1
    else:
        eps = EPS
    assert isinstance(ours, xp.ndarray)
    assert max(abs(to_numpy(ours) - theirs) / theirs) < eps


@pytest.mark.parametrize("func", funcs[:3])
def test_z_at_value(cosmo, func, backend, method):
    if "cupy" in backend.__name__ and method == "analytic":
        pytest.skip()
    disable_units()
    xp = backend

    instance, alt = get_equivalent_cosmologies(cosmo)

    object.__setattr__(instance, "method", method)

    ours = getattr(instance, func)
    theirs = getattr(alt, func)

    redshifts = np.linspace(1e-3, 10, 10)
    vals = theirs(redshifts)

    if hasattr(vals, "value"):
        ovals = xp.asarray(vals.value)
    else:
        ovals = xp.asarray(vals)

    ours = wcosmo.z_at_value(ours, ovals)
    theirs = cosmology.z_at_value(theirs, vals).value

    object.__setattr__(instance, "method", "pade")

    assert isinstance(ours, xp.ndarray)
    assert max(abs(to_numpy(ours) - theirs) / theirs) < EPS


@pytest.mark.parametrize("func", ["hubble_time", "hubble_distance"])
def test_properties(cosmo, func):
    instance, alt = get_equivalent_cosmologies(cosmo)
    ours = getattr(instance, func)
    theirs = getattr(alt, func)

    ours = ours
    theirs = theirs

    assert abs(strip_units(ours) - strip_units(theirs)) < 1e-8


def test_detector_to_source_and_source_to_detector_are_inverse(cosmo, backend):
    xp = backend

    ours, _ = get_equivalent_cosmologies(cosmo)

    source_mass_1, source_mass_2 = xp.asarray(np.random.uniform(20, 30, (2, 1000)))
    redshifts = xp.asarray(np.random.uniform(1e-4, 1, 1000))

    detector_frame = ours.source_to_detector_frame(
        source_mass_1, source_mass_2, redshifts
    )
    final_mass_1, final_mass_2, final_redshifts = ours.detector_to_source_frame(
        *detector_frame
    )

    assert max(abs(final_mass_1 - source_mass_1)) < EPS
    assert max(abs(final_mass_2 - source_mass_2)) < EPS
    assert max(abs(final_redshifts - redshifts)) < EPS
    assert isinstance(final_mass_1, xp.ndarray)


def test_dDLdz_is_the_gradient(cosmo):
    """
    Test that the gradient of the luminosity distance via the analytic expression
    is the same as the value obtained with autodiff.
    """
    jax = pytest.importorskip("jax")
    os.environ["WCOSMO_ARRAY_API"] = "jax"
    enable_units()

    ours, _ = get_equivalent_cosmologies(cosmo)

    if ours.Om0 == 0.0:
        pytest.skip("Gradient doesn't currently work for Om0=0")

    auto_gradient = jax.grad(lambda z: ours.luminosity_distance(z).value)

    points = jax.numpy.linspace(0.1, 10, 1000)

    autodiffed = jax.vmap(auto_gradient)(points)
    analytic = ours.dDLdz(points).value

    assert max(abs(autodiffed - analytic) / analytic) < EPS
