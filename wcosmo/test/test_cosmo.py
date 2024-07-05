import numpy as np
import pytest
from astropy import cosmology
from gwpopulation.backend import set_backend
from gwpopulation.utils import to_numpy

from .. import astropy, wcosmo
from ..utils import disable_units, strip_units

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
    "critical_density",
    "distmod",
    "H",
    "age",
]

EPS = 1e-3


@pytest.mark.parametrize("func", funcs)
def test_redshift_function(cosmo, func, backend, units):
    ours = astropy.available[cosmo]
    theirs = getattr(cosmology, cosmo)

    redshifts = np.linspace(1e-3, 10, 1000)

    ours = to_numpy(getattr(ours, func)(redshifts))
    theirs = getattr(theirs, func)(redshifts)
    if not units:
        theirs = strip_units(theirs)
    elif hasattr(theirs, "unit"):
        assert ours.unit == theirs.unit
    assert max(abs(ours - theirs) / theirs) < EPS


@pytest.mark.parametrize("func", funcs[:3])
def test_z_at_value(cosmo, func, backend):
    disable_units()
    from gwpopulation.utils import xp

    ours = getattr(astropy.available[cosmo], func)
    theirs = getattr(getattr(cosmology, cosmo), func)

    redshifts = np.linspace(1e-3, 10, 10)
    vals = theirs(redshifts)

    if hasattr(vals, "value"):
        ovals = xp.asarray(vals.value)
    else:
        ovals = xp.asarray(vals)

    ours = wcosmo.z_at_value(ours, ovals)
    theirs = cosmology.z_at_value(theirs, vals).value
    assert max(abs(ours - theirs) / theirs) < EPS


@pytest.mark.parametrize("func", ["hubble_time", "hubble_distance"])
def test_properties(cosmo, func):
    ours = getattr(astropy.available[cosmo], func)
    theirs = getattr(getattr(cosmology, cosmo), func)

    ours = ours
    theirs = theirs

    if hasattr(ours, "unit"):
        ours = ours.value
    assert strip_units(ours) == strip_units(theirs)


def test_detector_to_source_and_source_to_detector_are_inverse(cosmo, backend):
    from gwpopulation.utils import xp

    ours = astropy.available[cosmo]

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


def test_dDLdz_is_the_gradient(cosmo):
    """
    Test that the gradient of the luminosity distance via the analytic expression
    is the same as the value obtained with autodiff.
    """
    jax = pytest.importorskip("jax")
    set_backend("jax")

    ours = astropy.available[cosmo]

    auto_gradient = jax.grad(ours.luminosity_distance)

    points = jax.numpy.linspace(0.1, 10, 1000)

    assert max(abs(jax.vmap(auto_gradient)(points) - ours.dDLdz(points))) < EPS
