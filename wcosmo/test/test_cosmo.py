
import numpy as np
import pytest
from astropy import cosmology
from gwpopulation.utils import to_numpy

from .. import wcosmo

funcs = [
    "luminosity_distance",
    "comoving_volume",
    "comoving_distance",
    "efunc",
    "inv_efunc",
    "differential_comoving_volume",
]


@pytest.mark.parametrize("func", funcs)
def test_redshift_function(cosmo, func, backend):
    ours = wcosmo.available[cosmo]
    theirs = getattr(cosmology, cosmo)

    redshifts = np.linspace(1e-3, 10, 1000)

    ours = to_numpy(getattr(ours, func)(redshifts))
    theirs = getattr(theirs, func)(redshifts)
    if hasattr(theirs, "value"):
        theirs = theirs.value
    assert max(abs(ours - theirs) / theirs) < 1e-2


@pytest.mark.parametrize("func", funcs[:3])
def test_z_at_value(cosmo, func, backend):
    from gwpopulation.utils import xp
    ours = getattr(wcosmo.available[cosmo], func)
    theirs = getattr(getattr(cosmology, cosmo), func)

    redshifts = np.linspace(1e-3, 10, 10)
    vals = theirs(redshifts)

    if hasattr(vals, "value"):
        ovals = xp.asarray(vals.value)
    else:
        ovals = xp.asarray(vals)

    ours = wcosmo.z_at_value(ours, ovals)
    theirs = cosmology.z_at_value(theirs, vals).value
    assert max(abs(ours - theirs) / theirs) < 1e-2
