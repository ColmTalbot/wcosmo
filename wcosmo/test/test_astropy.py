import pytest


def test_astropy_cosmology_not_clobbered():
    """See https://github.com/ColmTalbot/wcosmo/issues/15"""
    import astropy.cosmology

    import wcosmo.astropy

    assert "wcosmo" not in astropy.cosmology.Planck15.__module__


def test_jits():
    pytest.importorskip("jax")
    from jax import jit
    from jax import numpy as xp
    from jax.scipy.linalg import toeplitz

    import wcosmo
    from wcosmo.astropy import FlatwCDM
    from wcosmo.utils import disable_units

    @jit
    def test_func(h0):
        cosmo = FlatwCDM(h0, 0.1, -1)
        return cosmo.luminosity_distance(0.1)

    wcosmo.xp = xp
    wcosmo.utils.xp = xp
    wcosmo.taylor.xp = xp
    wcosmo.taylor.toeplitz = toeplitz
    disable_units()

    assert test_func(67.0) == 489.96887
