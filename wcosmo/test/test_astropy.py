import pytest


def test_astropy_cosmology_not_clobbered():
    """See https://github.com/ColmTalbot/wcosmo/issues/15"""
    import astropy.cosmology

    import wcosmo.astropy

    assert "wcosmo" not in astropy.cosmology.Planck15.__module__


def test_jits():
    pytest.importorskip("jax")
    import gwpopulation
    from astropy.cosmology import FlatLambdaCDM
    from jax import jit

    import wcosmo
    from wcosmo.astropy import FlatwCDM
    from wcosmo.utils import disable_units

    @jit
    def test_func(h0):
        cosmo = FlatwCDM(h0, 0.1, -1)
        return cosmo.luminosity_distance(0.1)

    gwpopulation.set_backend("jax")
    disable_units()

    assert (
        abs(
            float(test_func(67.0))
            - FlatLambdaCDM(67.0, 0.1).luminosity_distance(0.1).value
        )
        < 1
    )
