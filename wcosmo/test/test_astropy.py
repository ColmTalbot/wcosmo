import pytest


def test_astropy_cosmology_not_clobbered():
    """See https://github.com/ColmTalbot/wcosmo/issues/15"""
    import astropy.cosmology

    import wcosmo.astropy  # noqa

    assert "wcosmo" not in astropy.cosmology.Planck15.__module__


def test_jits():
    pytest.importorskip("jax")
    import gwpopulation
    from astropy.cosmology import FlatLambdaCDM
    from jax import jit

    from wcosmo.astropy import FlatwCDM

    gwpopulation.set_backend("jax")

    @jit
    def test_func(h0):
        cosmo = FlatwCDM(h0, 0.1, -1)
        return cosmo.luminosity_distance(0.1)

    my_result = test_func(67.0)
    their_result = FlatLambdaCDM(67.0, 0.1).luminosity_distance(0.1)

    assert abs(float(my_result.value) - their_result.value) < 1
    assert my_result.unit == their_result.unit
