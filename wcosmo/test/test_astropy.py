def test_astropy_cosmology_not_clobbered():
    """See https://github.com/ColmTalbot/wcosmo/issues/15"""
    import astropy.cosmology

    import wcosmo.astropy

    assert "wcosmo" not in astropy.cosmology.Planck15.__module__
