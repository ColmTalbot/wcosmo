def test_astropy_cosmology_not_clobbered():
    import wcosmo.astropy
    import astropy.cosmology
    assert "wcosmo" not in astropy.cosmology.Planck15.__module__