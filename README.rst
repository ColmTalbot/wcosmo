Backend agnostic astropy-like cosmology
=======================================

The primary intention for this package is for use with :code:`GWPopulation`
but the main functionality can be used externally.

Basic usage
-----------

To import an astropy-like cosmology (without units)

.. code-block:: python

    from wcosmo import FlatwCDM
    cosmology = FlatwCDM(H0=70, Om0=0.3, w0=-1)

To use this in :code:`GWPopulation`

.. code-block:: python

    import gwpopulation as gwpop
    from wcosmo.models import CosmoModel, PowerLawRedshift

    model = CosmoModel(
        model_functions=[
            gwpop.models.mass.two_component_primary_mass_ratio,
            gwpop.models.spin.iid_spin,
            PowerLawRedshift(cosmo_model="FlatwCDM"),
        ],
        cosmo_model="FlatwCDM",
    )

Changing backend
----------------

The backend can be switched automatically using, e.g.,

.. code-block:: python

    import gwpopulation
    gwpopulation.backend.set_backend("jax")

Manual backend setting can be done as follows:

.. code-block:: python

    import jax.numpy as jnp
    from jax.scipy.linalg.toeplitz import toeplitz

    from wcosmo import wcosmo
    wcosmo.xp = jnp
    wcosmo.toeplitz = toeplitz
