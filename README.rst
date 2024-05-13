Backend agnostic astropy-like cosmology.

The primary intention for this package is for use with :code:`GWPopulation`
but the main functionality can be used externally.

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
