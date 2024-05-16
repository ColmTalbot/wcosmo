Backend agnostic astropy-like cosmology
=======================================

The primary intention for this package is for use with :code:`GWPopulation`
but the main functionality can be used externally.

Installation and contribution
-----------------------------

Currently installation is only available from source

.. code-block:: console

    $ pip install git+https://github.com/ColmTalbot/wcosmo.git

for development you should follow a standard fork-and-pull workflow.

- First create a new fork at :code:`github.com/UserName/wcosmo`.
- Clone your fork

  .. code-block:: console

    $ git clone git@github.com:UserName/wcosmo.git

  or use a GitHub codespace.
- Install the local version with

  .. code-block:: console

    $ python -m pip install .

- Make any desired edits and push to your fork.
- Open a pull request into :code:`git@github.com:ColmTalbot/wcosmo.git`.

Basic usage
-----------

To import an astropy-like cosmology (without units)

.. code-block:: python

    from wcosmo import FlatwCDM
    cosmology = FlatwCDM(H0=70, Om0=0.3, w0=-1)

This code is automatically used in :code:`GWPopulation` when using either
:code:`gwpopulation.experimental.cosmo_models.CosmoModel` and/or
:code:`PowerLawRedshift`

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

    from wcosmo import wcosmo, utils
    wcosmo.xp = jnp
    utils.xp = jnp
    utils.toeplitz = toeplitz
