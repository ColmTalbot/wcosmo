Backend agnostic astropy-like cosmology
=======================================

An efficient implementation of :code:`astropy`-like cosmology compatible
with :code:`numpy`-like backends, e.g., :code:`jax` and :code:`cupy`.

There are two main features leading to superior efficiency to :code:`astropy`:

- Integrals of :math:`E(z)` and related functions are performed analytically
  with Pade approximations or analytic expressions using hypergeometric functions.
- Support for :code:`jax` and :code:`cupy` backends allow hardware
  acceleration, just-in-time compilation, and automatic differentiation.

The primary limitations are:

- Only flat cosmologies are supported with two components with constant
  equations of state, e.g., :code:`FlatwCDM`.
- Approximations to the various integrals generally agree with :code:`astropy`
  at the <0.1% level.
- The :code:`astropy` units are incompatible with the :code:`cupy` backend.
  Units are supported with the :code:`jax` backend using :code:`unxt`.

Installation and contribution
-----------------------------

:code:`wcosmo` can be installed via :code:`conda-forge`, :code:`pypi` or from
source.

.. code-block:: console

    $ mamba install -c conda-forge wcosmo
    $ pip install wcosmo
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

To import an astropy-like cosmology

.. code-block:: python

    >>> from wcosmo import FlatwCDM
    >>> cosmology = FlatwCDM(H0=70, Om0=0.3, w0=-1)
    >>> cosmology.luminosity_distance(1)

The built-in cosmologies in :code:`astropy` are all available, e.g.,

.. code-block:: python

    >>> from wcosmo import Planck18
    >>> Planck18.luminosity_distance(1)
    <Quantity 6797.43628659 Mpc>
  
they can also be accessed using :code:`wcosmo.available`

.. code-block:: python

    >>> from wcosmo import available
    >>> available["Planck18"].luminosity_distance(1)
    <Quantity 6797.43628659 Mpc>

Explicit usage of :code:`astropy` units can be freely enabled/disabled.
In this case, the values will have the default units for each method.

.. code-block:: python

    >>> from wcosmo import FlatwCDM
    >>> from wcosmo.utils import disable_units, enable_units
    >>> cosmology = FlatwCDM(H0=70, Om0=0.3, w0=-1)

    >>> disable_units()
    >>> cosmology.luminosity_distance(1)
    6607.657732077576

    >>> enable_units()
    >>> cosmology.luminosity_distance(1)
    <Quantity 6607.65773208 Mpc>


Changing backend
^^^^^^^^^^^^^^^^

:code:`wcosmo` mostly relies on implicit backend switching. The backend is
determined automatically based on the input arguments. When an input value is
a :code:`Python` built-in type, the default backend is chosen using the
environment variable :code:`WCOSMO_ARRAY_API`. The default is :code:`numpy`.

GWPopulation
^^^^^^^^^^^^

The original intention for this package was for use with :code:`GWPopulation`.
This code is automatically used in :code:`GWPopulation` when using either
:code:`gwpopulation.experimental.cosmo_models.CosmoModel` and/or
:code:`PowerLawRedshift`
