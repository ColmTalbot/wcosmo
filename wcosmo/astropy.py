"""
:code:`astropy`-like cosmology objects to work with :code:`wcosmo`.

By default, this provides a drop-in replacement for :code:`astropy.FlatLambdaCDM`
and :code:`astropy.FlatwCDM` cosmology objects along with all of the pre-defined
astropy cosmologies.

By changing the backend and disabling units, these classes can then be used with
:code:`numpy`, :code:`jax`, or :code:`cupy`.
"""

import sys
from dataclasses import dataclass, field

import astropy.cosmology as _acosmo
import numpy as xp

from .utils import autodoc, method_autodoc, strip_units
from .wcosmo import *

USE_UNITS = True

__all__ = [
    "FlatwCDM",
    "FlatLambdaCDM",
    "available",
] + list(_acosmo.available)


class WCosmoMixin:
    """
    Mixin to provide access to the :code:`wcosmo` functionality to :code:`astropy`
    cosmology objects.

    We clobber all units to ensure consistent behavior across backends.

    Notes
    -----

    The following methods are not compatible with non-:code:`numpy` backends:

    - :code:`kpc_comoving_per_arcmin`
    - :code:`kpc_proper_per_arcmin`
    - :code:`nu_relative_density`

    These methods internally coerce the input to :code:`numpy` arrays if the backend
    supports implicit conversion. Additionally, we don't overwrite the various integrands,
    protected methods, and other utility methods, e.g., :code:`clone`.

    We include the following methods that are not present in :code:`astropy`:

    - :code:`dLdH` - derivative of the luminosity distance w.r.t. the Hubble distance
    - :code:`dDLdz` - Jacobian for the conversion of luminosity distance to redshift,
      see :func:`dDLdz`
    - :code:`detector_to_source_frame` - convert masses and luminosity distance from
      the detector frame to the source frame, also returns the jacobian,
      see :func:`detector_to_source_frame`
    - :code:`source_to_detector_frame` - convert masses and redshift from the source
      frame to the detector frame, see :func:`source_to_detector_frame`
    """

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if not USE_UNITS:
            value = strip_units(value)
        return value

    @property
    def _kwargs(self):
        return {"H0": self.H0, "Om0": self.Om0, "w0": self.w0}

    @property
    @method_autodoc(alt=hubble_time)
    def hubble_time(self):
        return hubble_time(self.H0)

    @property
    @method_autodoc(alt=hubble_distance)
    def hubble_distance(self):
        return hubble_distance(self.H0)

    @method_autodoc(alt=luminosity_distance)
    def luminosity_distance(self, z):
        return luminosity_distance(z, **self._kwargs)

    @autodoc
    def dLdH(self, z):
        r"""
        Derivative of the luminosity distance w.r.t. the Hubble distance.

        .. math::

            \frac{{dd_L}}{{dd_H}} = \frac{{d_L}}{{d_H}}

        Parameters
        ----------
        {z}

        Returns
        -------
        array_like:
            The derivative of the luminosity distance w.r.t., the Hubble distance
        """
        return self.luminosity_distance(z) / self.hubble_distance

    @method_autodoc(alt=dDLdz)
    def dDLdz(self, z):
        return dDLdz(z, **self._kwargs)

    @method_autodoc(alt=differential_comoving_volume)
    def differential_comoving_volume(self, z):
        return differential_comoving_volume(z, **self._kwargs)

    @method_autodoc(alt=detector_to_source_frame)
    def detector_to_source_frame(self, m1z, m2z, dL, zmin=1e-4, zmax=100):
        return detector_to_source_frame(
            m1z, m2z, dL, **self._kwargs, zmin=zmin, zmax=zmax
        )

    @method_autodoc(alt=source_to_detector_frame)
    def source_to_detector_frame(self, m1, m2, z):
        return source_to_detector_frame(m1, m2, z, **self._kwargs)

    @method_autodoc(alt=efunc)
    def efunc(self, z):
        return efunc(z, self.Om0, self.w0)

    @method_autodoc(alt=inv_efunc)
    def inv_efunc(self, z):
        return inv_efunc(z, self.Om0, self.w0)

    @method_autodoc(alt=hubble_parameter)
    def H(self, z):
        return hubble_parameter(z, **self._kwargs)

    @method_autodoc(alt=comoving_distance)
    def comoving_distance(self, z):
        return comoving_distance(z, **self._kwargs)

    @method_autodoc(alt=comoving_volume)
    def comoving_volume(self, z):
        return comoving_volume(z, **self._kwargs)

    @method_autodoc(alt=lookback_time)
    def lookback_time(self, z):
        return lookback_time(z, **self._kwargs)

    @method_autodoc(alt=absorption_distance)
    def absorption_distance(self, z):
        return absorption_distance(z, self.Om0, self.w0)

    @autodoc
    def age(self, z, zmax=1e5):
        """
        Compute the age of the universe at redshift z.

        Parameters
        ----------
        {z}
        zmax: float, optional
            The maximum redshift to consider, default is 1e5

        Returns
        -------
        age: array_like
            The age of the universe in Gyr
        """
        return self.lookback_time(zmax) - self.lookback_time(z)

    comoving_transverse_distance = comoving_distance

    @autodoc
    def distmod(self, z):
        """
        Compute the distance modulus at redshift z.

        Parameters
        ----------
        {z}

        Returns
        -------
        distmod: array_like
            The distance modulus (units: mag)
        """
        distance = strip_units(self.luminosity_distance(z))
        return 5 * xp.log10(xp.abs(distance)) + 25

    @autodoc
    def de_density_scale(self, z):
        """
        Dark energy density at redshift z.

        Parameters
        ----------
        {z}

        Returns
        -------
        rho_de: array_like
            The dark energy density at redshift z
        """
        return (z + 1) ** (3 * (1 + self.w0))


@dataclass(frozen=True)
class FlatwCDM(WCosmoMixin, _acosmo.FlatwCDM):
    pass


@dataclass(frozen=True)
class FlatLambdaCDM(WCosmoMixin, _acosmo.FlatLambdaCDM):
    w0: float = field(init=False, default=-1)


def __getattr__(name):
    alt = _acosmo.__getattr__(name)
    cosmo = FlatLambdaCDM(**alt.parameters)
    setattr(sys.modules[__name__], name, cosmo)
    return cosmo


class _Available:

    def keys(self):
        return ("FlatLambdaCDM", "FlatwCDM") + _acosmo.available

    def __getitem__(self, key):
        return getattr(sys.modules[__name__], key)

    def __repr__(self):
        return repr(self.keys())


available = _Available()
