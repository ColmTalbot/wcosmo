"""
:code:`astropy`-like cosmology objects to work with :code:`wcosmo`.

By default, this provides a drop-in replacement for :code:`astropy.FlatLambdaCDM`
and :code:`astropy.FlatwCDM` cosmology objects along with all of the pre-defined
astropy cosmologies.

By changing the backend and disabling units, these classes can then be used with
:code:`numpy`, :code:`jax`, or :code:`cupy`.
"""

import sys

import astropy.cosmology as _acosmo
import numpy as xp
from astropy import units

from .utils import autodoc, convert_quantity_if_necessary, method_autodoc, strip_units
from .wcosmo import *

USE_UNITS = True

__all__ = [
    "FlatwCDM",
    "FlatLambdaCDM",
    "available",
]


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

    @property
    def H0(self):
        return self._H0

    @H0.setter
    def H0(self, value):
        self._H0 = convert_quantity_if_necessary(value, unit="km s^-1 Mpc^-1")

    @property
    def _kwargs(self):
        kwargs = {"H0": self.H0, "Om0": self.Om0, "w0": self.w0, "method": self.method}
        if not USE_UNITS:
            kwargs = {key: strip_units(value) for key, value in kwargs.items()}
        return kwargs

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
        kwargs = self._kwargs
        kwargs.pop("method")
        return hubble_parameter(z, **kwargs)

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
        value = 5 * xp.log10(xp.abs(distance)) + 25
        if USE_UNITS:
            value <<= units.mag
        return value

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


class FlatwCDM(WCosmoMixin):
    def __init__(
        self,
        H0,
        Om0,
        w0=-1,
        Tcmb0=None,
        Neff=None,
        m_nu=None,
        Ob0=None,
        *,
        zmin=1e-4,
        zmax=100,
        method="pade",
        name=None,
        meta=None,
    ):
        """FLRW cosmology with a constant dark energy EoS and no spatial curvature.

        This has one additional attribute beyond those of FLRW.

        Docstring copied from :code:`astropy.cosmology.flrw.wcdm.FlatwCDM`

        Parameters
        ----------
        H0 : float or scalar quantity-like ['frequency']
            Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

        Om0 : float
            Omega matter: density of non-relativistic matter in units of the
            critical density at z=0.

        w0 : float, optional
            Dark energy equation of state at all redshifts. This is
            pressure/density for dark energy in units where c=1. A cosmological
            constant has w0=-1.0.

        Tcmb0 : float or scalar quantity-like ['temperature'], optional
            Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K].
            Setting this to zero will turn off both photons and neutrinos
            (even massive ones).

        Neff : float, optional
            Effective number of Neutrino species. Default 3.04.

        m_nu : quantity-like ['energy', 'mass'] or array-like, optional
            Mass of each neutrino species in [eV] (mass-energy equivalency enabled).
            If this is a scalar Quantity, then all neutrino species are assumed to
            have that mass. Otherwise, the mass of each species. The actual number
            of neutrino species (and hence the number of elements of m_nu if it is
            not scalar) must be the floor of Neff. Typically this means you should
            provide three neutrino masses unless you are considering something like
            a sterile neutrino.

        Ob0 : float or None, optional
            Omega baryons: density of baryonic matter in units of the critical
            density at z=0.  If this is set to None (the default), any computation
            that requires its value will raise an exception.

        method: str (optional, keyword-only)
            The integration method, should be one of :code:`pade` or :code:`analytic`
            for the pade approximation or analytic hypergeometric methods
            respectively.

        name : str or None (optional, keyword-only)
            Name for this cosmological object.

        meta : mapping or None (optional, keyword-only)
            Metadata for the cosmology, e.g., a reference.

        Examples
        --------
        >>> from astropy.cosmology import FlatwCDM
        >>> cosmo = FlatwCDM(H0=70, Om0=0.3, w0=-0.9)

        The comoving distance in Mpc at redshift z:

        >>> z = 0.5
        >>> dc = cosmo.comoving_distance(z)
        """
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.zmin = zmin
        self.zmax = zmax
        self.method = method
        self.name = name
        self.meta = meta


class FlatLambdaCDM(WCosmoMixin):
    def __init__(
        self,
        H0,
        Om0,
        Tcmb0=None,
        Neff=None,
        m_nu=None,
        Ob0=None,
        *,
        zmin=1e-4,
        zmax=100,
        method="pade",
        name=None,
        meta=None,
    ):
        """FLRW cosmology with a cosmological constant and no curvature.

        This has no additional attributes beyond those of FLRW.

        Docstring copied from :code:`astropy.cosmology.flrw.lambdacdm.FlatLambdaCDM`

        Parameters
        ----------
        H0 : float or scalar quantity-like ['frequency']
            Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

        Om0 : float
            Omega matter: density of non-relativistic matter in units of the
            critical density at z=0.

        Tcmb0 : float or scalar quantity-like ['temperature'], optional
            Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K].
            Setting this to zero will turn off both photons and neutrinos
            (even massive ones).

        Neff : float, optional
            Effective number of Neutrino species. Default 3.04.

        m_nu : quantity-like ['energy', 'mass'] or array-like, optional
            Mass of each neutrino species in [eV] (mass-energy equivalency enabled).
            If this is a scalar Quantity, then all neutrino species are assumed to
            have that mass. Otherwise, the mass of each species. The actual number
            of neutrino species (and hence the number of elements of m_nu if it is
            not scalar) must be the floor of Neff. Typically this means you should
            provide three neutrino masses unless you are considering something like
            a sterile neutrino.

        Ob0 : float or None, optional
            Omega baryons: density of baryonic matter in units of the critical
            density at z=0.  If this is set to None (the default), any computation
            that requires its value will raise an exception.

        method: str (optional, keyword-only)
            The integration method, should be one of :code:`pade` or :code:`analytic`
            for the pade approximation or analytic hypergeometric methods
            respectively.

        name : str or None (optional, keyword-only)
            Name for this cosmological object.

        meta : mapping or None (optional, keyword-only)
            Metadata for the cosmology, e.g., a reference.

        Examples
        --------
        >>> from astropy.cosmology import FlatLambdaCDM
        >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        The comoving distance in Mpc at redshift z:

        >>> z = 0.5
        >>> dc = cosmo.comoving_distance(z)
        """
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = -1
        self.zmin = zmin
        self.zmax = zmax
        self.method = method
        self.name = name
        self.meta = meta


_known_cosmologies = dict()


def __getattr__(name):
    if f"{name}_{xp.__name__}" in _known_cosmologies:
        return _known_cosmologies[f"{name}_{xp.__name__}"]
    elif name not in __all__:
        alt = _acosmo.__getattr__(name)
        params = {
            key: convert_quantity_if_necessary(arg)
            for key, arg in alt.parameters.items()
        }
        cosmo = FlatLambdaCDM(**params)
        _known_cosmologies[f"{name}_{xp.__name__}"] = cosmo
        return cosmo


class _Available:

    def keys(self):
        return ("FlatLambdaCDM", "FlatwCDM") + _acosmo.available

    def __getitem__(self, key):
        return getattr(sys.modules[__name__], key)

    def __repr__(self):
        return repr(self.keys())


available = _Available()
