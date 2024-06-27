"""
Core implementation of cosmology functionality.
"""

import sys

import numpy as xp
from astropy import cosmology as _acosmo

from .taylor import analytic_integral
from .utils import (
    GYR_KM_PER_S_MPC,
    SPEED_OF_LIGHT_KM_PER_S,
    autodoc,
    maybe_jit,
    method_autodoc,
)

__all__ = [
    "FlatwCDM",
    "FlatLambdaCDM",
    "Planck13",
    "Planck15",
    "Planck18",
    "WMAP1",
    "WMAP3",
    "WMAP5",
    "WMAP7",
    "WMAP9",
    "available",
    "comoving_distance",
    "comoving_volume",
    "detector_to_source_frame",
    "differential_comoving_volume",
    "dDLdz",
    "efunc",
    "hubble_distance",
    "hubble_time",
    "inv_efunc",
    "lookback_time",
    "luminosity_distance",
    "source_to_detector_frame",
    "z_at_value",
]


@autodoc
@maybe_jit
def efunc(z, Om0, w0=-1):
    r"""
    Compute the :math:`E(z)` function for a flat wCDM cosmology.

    .. math::

        E(z; \Omega_{{m,0}}, w_0) = \sqrt{{\Omega_{{m,0}} (1 + z)^3
        + (1 - \Omega_{{m,0}}) (1 + z)^{{3(1 + w_0)}}}}

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    E(z): array_like
        The E(z) function
    """
    zp1 = 1 + z
    return (Om0 * zp1**3 + (1 - Om0) * zp1 ** (3 * (1 + w0))) ** 0.5


@maybe_jit
@autodoc
def inv_efunc(z, Om0, w0=-1):
    """
    Compute the inverse of the E(z) function for a flat wCDM cosmology.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    inv_efunc: array_like
        The inverse of the E(z) function
    """
    return 1 / efunc(z, Om0, w0)


@autodoc
def hubble_distance(H0):
    r"""
    Compute the Hubble distance :math:`D_H = c H_0^{{-1}}` in Mpc.

    Parameters
    ----------
    {H0}

    Returns
    -------
    D_H: float
        The Hubble distance in Mpc
    """
    return SPEED_OF_LIGHT_KM_PER_S / H0


@autodoc
def hubble_time(H0):
    r"""
    Compute the Hubble time :math:`t_H = H_0^{{-1}}` in Gyr.

    Parameters
    ----------
    {H0}

    Returns
    -------
    t_H: float
        The Hubble time in Gyr
    """
    return GYR_KM_PER_S_MPC / H0


@autodoc
def hubble_parameter(z, H0, Om0, w0=-1):
    r"""
    Compute the Hubble parameter :math:`H(z)` for a flat wCDM cosmology.

    .. math::

        H(z; H_0, \Omega_{{m,0}}, w_0) = \frac{{d_H(H_0)}}{{E(z; \Omega_{{m,0}}, w_0)}}

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    H(z): array_like
        The Hubble parameter
    """
    return hubble_distance(H0=H0) * inv_efunc(z=z, H0=H0, Om0=Om0, w0=w0)


@maybe_jit
@autodoc
def comoving_distance(z, H0, Om0, w0=-1):
    r"""
    Compute the comoving distance using an analytic integral of the
    Pade approximation.

    .. math::

        d_{{C}} = d_{{H}} \int_{{0}}^{{z}}
        \frac{{dz'}}{{E(z'; \Omega_{{m,0}}, w_0)}}

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    comoving_distance: array_like
        The comoving distance in Mpc
    """
    return analytic_integral(z, Om0, w0) * hubble_distance(H0)


@maybe_jit
@autodoc
def lookback_time(z, H0, Om0, w0=-1):
    r"""
    Compute the lookback time using an analytic integral of the
    Pade approximation.

    .. math::

        t_{{L}} = t_{{H}} \int_{{0}}^{{z}}
        \frac{{dz'}}{{(1 + z')E(z'; \Omega_{{m,0}}, w_0)}}

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    lookback_time: array_like
        The lookback time in km / s / Mpc
    """
    return analytic_integral(z, Om0, w0, zpower=-1) * hubble_time(H0)


@maybe_jit
@autodoc
def absorption_distance(z, Om0, w0=-1):
    r"""
    Compute the absorption distance using an analytic integral of the
    Pade approximation.

    .. math::

        d_{{A}} = \int_{{0}}^{{z}}
        \frac{{dz' (1 + z')^2}}{{E(z'; \Omega_{{m,0}}, w_0)}}

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    absorption_distance: array_like
        The absorption distance in Mpc
    """
    return analytic_integral(z, Om0, w0, zpower=2)


@maybe_jit
@autodoc
def luminosity_distance(z, H0, Om0, w0=-1):
    r"""
    Compute the luminosity distance using an analytic integral of the
    Pade approximation.

    .. math::

        d_L = (1 + z') d_{{H}} \int_{{0}}^{{z}}
        \frac{{dz'}}{{E(z'; \Omega_{{m,0}}, w_0)}}

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    luminosity_distance: array_like
        The luminosity distance in Mpc
    """
    return (1 + z) * comoving_distance(z, H0, Om0, w0)


@maybe_jit
@autodoc
def dDLdz(z, H0, Om0, w0=-1):
    r"""
    The Jacobian for the conversion of redshift to luminosity distance.

    .. math::

        \frac{{dd_{{L}}}}{{z}} = d_C(z; H_0, \Omega_{{m,0}}, w_0)
        + (1 + z) d_{{H}} E(z; \Omega_{{m, 0}}, w0)

    Here :math:`d_{{C}}` is comoving distance and :math:`d_{{H}}` is the Hubble
    distance.

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    dDLdz: array_like
        The derivative of the luminosity distance with respect to redshift
        in Mpc

    Notes
    -----
    This function does not have a direct analog in the :code:`astropy`
    cosmology objects, but is needed for accounting for expressing
    distributions of redshift as distributions over luminosity distance.
    """
    dC = comoving_distance(z, H0=H0, Om0=Om0, w0=w0)
    Ez_i = inv_efunc(z, Om0=Om0, w0=w0)
    D_H = hubble_distance(H0)
    return dC + (1 + z) * D_H * Ez_i


@autodoc
def z_at_value(func, fval, zmin=1e-4, zmax=100, **kwargs):
    """
    Compute the redshift at which a function equals a given value.

    This follows the approach in :code:`astropy`'s :code:`z_at_value`
    function closely, but uses linear interpolation instead of a root finder.

    Parameters
    ----------
    func: callable
        The function to evaluate, e.g., :code:`Planck15.luminosity_distance`,
        this should take :code:`fval` as the only input.
    fval: float
        The value of the function at the desired redshift
    {zmin}
    {zmax}

    Returns
    -------
    z: float
        The redshift at which the function equals the desired value
    """
    zs = xp.logspace(xp.log10(zmin), xp.log10(zmax), 1000)
    return xp.interp(
        xp.asarray(fval), func(zs, **kwargs), zs, left=zmin, right=zmax, period=None
    )


@maybe_jit
@autodoc
def differential_comoving_volume(z, H0, Om0, w0=-1):
    r"""
    Compute the differential comoving volume element.

    .. math::

        \frac{{dV_{{C}}}}{{dz}} = d_C^2(z; H_0, \Omega_{{m,0}}, w_0) d_H
        E(z; \Omega_{{m, 0}}, w_0)

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    dVc: array_like
        The differential comoving volume element in :math:`\rm{{Gpc}}^3`
    """
    dC = comoving_distance(z, H0, Om0, w0=w0)
    Ez_i = inv_efunc(z, Om0, w0=w0)
    D_H = hubble_distance(H0)
    return dC**2 * D_H * Ez_i


@maybe_jit
@autodoc
def detector_to_source_frame(m1z, m2z, dL, H0, Om0, w0=-1, zmin=1e-4, zmax=100):
    """
    Convert masses and luminosity distance from the detector frame to
    source frame masses and redshift.

    This passes through the arguments to `z_at_value` to compute the
    redshift.

    Parameters
    ----------
    {m1z}
    {m2z}
    {dL}
    {H0}
    {Om0}
    {w0}
    {zmin}
    {zmax}

    Returns
    -------
    m1, m2, z: array_like
        The primary and secondary masses in the source frame and the redshift
    """
    z = z_at_value(luminosity_distance, dL, zmin=zmin, zmax=zmax, H0=H0, Om0=Om0, w0=w0)
    m1 = m1z / (1 + z)
    m2 = m2z / (1 + z)
    return m1, m2, z


@maybe_jit
@autodoc
def source_to_detector_frame(m1, m2, z, H0, Om0, w0=-1):
    """
    Convert masses and redshift from the source frame to the detector frame.

    Parameters
    ----------
    {m1}
    {m2}
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    m1z, m2z, dL: array_like
        The primary and secondary masses in the detector frame and the
        luminosity distance
    """
    dL = luminosity_distance(z, H0, Om0, w0=w0)
    return m1 * (1 + z), m2 * (1 + z), dL


@maybe_jit
@autodoc
def comoving_volume(z, H0, Om0, w0=-1):
    r"""
    Compute the comoving volume out to redshift z.

    .. math::

        V_C = \frac{{4\pi}}{{3}} d^3_C(z; H_0, \Omega_{{m,0}}, w_0)

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    Vc: array_like
        The comoving volume in :math:`\rm{{Gpc}}^3`
    """
    return 4 / 3 * xp.pi * comoving_distance(z, H0, Om0, w0=w0) ** 3


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
    supports implicit conversion. Additionally, we don't overwrite the various integrands
    and other utility methods, e.g., :code:`clone`.

    We include the following methods that are not present in :code:`astropy`:

    - :code:`dLdH` - derivative of the luminosity distance w.r.t. the Hubble distance
    - :code:`dDLdz` - Jacobian for the conversion of luminosity distance to redshift
    - :code:`detector_to_source_frame` - convert masses and luminosity distance from
      the detector frame to the source frame, also returns the jacobian,
      see :func:`detector_to_source_frame`
    - :code:`source_to_detector_frame` - convert masses and redshift from the source
      frame to the detector frame, see :func:`source_to_detector_frame`
    """

    @property
    def _kwargs(self):
        return {"H0": self.H0.value, "Om0": self.Om0, "w0": self.w0}

    @property
    def hubble_distance(self):
        """
        Compute the Hubble distance :math:`D_H = c H_0^{-1}` in Mpc.

        Returns
        -------
        D_H: float
            The Hubble distance in Mpc
        """
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
    def detector_to_source_frame(self, m1z, m2z, dL):
        return detector_to_source_frame(
            m1z, m2z, dL, **self._kwargs, zmin=self.zmin, zmax=self.zmax
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
        return 5 * xp.log10(xp.abs(self.luminosity_distance(z))) + 25

    @autodoc
    def critical_density(self, z):
        """Critical density in grams per cubic cm at redshift ``z``.

        Parameters
        ----------
        {z}

        Returns
        -------
        rho: array-like
            Critical density in g/cm^3 at each input redshift.
        """
        return self._critical_density0.value * (self.efunc(z)) ** 2

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


class FlatwCDM(WCosmoMixin, _acosmo.FlatwCDM):
    pass


class FlatLambdaCDM(WCosmoMixin, _acosmo.FlatLambdaCDM):

    def __post_init__(self):
        self.w0 = -1.0


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
