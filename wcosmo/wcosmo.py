"""
Functional implementation of cosmological parameters.

.. Note::

    This module is intended to be used as a functional alternative to the
    :code:`astropy` class-based method. In general, units will be propagated
    when using :code:`numpy`, but all of the code can also be used without
    units. Special care is needed with :func:`hubble_distance` and
    :func:`hubble_time` as these functions use constants which have units
    unless :func:`wcosmo.utils.disable_units` has been called.
"""

import numpy as xp

from . import constants
from .integrate import analytic_integral
from .utils import autodoc, maybe_jit

__all__ = [
    "absorption_distance",
    "comoving_distance",
    "comoving_volume",
    "detector_to_source_frame",
    "differential_comoving_volume",
    "dDLdz",
    "efunc",
    "hubble_distance",
    "hubble_parameter",
    "hubble_time",
    "inv_efunc",
    "lookback_time",
    "absorption_distance",
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
    return constants.c_km_per_s / H0


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
    return constants.gyr_km_per_s_mpc / H0


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
    return H0 * efunc(z=z, Om0=Om0, w0=w0)


@maybe_jit
@autodoc
def comoving_distance(z, H0, Om0, w0=-1, method="pade"):
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
    return analytic_integral(z, Om0, w0, method=method) * hubble_distance(H0)


@maybe_jit
@autodoc
def lookback_time(z, H0, Om0, w0=-1, method="pade"):
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
    return analytic_integral(z, Om0, w0, zpower=-1, method=method) * hubble_time(H0)


@maybe_jit
@autodoc
def absorption_distance(z, Om0, w0=-1, method="pade"):
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
    return analytic_integral(z, Om0, w0, zpower=2, method=method)


@maybe_jit
@autodoc
def luminosity_distance(z, H0, Om0, w0=-1, method="pade"):
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
    return (1 + z) * comoving_distance(z, H0, Om0, w0, method=method)


@maybe_jit
@autodoc
def dDLdz(z, H0, Om0, w0=-1, method="pade"):
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
    dC = comoving_distance(z, H0=H0, Om0=Om0, w0=w0, method=method)
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
    fval: float | array-like
        The value of the function at the desired redshift
    {zmin}
    {zmax}

    Returns
    -------
    z: float
        The redshift at which the function equals the desired value
    """
    zs = xp.logspace(xp.log10(zmin), xp.log10(zmax), 1000)
    xs = func(zs, **kwargs)
    values = xp.interp(fval, xs, zs, left=zmin, right=zmax, period=None)
    from .utils import convert_quantity_if_necessary

    return convert_quantity_if_necessary(values, None)


@maybe_jit
@autodoc
def differential_comoving_volume(z, H0, Om0, w0=-1, method="pade"):
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
    dC = comoving_distance(z, H0, Om0, w0=w0, method=method)
    Ez_i = inv_efunc(z, Om0, w0=w0)
    D_H = hubble_distance(H0)
    return dC**2 * D_H * Ez_i / constants.steradian


@maybe_jit
@autodoc
def detector_to_source_frame(
    m1z, m2z, dL, H0, Om0, w0=-1, method="pade", zmin=1e-4, zmax=100
):
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
    z = z_at_value(
        luminosity_distance,
        dL,
        zmin=zmin,
        zmax=zmax,
        H0=H0,
        Om0=Om0,
        w0=w0,
        method=method,
    )
    return m1z / (1 + z), m2z / (1 + z), z


@maybe_jit
@autodoc
def source_to_detector_frame(m1, m2, z, H0, Om0, w0=-1, method="pade"):
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
    dL = luminosity_distance(z, H0, Om0, w0=w0, method=method)
    return m1 * (1 + z), m2 * (1 + z), dL


@maybe_jit
@autodoc
def comoving_volume(z, H0, Om0, w0=-1, method="pade"):
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
    return 4 / 3 * xp.pi * comoving_distance(z, H0, Om0, w0=w0, method=method) ** 3
