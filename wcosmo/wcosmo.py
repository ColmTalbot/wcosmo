"""
Core implementation of cosmology functionality.
"""

from functools import partial

import numpy as xp

from .utils import maybe_jit, pade, _autodoc, _method_autodoc


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
    "flat_wcdm_taylor_expansion",
    "efunc",
    "inv_efunc",
    "hubble_distance",
    "comoving_distance",
    "luminosity_distance",
    "dDLdz",
    "z_at_value",
    "detector_to_source_frame",
    "source_to_detector_frame",
    "differential_comoving_volume",
    "comoving_volume",
]


@_autodoc
def flat_wcdm_taylor_expansion(w0):
    r"""
    Taylor coefficients expansion of :math:`E(z)` as as a function
    of :math:`w_0`.

    .. math::

        F(x; w_0) = 2\sum_{{n=0}}^{{\infty}} \binom{{-\frac{{1}}{{2}}}}{{n}}
        \frac{{x^n}}{{\left(1 - 6nw_0\right)}}

    We include terms up to :math:`n=16`.

    Parameters
    ----------
    {w0}

    Returns
    -------
    array_like
        The Taylor expansion coefficients.
    """
    return xp.array(
        [
            w0**0,
            -1 / (2 * (1 - 6 * w0)),
            3 / (8 * (1 - 12 * w0)),
            -5 / (16 * (1 - 18 * w0)),
            35 / (128 * (1 - 24 * w0)),
            -63 / (256 * (1 - 30 * w0)),
            231 / (1024 * (1 - 36 * w0)),
            -429 / (2048 * (1 - 42 * w0)),
            6435 / (32768 * (1 - 48 * w0)),
            -12155 / (65536 * (1 - 54 * w0)),
            46189 / (262144 * (1 - 60 * w0)),
            -88179 / (524288 * (1 - 66 * w0)),
            676039 / (4194304 * (1 - 72 * w0)),
            -1300075 / (8388608 * (1 - 78 * w0)),
            5014575 / (33554432 * (1 - 84 * w0)),
            -9694845 / (67108864 * (1 - 90 * w0)),
            300540195 / (268435456 * (1 - 96 * w0)),
        ]
    )


@maybe_jit
@_autodoc
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
@_autodoc
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


@_autodoc
def hubble_distance(H0):
    r"""
    Compute the Hubble distance :math:`D_H = c H_0^{{-1}}`.

    Parameters
    ----------
    {H0}

    Returns
    -------
    D_H: float
        The Hubble distance in Mpc
    """
    speed_of_light = 299792.458
    return speed_of_light / H0


@_autodoc
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


@_autodoc
def Phi(z, Om0, w0=-1):
    r"""
    Compute the Pade approximation to :math:`1 / E(z)` as described in
    arXiv:1111.6396. We extend it to include a variable dark energy
    equation of state and include more terms in the expansion

    .. math::

        \Phi(z; \Omega_{{m, 0}}, w_0) =
        \frac{{\sum_i^n \alpha_i x^i}}{{1 + \sum_{{j=1}}^m \beta_j x^j}}.

    Here the expansion is in terms of

    .. math::

        x = \left(\frac{{1 - \Omega_{{m, 0}}}}{{\Omega_{{m, 0}}}}\right) (1 + z)^{{3 w_0}}.

    In practice we use :math:`m=n=7` whereas Adachi and Kasai use :math:`m=n=3`.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    Phi: array_like
        The Pade approximation to :math:`1 / E(z)`
    """
    x = (1 - Om0) / Om0 * (1 + z) ** (3 * w0)
    p, q = flat_wcdm_pade_coefficients(w0=w0)
    return xp.polyval(p, x) / xp.polyval(q, x)


@_autodoc
def flat_wcdm_pade_coefficients(w0=-1):
    """
    Compute the Pade coefficients as described in arXiv:1111.6396.
    We make two primary changes:

    - allow a variable dark energy equation of state :math:`w_0` by changing
      the definition of :math:`x`.
    - include more terms (17) in the Taylor expansion.

    Parameters
    ----------
    {w0}

    Returns
    -------
    p, q: array_like
        The Pade coefficients
    """
    coeffs = flat_wcdm_taylor_expansion(w0=w0)
    p, q = pade(coeffs, len(coeffs) // 2, len(coeffs) // 2)
    return p, q


@_autodoc
def analytic_integral(z, Om0, w0=-1):
    r"""
    .. math::

        f(z; \Omega_{{m, 0}}, w_0) = \int_{{\infty}}^{{z}}
        \frac{{dz'}}{{E(z'; \Omega_{{m, 0}}, w_0)}}
        = \frac{{-2\Phi(z; \Omega_{{m, 0}}, w_0)}}{{\sqrt{{\Omega_{{m, 0}}(1 + z)}}}}.

    The integral is approximated using the Pade approximation and is up
    to a factor the term in the braces in (1.1) of Adachi and Kasai.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    integral: array_like
        The integral of :math:`1 / E(z)` from :math:`\infty` to :math:`z`
    """
    return -2 / Om0**0.5 * Phi(z, Om0, w0) / (1 + z) ** 0.5


@maybe_jit
@_autodoc
def comoving_distance(z, H0, Om0, w0=-1):
    r"""
    Compute the comoving distance using an analytic integral of the
    Pade approximation.

    .. math::

        d_{{C}} = \frac{{c}}{{H_0}} \frac{{2}}{{\sqrt{{\Omega_{{m,0}}}}}}
        \left( \Phi(0; \Omega_{{m, 0}}, w_0)
        - \frac{{\Phi(z; \Omega_{{m, 0}}, w_0)}}{{\sqrt{{(1 + z)}}}}\right)

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
    integral = analytic_integral(z, Om0=Om0, w0=w0) - analytic_integral(0, Om0=Om0, w0=w0)
    return integral * hubble_distance(H0)


@maybe_jit
@_autodoc
def luminosity_distance(z, H0, Om0, w0=-1):
    r"""
    Compute the luminosity distance using an analytic integral of the
    Pade approximation.

    .. math::

        d_L = \frac{{c}}{{H_0}} \frac{{2(1 + z)}}{{\sqrt{{\Omega_{{m,0}}}}}}
        \left( \Phi(0; \Omega_{{m, 0}}, w_0)
        - \frac{{\Phi(z; \Omega_{{m, 0}}, w_0)}}{{\sqrt{{(1 + z)}}}}\right)

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
@_autodoc
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


@_autodoc
def z_at_value(func, fval, zmin=1e-4, zmax=100):
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
    return xp.interp(xp.asarray(fval), func(zs), zs, left=zmin, right=zmax, period=None)


@maybe_jit
@_autodoc
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
@_autodoc
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
    distance_func = partial(luminosity_distance, H0=H0, Om0=Om0, w0=w0)
    z = z_at_value(distance_func, dL, zmin=zmin, zmax=zmax)
    m1 = m1z / (1 + z)
    m2 = m2z / (1 + z)
    return m1, m2, z


@maybe_jit
@_autodoc
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
@_autodoc
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


@_autodoc
class FlatwCDM:
    r"""
    Implementation of flat wCDM cosmology to (approximately) match the
    :code:`astropy` API.

    .. math::

        E(z) = \sqrt{{\Omega_{{m,0}} (1 + z)^3
        + (1 - \Omega_{{m,0}}) (1 + z)^{{3(1 + w_0)}}}}

    Parameters
    ----------
    {H0}
    {Om0}
    {w0}
    {zmin}
    {zmax}
    {name}
    {meta}
    """

    def __init__(
        self,
        H0,
        Om0,
        w0,
        *,
        zmin=1e-4,
        zmax=100,
        name=None,
        meta=None,
    ):
        self.H0 = H0
        self.Om0 = Om0
        self.w0 = w0
        self.zmin = zmin
        self.zmax = zmax
        self.name = name
        self.meta = meta

    @property
    def _kwargs(self):
        return {"H0": self.H0, "Om0": self.Om0, "w0": self.w0}

    @property
    def meta(self):
        """
        Meta data for the cosmology to hold additional information, e.g.,
        citation information
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        if meta is None:
            meta = {}
        self._meta = meta

    @property
    @_method_autodoc(strip_wcdm=True, alt=hubble_distance)
    def hubble_distance(self):
        return hubble_distance(self.H0)

    @_method_autodoc(strip_wcdm=True, alt=luminosity_distance)
    def luminosity_distance(self, z):
        return luminosity_distance(z, **self._kwargs)

    @_autodoc
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

    @_method_autodoc(strip_wcdm=True, alt=dDLdz)
    def dDLdz(self, z):
        return dDLdz(z, **self._kwargs)

    @_method_autodoc(strip_wcdm=True, alt=differential_comoving_volume)
    def differential_comoving_volume(self, z):
        return differential_comoving_volume(z, **self._kwargs)

    @_method_autodoc(strip_wcdm=True, alt=detector_to_source_frame)
    def detector_to_source_frame(self, m1z, m2z, dL):
        return detector_to_source_frame(
            m1z, m2z, dL, **self._kwargs, zmin=self.zmin, zmax=self.zmax
        )

    @_method_autodoc(strip_wcdm=True, alt=source_to_detector_frame)
    def source_to_detector_frame(self, m1, m2, z):
        return source_to_detector_frame(m1, m2, z, **self._kwargs)

    @_method_autodoc(strip_wcdm=True, alt=efunc)
    def efunc(self, z):
        return efunc(z, self.Om0, self.w0)

    @_method_autodoc(strip_wcdm=True, alt=inv_efunc)
    def inv_efunc(self, z):
        return inv_efunc(z, self.Om0, self.w0)

    @_method_autodoc(strip_wcdm=True, alt=hubble_parameter)
    def H(self, z):
        return hubble_parameter(z, **self._kwargs)

    @_method_autodoc(strip_wcdm=True, alt=comoving_distance)
    def comoving_distance(self, z):
        return comoving_distance(z, **self._kwargs)

    @_method_autodoc(strip_wcdm=True, alt=comoving_volume)
    def comoving_volume(self, z):
        return comoving_volume(z, **self._kwargs)


@_autodoc
class FlatLambdaCDM(FlatwCDM):
    r"""
    Implementation of a flat :math:`\Lambda\rm{{CDM}}` cosmology to
    (approximately) match the :code:`astropy` API. This is the same as
    the :code:`FlatwCDM` with :math:`w_0=-1`.

    .. math::

        E(z) = \sqrt{{\Omega_{{m,0}} (1 + z)^3 + (1 - \Omega_{{m,0}})}}

    Parameters
    ----------
    {H0}
    {Om0}
    {zmin}
    {zmax}
    {name}
    {meta}
    """

    def __init__(
        self,
        H0,
        Om0,
        *,
        zmin=1e-4,
        zmax=100,
        name=None,
        meta=None,
    ):
        super().__init__(
            H0=H0, Om0=Om0, w0=-1, zmin=zmin, zmax=zmax, name=name, meta=meta
        )


Planck13 = FlatLambdaCDM(H0=67.77, Om0=0.30712, name="Planck13")
Planck15 = FlatLambdaCDM(H0=67.74, Om0=0.3075, name="Planck15")
Planck18 = FlatLambdaCDM(H0=67.66, Om0=0.30966, name="Planck18")
WMAP1 = FlatLambdaCDM(H0=72.0, Om0=0.257, name="WMAP1")
WMAP3 = FlatLambdaCDM(H0=70.1, Om0=0.276, name="WMAP3")
WMAP5 = FlatLambdaCDM(H0=70.2, Om0=0.277, name="WMAP5")
WMAP7 = FlatLambdaCDM(H0=70.4, Om0=0.272, name="WMAP7")
WMAP9 = FlatLambdaCDM(H0=69.32, Om0=0.2865, name="WMAP9")
available = dict(
    Planck13=Planck13,
    Planck15=Planck15,
    Planck18=Planck18,
    WMAP1=WMAP1,
    WMAP3=WMAP3,
    WMAP5=WMAP5,
    WMAP7=WMAP7,
    WMAP9=WMAP9,
    FlatLambdaCDM=FlatLambdaCDM,
    FlatwCDM=FlatwCDM,
)
