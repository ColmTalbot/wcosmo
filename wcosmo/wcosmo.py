from functools import partial

import numpy as xp

from .utils import maybe_jit, pade


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


_cosmology_docstrings_ = dict(
    z="""z: array_like
        Redshift""",
    Om0="""Om0: array_like
        The matter density fraction""",
    w0="""w0: array_like
        The (constant) equation of state parameter for dark energy""",
    H0="""H0: float
        The Hubble constant in km/s/Mpc""",
    zmin="""zmin: float
        The minimum redshift used in the conversion from distance to redshift,
        default=1e-4""",
    zmax="""zmax: float
        The maximum redshift used in the conversion from distance to redshift,
        default=100""",
    name="""name: str
        The name for the cosmology, mostly used for fixed instances""",
    meta="""meta: dict
        Additional metadata describing the cosmology, e.g., citation
        information""",
)


def _autodoc(func):
    """
    Simple decorator to mark that a docstring needs formatting
    """
    func.__doc__ = func.__doc__.format(**_cosmology_docstrings_)
    return func


@_autodoc
def flat_wcdm_taylor_expansion(w0):
    r"""
    Taylor coefficients expansion of E(z) as as a function
    of w.

    .. math::

        F(x; w) = \sum_{{n=0}}^{{\infty}} \binom{{-1/2}}{{n}} x^n / (1/2 - 3wn)
                = 2 \sum_{{n=0}}^{{\infty}} \binom{{-1/2}}{{n}} x^n / (1 - 6wn)

    Parameters
    ----------
    {w0}

    Returns
    -------
    xp.ndarray
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
    """
    Compute the E(z) function for a flat Lambda CDM cosmology.

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
    Compute the inverse of the E(z) function for a flat Lambda CDM cosmology.

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
    """
    Compute the Hubble distance.

    Parameters
    ----------
    {H0}

    Returns
    -------
    D_H: float
        The Hubble distance in Mpc

    Notes
    -----
    I hard code the speed of light in km/s
    """
    speed_of_light = 299792.458
    return speed_of_light / H0


@_autodoc
def Phi(z, Om0, w0=-1):
    """
    Compute the Pade approximation to 1 / E(z) as described in arXiv:1111.6396.
    We extend it to include a variable dark energy equation of state and
    include more terms in the expansion.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    Phi: array_like
        The Pade approximation to 1 / E(z)
    """
    x = (1 - Om0) / Om0 * (1 + z) ** (-3)
    p, q = flat_wcdm_pade_coefficients(w0=w0)
    return xp.polyval(p, x) / xp.polyval(q, x)


@_autodoc
def flat_wcdm_pade_coefficients(w0=-1):
    """
    Compute the Pade coefficients as described in arXiv:1111.6396.
    I expand the series to a bunch more terms to get a better fit.


    Parameters
    ----------
    {w0}

    Returns
    -------
    p, q: xp.ndarray
        The Pade coefficients
    """
    coeffs = flat_wcdm_taylor_expansion(w0=w0)
    p, q = pade(coeffs, len(coeffs) // 2, len(coeffs) // 2)
    return p, q


@_autodoc
def analytic_integral(z, Om0, w0=-1):
    """
    Evaluate the analytic integral of 1 / E(z) from infty to z
    assuming the Pade approximation to 1 / E(z).

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    integral: array_like
        The integral of 1 / E(z) from infty to z
    """
    return -2 / Om0**0.5 * Phi(z, Om0, w0) / (1 + z) ** 0.5


@maybe_jit
@_autodoc
def comoving_distance(z, H0, Om0, w0=-1):
    """
    Compute the comoving distance using an analytic integral of the
    Pade approximation.

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
    integral = analytic_integral(z, Om0=Om0) - analytic_integral(0, Om0=Om0, w0=w0)
    return integral * hubble_distance(H0)


@maybe_jit
@_autodoc
def luminosity_distance(z, H0, Om0, w0=-1):
    """
    Compute the luminosity distance using an analytic integral of the
    Pade approximation.

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
    """
    The Jacobian for the conversion of redshift to luminosity distance.

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

    Notes
    -----
    This function does not have a direct analog in the `astropy` cosmology
    objects, but is needed for accounting for fitting in luminosity distance
    """
    dC = comoving_distance(z, H0=H0, Om0=Om0, w0=w0)
    Ez_i = inv_efunc(z, Om0=Om0, w0=w0)
    D_H = hubble_distance(H0)
    return dC + (1 + z) * D_H * Ez_i


@_autodoc
def z_at_value(func, fval, zmin=1e-4, zmax=100):
    """
    Compute the redshift at which a function equals a given value.

    This follows the approach in `astropy`'s `z_at_value` function
    closely, but uses linear interpolation instead of a root finder.

    Parameters
    ----------
    func: callable
        The function to evaluate, e.g., luminosity_distance
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
    """
    Compute the differential comoving volume element.

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    dVc: array_like
        The differential comoving volume element in Gpc^3
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
    m1z: array_like
        The primary mass in the detector frame
    m2z: array_like
        The secondary mass in the detector frame
    dL: array_like
        The luminosity distance in Mpc
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
    m1: array_like
        The primary mass in the source frame
    m2: array_like
        The secondary mass in the source frame
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
    """
    Compute the comoving volume out to redshift z.

    Parameters
    ----------
    {z}
    {H0}
    {Om0}
    {w0}

    Returns
    -------
    Vc: array_like
        The comoving volume in Gpc^3
    """
    return 4 / 3 * xp.pi * comoving_distance(z, H0, Om0, w0=w0) ** 3


@_autodoc
class FlatwCDM:
    r"""
    Implementation of FlatwCDM cosmology to (approximately) match the astropy
    API.

    .. math::

        E(z) = \sqrt{{\Omega_{{m,0}} (1 + z)^3 + (1 - \Omega_{{m,0}}) (1 + z)^{{3(1 - w0)}}}}

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
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        if meta is None:
            meta = {}
        self._meta = meta

    @property
    def _kwargs(self):
        return {"H0": self.H0, "Om0": self.Om0, "w0": self.w0}

    @property
    def hubble_distance(self):
        return hubble_distance(self.H0)

    def luminosity_distance(self, z):
        return luminosity_distance(z, **self._kwargs)

    def dLdH(self, z):
        return self.luminosity_distance(z) / self.hubble_distance

    def dDLdz(self, z):
        return dDLdz(z, **self._kwargs)

    def differential_comoving_volume(self, z):
        return differential_comoving_volume(z, **self._kwargs)

    def detector_to_source_frame(self, m1z, m2z, dL):
        return detector_to_source_frame(
            m1z, m2z, dL, **self._kwargs, zmin=self.zmin, zmax=self.zmax
        )

    def source_to_detector_frame(self, m1, m2, z):
        return source_to_detector_frame(m1, m2, z, **self._kwargs)

    def log_differential_comoving_volume(self, z):
        return xp.log(self.differential_comoving_volume(z))

    def efunc(self, z):
        return efunc(z, self.Om0)

    def inv_efunc(self, z):
        return inv_efunc(z, self.Om0)

    def H(self, z):
        return self.H0 * self.efunc(z)

    def comoving_distance(self, z):
        return comoving_distance(z, **self._kwargs)

    def comoving_volume(self, z):
        return comoving_volume(z, **self._kwargs)


@_autodoc
class FlatLambdaCDM(FlatwCDM):
    r"""
    Implementation of FlatLambdaCDM cosmology to (approximately) match the
    astropy API. This is the same as the :code:`FlatwCDM` with :code:`w0=-1`.

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

    def __init__(self, H0, Om0, *, zmin=0.0001, zmax=100, name=None, meta=None):
        super().__init__(H0, Om0, w0=-1, zmin=zmin, zmax=zmax, name=name, meta=meta)


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
