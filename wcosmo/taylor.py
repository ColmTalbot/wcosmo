r"""
Utilities based on the Taylor expansion of the functions of the form
:math:`\frac{(1 + z)^k}{E(z)}` for flat wCDM cosmology.
These functions and their integral are estimated using the Pade approximation.
"""

import numpy as np
from plum import Function, dispatch
from scipy.interpolate import pade as sc_pade

from .utils import array_namespace, autodoc

__all__ = [
    "flat_wcdm_pade_coefficients",
    "flat_wcdm_taylor_expansion",
    "indefinite_integral_pade",
]


pade = Function(sc_pade)


@dispatch
def pade(*args, **kwargs):  # noqa: F811
    p, q = sc_pade(*args, **kwargs)
    return p.coeffs, q.coeffs


def _binomial_coefficients(*, xp):
    return xp.array(
        [
            1,
            -1 / 2,
            3 / 8,
            -5 / 16,
            35 / 128,
            -63 / 256,
            231 / 1024,
            -429 / 2048,
            6435 / 32768,
            -12155 / 65536,
            46189 / 262144,
            -88179 / 524288,
            676039 / 4194304,
            -1300075 / 8388608,
            5014575 / 33554432,
            -9694845 / 67108864,
            300540195 / 268435456,
        ]
    )


@autodoc
def flat_wcdm_taylor_expansion(w0, zpower=0, *, xp=np):
    r"""
    Taylor coefficients expansion of :math:`E(z)` as as a function
    of :math:`w_0` and an arbitrary power :math:`k` of :math:`1 + z`.

    .. math::

        F(x; w_0; k) = 2\sum_{{n=0}}^{{\infty}} \binom{{-\frac{{1}}{{2}}}}{{n}}
        \frac{{x^n}}{{\left(1 - 2k - 6nw_0\right)}}

    We include terms up to :math:`n=16`.

    Parameters
    ----------
    {w0}
    {zpower}

    Returns
    -------
    array_like
        The Taylor expansion coefficients.
    """
    what = (w0 + abs(w0)) / 2
    denominator = 1 - 2 * zpower + 6 * abs(w0) * xp.arange(0, 17) + 3 * what
    return _binomial_coefficients(xp=xp) / denominator


@autodoc
def flat_wcdm_pade_coefficients(w0=-1, zpower=0, *, xp=np):
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
    coeffs = flat_wcdm_taylor_expansion(w0, zpower=zpower, xp=xp)
    p, q = pade(coeffs, len(coeffs) // 2, len(coeffs) // 2)
    return p, q


@autodoc
def indefinite_integral_pade(z, Om0, w0=-1, zpower=0):
    r"""
    Compute the Pade approximation to :math:`(1+z)^k / E(z)` as described in
    arXiv:1111.6396. We extend it to include a variable dark energy
    equation of state, other integrands via powers of :math:`1 + z` and
    include more terms in the expansion

    .. math::

        I(z; \Omega_{{m, 0}}, w_0) = \frac{{-2 (1 + z)^{{\frac{{1}}{{2}}-k}}}}
        {{\Omega_{{m, 0}}^{{\frac{{1}}{{2}}}}}} \Phi(z; \Omega_{{m, 0}}, w_0).

    .. math::

        \Phi(z; \Omega_{{m, 0}}, w_0) =
        \frac{{\sum_i^n \alpha_i x^i}}{{1 + \sum_{{j=1}}^m \beta_j x^j}}.

    Here the expansion is in terms of

    .. math::

        x = \left(\frac{{1 - \Omega_{{m, 0}}}}{{\Omega_{{m, 0}}}}\right)
        (1 + z)^{{3 w_0}}.

    In practice we use :math:`m=n=7` whereas Adachi and Kasai use :math:`m=n=3`.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    I: array_like
        The indefinite integral of :math:`(1+z)^k / E(z)`
    """
    xp = array_namespace(z)
    what = (w0 + abs(w0)) / 2
    sign = xp.sign(xp.array(w0))
    abs_sign = abs(sign)
    gamma = (Om0 ** (sign - abs_sign) * (1 - Om0) ** (-sign - abs_sign)) ** 0.25
    normalization = -2 * gamma * (1 + z) ** (zpower - 0.5 - 3 * what / 2)
    Om0 = xp.array(Om0)
    with np.errstate(divide="ignore"):
        x = (Om0 / (1 - Om0)) ** sign * (1 + z) ** (-3 * abs(w0))
    p, q = flat_wcdm_pade_coefficients(w0=w0, zpower=zpower, xp=xp)
    return normalization * (xp.polyval(p, x) / xp.polyval(q, x)) ** abs_sign
