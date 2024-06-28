r"""
Utilities based on the Taylor expansion of the functions of the form
:math:`\frac{(1 + z)^k}{E(z)}` for flat wCDM cosmology.
These functions and their integral are estimated using the Pade approximation.
"""

import numpy as xp
from scipy.linalg import toeplitz

from .utils import autodoc

__all__ = [
    "analytic_integral",
    "flat_wcdm_pade_coefficients",
    "flat_wcdm_taylor_expansion",
]


def pade(an, m, n=None):
    """
    Return Pade approximation to a polynomial as the ratio of two polynomials.

    Parameters
    ----------
    an : (N,) array_like
        Taylor series coefficients.
    m : int
        The order of the returned approximating polynomial `q`.
    n : int, optional
        The order of the returned approximating polynomial `p`. By default,
        the order is ``len(an)-1-m``.

    Returns
    -------
    p, q : Polynomial class
        The Pade approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    Notes
    -----
    This code has been slightly edited from the scipy implementation to:

    - Use xp instead of np to support multiple backends
    - Directly use the fact that part of the matrix is Toeplitz

    """
    an = xp.asarray(an)
    if n is None:
        n = len(an) - 1 - m
        if n < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    if n < 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    N = m + n
    if N > len(an) - 1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    an = an[: N + 1]
    Akj = xp.eye(N + 1, n + 1, dtype=an.dtype)
    Bkj = toeplitz(xp.r_[0.0, -an[:-1]], xp.zeros(m))
    Ckj = xp.hstack((Akj, Bkj))
    pq = xp.linalg.solve(Ckj, an)
    p = pq[: n + 1]
    q = xp.r_[1.0, pq[n + 1 :]]
    return p[::-1], q[::-1]


@autodoc
def flat_wcdm_taylor_expansion(w0, zpower=0):
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
    power = 1 - 2 * zpower
    return xp.array(
        [
            w0**0 / power,
            -1 / (2 * (power - 6 * w0)),
            3 / (8 * (power - 12 * w0)),
            -5 / (16 * (power - 18 * w0)),
            35 / (128 * (power - 24 * w0)),
            -63 / (256 * (power - 30 * w0)),
            231 / (1024 * (power - 36 * w0)),
            -429 / (2048 * (power - 42 * w0)),
            6435 / (32768 * (power - 48 * w0)),
            -12155 / (65536 * (power - 54 * w0)),
            46189 / (262144 * (power - 60 * w0)),
            -88179 / (524288 * (power - 66 * w0)),
            676039 / (4194304 * (power - 72 * w0)),
            -1300075 / (8388608 * (power - 78 * w0)),
            5014575 / (33554432 * (power - 84 * w0)),
            -9694845 / (67108864 * (power - 90 * w0)),
            300540195 / (268435456 * (power - 96 * w0)),
        ]
    )


@autodoc
def flat_wcdm_pade_coefficients(w0=-1, zpower=0):
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
    coeffs = flat_wcdm_taylor_expansion(w0, zpower=zpower)
    p, q = pade(coeffs, len(coeffs) // 2, len(coeffs) // 2)
    return p, q


@autodoc
def indefinite_integral(z, Om0, w0=-1, zpower=0):
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
    x = (1 - Om0) / Om0 * (1 + z) ** (3 * w0)
    p, q = flat_wcdm_pade_coefficients(w0=w0, zpower=zpower)
    normalization = -2 / Om0**0.5 / (1 + z) ** (0.5 - zpower)
    return normalization * xp.polyval(p, x) / xp.polyval(q, x)


@autodoc
def analytic_integral(z, Om0, w0=-1, zpower=0):
    r"""
    Compute an integral of the form
    :math:`\int_{{\infty}}^z \frac{{(1+z)^k}}{{E(z)}}` using

    .. math::

        f(z; \Omega_{{m, 0}}, w_0) = \int_{{\infty}}^{{z}}
        \frac{{dz' (1 + z')^k}}{{E(z'; \Omega_{{m, 0}}, w_0)}}
        = \frac{{-2\Phi(z; \Omega_{{m, 0}}, w_0) (1+z)^k }}
        {{\sqrt{{\Omega_{{m, 0}}(1 + z)}}}}.

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
    kwargs = dict(Om0=Om0, w0=w0, zpower=zpower)
    return indefinite_integral(z, **kwargs) - indefinite_integral(0, **kwargs)
