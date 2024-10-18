r"""
Utilities based on the Taylor expansion of the functions of the form
:math:`\frac{(1 + z)^k}{E(z)}` for flat wCDM cosmology.
These functions and their integral are estimated using the Pade approximation.
"""

import numpy as np
from scipy.linalg import toeplitz

from .utils import autodoc

xp = np

__all__ = [
    "flat_wcdm_pade_coefficients",
    "flat_wcdm_taylor_expansion",
    "indefinite_integral_pade",
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


def _binomial_coefficients():
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
    what = (w0 + abs(w0)) / 2
    denominator = 1 - 2 * zpower + 6 * abs(w0) * xp.arange(0, 17) + 3 * what
    return _binomial_coefficients() / denominator


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
    what = (w0 + abs(w0)) / 2
    sign = xp.sign(w0)
    abs_sign = abs(sign)
    gamma = (Om0 ** (sign - abs_sign) * (1 - Om0) ** (-sign - abs_sign)) ** 0.25
    normalization = -2 * gamma * (1 + z) ** (zpower - 0.5 - 3 * what / 2)
    # jax will evaluate all the branches of the pade integral and so we
    # need to manually catch zero division errors.
    try:
        x = (Om0 / (1 - Om0)) ** sign * (1 + z) ** (-3 * abs(w0))
    except ZeroDivisionError:
        return z * 0.0
    p, q = flat_wcdm_pade_coefficients(w0=w0, zpower=zpower)
    return normalization * (xp.polyval(p, x) / xp.polyval(q, x)) ** abs_sign
