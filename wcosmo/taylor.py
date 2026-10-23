r"""
Utilities based on the Taylor expansion of the functions of the form
:math:`\frac{(1 + z)^k}{E(z)}` for flat wCDM cosmology.
These functions and their integral are estimated using a continued fraction,
equivalent to a Pade approximation.
"""

import numpy as np
from plum import Function, dispatch
from scipy.interpolate import pade as sc_pade

from .utils import array_namespace, autodoc

__all__ = [
    "definite_integral_pade",
    "flat_wcdm_continued_fraction_coefficients",
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
            300540195 / 2147483648,
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

    The distance integrals in this module use the equivalent, but more stable
    and accurate, continued fraction in
    :func:`flat_wcdm_continued_fraction_coefficients` instead.

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
def flat_wcdm_continued_fraction_coefficients(w0=-1, zpower=0, *, n_terms=32, xp=np):
    r"""
    Coefficients of a continued fraction approximation to :math:`\Phi`.

    Solving for the coefficients in :func:`flat_wcdm_pade_coefficients` is
    ill-conditioned and in single precision can introduce spurious poles in
    the physical domain. Instead, we note that

    .. math::

        \Phi(x; w_0; k) = \frac{{1}}{{A}}
        {{}}_2F_1\left(\frac{{1}}{{2}}, b; b + 1; -x\right)
        = \frac{{(1 + x)^{{\frac{{1}}{{2}}}}}}{{A}}
        {{}}_2F_1\left(b + \frac{{1}}{{2}}, 1; b + 1; -x\right)

    with :math:`A = 1 - 2k + 3\hat{{w}}_0`, :math:`b = A / 6|w_0|`, and
    :math:`\hat{{w}}_0 = \max(w_0, 0)`.
    The final hypergeometric function has a closed-form continued fraction
    expansion due to Gauss with coefficients :math:`c_j` that are positive
    for :math:`b > 0`

    .. math::

        {{}}_2F_1\left(b + \frac{{1}}{{2}}, 1; b + 1; -x\right) =
        \cfrac{{1}}{{1 + \cfrac{{c_1 x}}{{1 + \cfrac{{c_2 x}}{{1 + \dots}}}}}}

    Truncating after :math:`2n` terms is equivalent to the :math:`[n/n]`
    Pade approximant to :math:`A \Phi (1 + x)^{{-\frac{{1}}{{2}}}}`.
    Since the continued fraction is evaluated directly, rather than converted
    to polynomials, we can use more terms than
    :func:`flat_wcdm_pade_coefficients` without loss of precision.

    The truncation error scales as :math:`\rho^{{2n}}` with
    :math:`\rho = (\sqrt{{1 + x}} - 1) / (\sqrt{{1 + x}} + 1)` and the
    largest :math:`x` is reached at :math:`z = 0`,
    :math:`x = (1 - \Omega_{{m, 0}}) / \Omega_{{m, 0}}` for :math:`w_0 < 0`,
    so more terms are needed for small :math:`\Omega_{{m, 0}}`.

    Parameters
    ----------
    {w0}
    {zpower}
    n_terms: int
        The number of terms in the continued fraction, default=32.

    Returns
    -------
    array_like
        The continued fraction coefficients :math:`c_j`.
    """
    what = (w0 + abs(w0)) / 2
    bb = (1 - 2 * zpower + 3 * what) / (6 * abs(w0))
    coeffs = list()
    for jj in range(1, n_terms + 1):
        ii = jj // 2
        if jj % 2:
            coeff = (bb + ii + 0.5) * (bb + ii) / ((bb + 2 * ii) * (bb + 2 * ii + 1))
        else:
            coeff = ii * (ii - 0.5) / ((bb + 2 * ii - 1) * (bb + 2 * ii))
        coeffs.append(xp.asarray(coeff))
    return xp.stack(coeffs)


def _n_terms(x, xp):
    r"""
    The number of continued fraction terms to use for the precision of
    :code:`x`. In double precision, 64 terms give a relative error in the
    distance integrals :math:`\lesssim 10^{{-12}}` for
    :math:`\Omega_{{m, 0}} \geq 0.05`, rising to :math:`\sim 10^{{-7}}` at
    :math:`\Omega_{{m, 0}} = 0.02` and :math:`\sim 10^{{-5}}` at
    :math:`\Omega_{{m, 0}} = 0.01`. In lower precision 32 terms are enough.
    """
    if xp.asarray(x).dtype == xp.float64:
        return 64
    return 32


def _continued_fraction_and_difference(coeffs, x0, x1, dx, root_0, root_1):
    r"""
    Evaluate the continued fraction defined in
    :func:`flat_wcdm_continued_fraction_coefficients` at :code:`x0` and
    :code:`x1` and the difference between the two given :math:`x_1 - x_0`
    using the backward recurrence

    .. math::

        f_j(x) = \frac{{1}}{{1 + c_j x f_{{j + 1}}(x)}}

        \Delta f_j = -c_j f_j(x_0) f_j(x_1)
        \left(\Delta x f_{{j + 1}}(x_0) + x_1 \Delta f_{{j + 1}}\right).

    This avoids explicitly forming :math:`f(x_1) - f(x_0)`.
    An error in :math:`\Delta f_{{j + 1}}` is multiplied by
    :math:`\prod_{{i \leq j}} c_i x_1 f_i(x_0) f_i(x_1)` by the end of the
    recurrence, this is less than :math:`0.4` for :math:`x_1 \leq x_0 \leq 1000`.
    The equivalent split
    :math:`\Delta x f_{{j + 1}}(x_1) + x_0 \Delta f_{{j + 1}}` replaces
    :math:`x_1` by :math:`x_0` in this product, which amplifies rounding
    errors by up to :math:`\sim 10^{{2}}` for :math:`x_0 = 9`
    (:math:`\Omega_{{m, 0}} = 0.1`) and :math:`\sim 10^{{15}}` for
    :math:`x_0 = 19` (:math:`\Omega_{{m, 0}} = 0.05`).

    The coefficients tend to :math:`\frac{{1}}{{4}}` and we start the
    recurrence from the value of the continued fraction with all remaining
    coefficients equal to :math:`\frac{{1}}{{4}}`,
    :math:`f(x) = 2 / (1 + \sqrt{{1 + x}})`, rather than :math:`1`. This
    reduces the truncation error by one to two orders of magnitude at no extra
    cost as :code:`root_0` :math:`= \sqrt{{1 + x_0}}` and :code:`root_1`
    :math:`= \sqrt{{1 + x_1}}` are needed anyway.
    """
    value_0 = 2 / (1 + root_0)
    value_1 = 2 / (1 + root_1)
    difference = -value_0 * value_1 * dx / (2 * (root_0 + root_1))
    for coeff in coeffs[::-1]:
        delta = dx * value_0 + x1 * difference
        value_0 = 1 / (1 + coeff * x0 * value_0)
        value_1 = 1 / (1 + coeff * x1 * value_1)
        difference = -coeff * delta * value_0 * value_1
    return value_0, value_1, difference


def _phi_and_difference(x0, x1, dx, w0, zpower, xp):
    r"""
    Compute :math:`\Phi(x_1)` and :math:`\Phi(x_1) - \Phi(x_0)` given
    :math:`x_1 - x_0` using the continued fraction from
    :func:`flat_wcdm_continued_fraction_coefficients`.
    """
    coeffs = flat_wcdm_continued_fraction_coefficients(
        w0=w0, zpower=zpower, n_terms=_n_terms(x1, xp), xp=xp
    )
    root_0 = (1 + x0) ** 0.5
    root_1 = (1 + x1) ** 0.5
    _, value, difference = _continued_fraction_and_difference(
        coeffs, x0, x1, dx, root_0, root_1
    )
    what = (w0 + abs(w0)) / 2
    norm = 1 - 2 * zpower + 3 * what
    phi = root_1 * value / norm
    difference = (dx / (root_0 + root_1) * value + root_0 * difference) / norm
    return phi, difference


def _phi(x, w0, zpower, xp):
    r"""
    Compute :math:`\Phi(x)` as in :func:`_phi_and_difference`, without the
    difference.
    """
    coeffs = flat_wcdm_continued_fraction_coefficients(
        w0=w0, zpower=zpower, n_terms=_n_terms(x, xp), xp=xp
    )
    root = (1 + x) ** 0.5
    value = 2 / (1 + root)
    for coeff in coeffs[::-1]:
        value = 1 / (1 + coeff * x * value)
    what = (w0 + abs(w0)) / 2
    return root * value / (1 - 2 * zpower + 3 * what)


@autodoc
def indefinite_integral_pade(z, Om0, w0=-1, zpower=0):
    r"""
    Compute the integral of :math:`(1+z)^k / E(z)` using a Pade approximation
    as described in arXiv:1111.6396. We extend it to include a variable dark
    energy equation of state, other integrands via powers of :math:`1 + z` and
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

    Adachi and Kasai use :math:`m=n=3`. We evaluate :math:`\Phi` with the
    equivalent continued fraction described in
    :func:`flat_wcdm_continued_fraction_coefficients`, using 64 terms
    (:math:`m=n=32`) in double precision and 32 terms otherwise.

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
    Om0 = xp.array(Om0)
    gamma = (Om0 ** (sign - abs_sign) * (1 - Om0) ** (-sign - abs_sign)) ** 0.25
    normalization = -2 * gamma * (1 + z) ** (zpower - 0.5 - 3 * what / 2)
    with np.errstate(divide="ignore"):
        x = (Om0 / (1 - Om0)) ** sign * (1 + z) ** (-3 * abs(w0))
    return normalization * _phi(x, w0, zpower, xp) ** abs_sign


@autodoc
def definite_integral_pade(z, Om0, w0=-1, zpower=0):
    r"""
    Compute the definite integral of :math:`(1+z)^k / E(z)` from :math:`0`
    to :math:`z` using the approximation in :func:`indefinite_integral_pade`.

    This is equivalent to :math:`I(z) - I(0)`, but we analytically
    rearrange the difference to avoid catastrophic cancellation at
    low redshift

    .. math::

        I(z) - I(0) = \frac{{-2}}{{\Omega_{{m, 0}}^{{\frac{{1}}{{2}}}}}}
        \left[\left((1 + z)^{{k - \frac{{1}}{{2}}}} - 1\right) \Phi(x_z)
        + \Phi(x_z) - \Phi(x_0)\right]

    where :math:`(1 + z)^{{k - \frac{{1}}{{2}}}} - 1` is computed with
    :code:`expm1` and :code:`log1p` and :math:`\Phi(x_z) - \Phi(x_0)` is
    computed alongside the continued fraction representation of :math:`\Phi`
    described in :func:`flat_wcdm_continued_fraction_coefficients`.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}
    {zpower}

    Returns
    -------
    I: array_like
        The integral of :math:`(1+z)^k / E(z)` from :math:`0` to :math:`z`
    """
    xp = array_namespace(z)
    what = (w0 + abs(w0)) / 2
    sign = xp.sign(xp.array(w0))
    abs_sign = abs(sign)
    Om0 = xp.array(Om0)
    gamma = (Om0 ** (sign - abs_sign) * (1 - Om0) ** (-sign - abs_sign)) ** 0.25
    log_zp1 = xp.log1p(z)
    with np.errstate(divide="ignore"):
        x0 = (Om0 / (1 - Om0)) ** sign
    x1 = x0 * xp.exp(-3 * abs(w0) * log_zp1)
    dx = x0 * xp.expm1(-3 * abs(w0) * log_zp1)
    phi, difference = _phi_and_difference(x0, x1, dx, w0, zpower, xp)
    normalization = xp.expm1((zpower - 0.5 - 3 * what / 2) * log_zp1)
    return -2 * gamma * (normalization * phi + difference)
