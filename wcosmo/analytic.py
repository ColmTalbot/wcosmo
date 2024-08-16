import numpy as np
from scipy.special import beta

from .utils import autodoc

xp = np

__all__ = ["indefinite_integral"]


@autodoc
def indefinite_integral(z, Om0, w0=-1, zpower=0):
    r"""
    Compute the integral of :math:`(1+z)^k / E(z)` as described in
    https://doi.org/10.4236/jhepgc.2021.73057.
    We extend it to include other integrands via powers of :math:`1 + z`.

    .. math::

        I(z; \Omega_{{m, 0}}, w_0) = \frac{{(1 + z)^{{k - \frac{{1}}{{2}}}}}}
        {{\Omega_{{m, 0}}^{{\frac{{1}}{{2}}}} 3 w_0}}
        B(\frac{{-1 + 2k}}{{6 w_0}}, 1)
        _{{2}}F_{{1}}(
            \frac{{1}}{{2}}, \frac{{-1 + 2k}}{{6 w_0}},
            \frac{{-1 + 2k + 6 w_0}}{{6 w_0}},
        x).

    Here the argument for the hypergeometric function is

    .. math::

        x = \left(\frac{{\Omega_{{m, 0}} - 1}}{{\Omega_{{m, 0}}}}\right)
        (1 + z)^{{3 w_0}}.

    There is a special case when :math:`w_0 = 0` where the hypergeometric
    function is not defined. In this case, we return the integral

    .. math::

        I(z; \Omega_{{m, 0}}, w_0) = \frac{{(1 + z)^{{k - \frac{{1}}{{2}}}}}}
        {{k - \frac{{1}}{{2}}}}.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    I: array_like
        The indefinite integral of :math:`(1+z)^k / E(z)`

    Notes
    -----
    The underlying hypergeometric function is not natively implemented in
    :code:`JAX` or :code:`cupy` so this will not be fully compatible.
    For demonstration, we can use the :code:`scipy` implementation with
    :code:`JAX` but this will not allow differentiation or GPU acceleration.

    This has been discussed in :code:`cupy` and may be implemented in the
    future (https://github.com/cupy/cupy/issues/8274).
    """
    value = (1 + z) ** (zpower - 1 / 2)
    try:
        x = ((Om0 - 1) / Om0) * (1 + z) ** (3 * w0)
        aa = 1 / 2
        bb = (zpower - 1 / 2) / (3 * w0)
        cc = bb + 1
        if "jax" in xp.__name__:
            from ._hyp2f1_jax import hyp2f1
        else:
            from scipy.special import hyp2f1
        values = hyp2f1(aa, bb, cc, x)
        normalization = beta(bb, cc - bb) * values / (3 * w0 * Om0**0.5)
    except ZeroDivisionError:
        normalization = 1 / (zpower - 1 / 2)
    result = value * normalization
    if isinstance(x, float) and not isinstance(result, float):
        result = result.item()
    return result
