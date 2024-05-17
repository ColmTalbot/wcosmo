"""
Helper functions that are not directly relevant to cosmology.
"""

import numpy as xp
from scipy.linalg import toeplitz


_cosmology_docstrings_ = dict(
    z="""z: array_like
        Redshift""",
    Om0="""Om0: array_like
        The matter density fraction""",
    w0="""w0: array_like
        The (constant) equation of state parameter for dark energy""",
    H0="""H0: array_like
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
    m1="""m1: array_like
        The primary mass in the source frame""",
    m2="""m2: array_like
        The secondary mass in the source frame""",
    m1z="""m1z: array_like
        The primary mass in the detector frame""",
    m2z="""m2z: array_like
        The secondary mass in the detector frame""",
    dL="""dL: array_like
        The luminosity distance in Mpc""",
)


def _autodoc(func, strip_wcdm=False, alt=None):
    """
    Simple decorator to mark that a docstring needs formatting
    """
    func.__doc__ = func.__doc__.format(**_cosmology_docstrings_)
    return func


def _method_autodoc(strip_wcdm=False, alt=None):
    """
    Simple decorator to mark that a docstring needs formatting.
    This will strip the class level attributes of :code:`FlatwCDM`
    from the dosctring and allow a docstring to be taken from
    another function.
    """

    def new_wrapper(func):
        def _strip_wcdm_parameters(doc):
            for key in ["H0", "Om0", "w0"]:
                doc = doc.replace(_cosmology_docstrings_[key], "")
            return doc

        if alt is not None:
            doc = alt.__doc__
        else:
            doc = func.__doc__
        if strip_wcdm:
            doc = _strip_wcdm_parameters(doc)
        func.__doc__ = doc
        return func

    return new_wrapper


def maybe_jit(func, *args, **kwargs):
    """
    A decorator to jit the function if using jax.

    This also allows aribtrary arguments to be passed through,
    e.g., to specify static arguments.

    This function is pretty useful and so might make it into
    :code:`gwpopulation` regardless of cosmology.
    """
    from .wcosmo import xp

    if "jax" in xp.__name__:
        from jax import jit

        return jit(func, *args, **kwargs)
    return func


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

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import pade
    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
    >>> p, q = pade(e_exp, 2)

    >>> e_exp.reverse()
    >>> e_poly = np.poly1d(e_exp)

    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``

    >>> e_poly(1)
    2.7166666666666668

    >>> p(1)/q(1)
    2.7179487179487181

    Notes
    -----
    This code has been slightly edited from the numpy implementation to:

    - Use xp instead of np to support multiple backends
    - Direclty use the fact that part of the matrix is Toeplitz

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
