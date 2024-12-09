import cupy as cp
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from cupyx.scipy.linalg import toeplitz
from plum import dispatch

from ..taylor import pade

try:
    cp.cuda.Device()
except CUDARuntimeError:
    raise ImportError


@pade.dispatch
def pade(an: cp.ndarray, m, n=None):
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
    Akj = cp.eye(N + 1, n + 1, dtype=an.dtype)
    Bkj = toeplitz(cp.r_[0.0, -an[:-1]], cp.zeros(m))
    Ckj = cp.hstack((Akj, Bkj))
    pq = cp.linalg.solve(Ckj, an)
    p = pq[: n + 1]
    q = cp.r_[1.0, pq[n + 1 :]]
    return p[::-1], q[::-1]
