from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import toeplitz
from jax.scipy.special import beta
from plum import dispatch
from scipy.special import hyp2f1 as sc_hyp2f1

from ..taylor import pade


@jax.jit
def hyp2f1(a, b, c, z):
    a, b, c, z = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(z)

    # Promote the input to inexact (float/complex).
    # Note that jnp.result_type() accounts for the enable_x64 flag.
    z = z.astype(jnp.result_type(float, z.dtype))

    _scipy_hyp2f1 = lambda a, b, c, z: sc_hyp2f1(a, b, c, z).astype(
        z.dtype
    )  # noqa E731

    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(a.shape, b.shape, c.shape, z.shape), dtype=z.dtype
    )

    return jax.pure_callback(_scipy_hyp2f1, result_shape_dtype, a, b, c, z)


hyp2f1 = jax.custom_jvp(hyp2f1)


@hyp2f1.defjvp
def hyp2f1_jvp(primals, tangents):
    a, b, c, z = primals
    _, _, _, z_dot = tangents
    dhyp2f1_dz = a * b / c * hyp2f1(a + 1, b + 1, c + 1, z)
    return hyp2f1(a, b, c, z), z_dot * dhyp2f1_dz


@partial(jax.jit, static_argnums=(4,))
@dispatch
def indefinite_integral(z: jax.Array, Om0=None, w0=-1, zpower=0, method="pade"):
    with np.errstate(divide="ignore"):
        return jax.lax.cond(
            (Om0 == 0) | (Om0 == 1) | (w0 == 0),
            indefinite_integral_one_component,
            partial(_indefinite_integral_two_component, method=method),
            z,
            Om0,
            w0,
            zpower,
        )


@jax.jit
@dispatch
def indefinite_integral_hypergeometric(z: jax.Array, Om0, w0=-1, zpower=0):
    from ..analytic import _indefinite_integral_hypergeometric

    return _indefinite_integral_hypergeometric(
        z, Om0, w0, zpower, hyp2f1=hyp2f1, beta=beta
    )


@partial(jax.jit, static_argnums=(4,))
def _indefinite_integral_two_component(z, Om0, w0=-1, zpower=0, method="pade"):
    from ..taylor import indefinite_integral_pade

    return jax.lax.cond(
        method == "pade",
        indefinite_integral_pade,
        indefinite_integral_hypergeometric,
        z,
        Om0,
        w0,
        zpower,
    )


@jax.jit
@dispatch
def indefinite_integral_one_component(z, Om0, w0=-1, zpower=0):
    power = zpower - 1 / 2 - (3 * w0 / 2) * (Om0 == 0)
    return jax.lax.cond(
        power != 0,
        lambda z: (1 + z) ** power / power,
        lambda z: jnp.log1p(z),
        z,
    )


@jax.jit
@pade.dispatch
def pade(an: jax.Array, m, n=None):
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
    an = jnp.asarray(an)
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
    Akj = jnp.eye(N + 1, n + 1, dtype=an.dtype)
    Bkj = toeplitz(jnp.r_[0.0, -an[:-1]], jnp.zeros(m))
    Ckj = jnp.hstack((Akj, Bkj))
    pq = jnp.linalg.solve(Ckj, an)
    p = pq[: n + 1]
    q = jnp.r_[1.0, pq[n + 1 :]]
    return p[::-1], q[::-1]
