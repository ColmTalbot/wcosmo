from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from jax.scipy.special import beta
from plum import dispatch
from scipy.special import hyp2f1 as sc_hyp2f1

from ..taylor import definite_integral_pade, indefinite_integral_pade


@jax.jit
def hyp2f1(a, b, c, z):
    a, b, c, z = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(z)

    # Promote the input to inexact (float/complex).
    # Note that jnp.result_type() accounts for the enable_x64 flag.
    z = z.astype(jnp.result_type(float, z.dtype))

    _scipy_hyp2f1 = lambda a, b, c, z: sc_hyp2f1(a, b, c, z).astype(z.dtype)  # noqa E731

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


def _is_zero(tangent):
    """
    Whether a tangent is known to be zero, either symbolically or because
    the corresponding primal is an integer.
    """
    return isinstance(tangent, SymbolicZero) or tangent.dtype == jax.dtypes.float0


def _with_integrand_jvp(func):
    """
    Wrap :code:`func(z, Om0, w0, zpower)`, an integral of
    :math:`(1 + z)^k / E(z)` with respect to redshift, with a custom JVP.

    The derivative with respect to :code:`z` is the integrand. The
    derivatives with respect to :code:`Om0` and :code:`w0` are computed in
    forward mode with unit tangents, so reverse-mode differentiation does
    not store the intermediate terms of the continued fraction in
    :mod:`wcosmo.taylor` for every redshift. Without this, the gradient of
    the differential comoving volume for :math:`10^7` redshifts in double
    precision needs more than 35 GB.

    The derivative with respect to :code:`zpower` is not implemented and is
    treated as zero.
    """
    wrapped = jax.custom_jvp(func)

    @partial(wrapped.defjvp, symbolic_zeros=True)
    def jvp(primals, tangents):
        from ..wcosmo import efunc

        z, Om0, w0, zpower = primals
        z_dot, Om0_dot, w0_dot, _ = tangents
        value = None
        terms = list()
        if not _is_zero(Om0_dot):
            value, d_Om0 = jax.jvp(
                lambda Om0: func(z, Om0, w0, zpower), (Om0,), (jnp.ones_like(Om0),)
            )
            terms.append(d_Om0 * Om0_dot)
        if not _is_zero(w0_dot):
            value, d_w0 = jax.jvp(
                lambda w0: func(z, Om0, w0, zpower), (w0,), (jnp.ones_like(w0),)
            )
            terms.append(d_w0 * w0_dot)
        if value is None:
            value = func(z, Om0, w0, zpower)
        if not _is_zero(z_dot):
            terms.append((1 + z) ** zpower / efunc(z, Om0, w0) * z_dot)
        return value, sum(terms, jnp.zeros_like(value))

    return wrapped


_definite_integral_pade = _with_integrand_jvp(definite_integral_pade)
_indefinite_integral_pade = _with_integrand_jvp(indefinite_integral_pade)


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
def definite_integral(z: jax.Array, Om0=None, w0=-1, zpower=0):
    with np.errstate(divide="ignore"):
        return jax.lax.cond(
            (Om0 == 0) | (Om0 == 1) | (w0 == 0),
            definite_integral_one_component,
            _definite_integral_pade,
            z,
            Om0,
            w0,
            zpower,
        )


@jax.jit
def definite_integral_one_component(z, Om0, w0=-1, zpower=0):
    power = zpower - 1 / 2 - (3 * w0 / 2) * (Om0 == 0)
    return jax.lax.cond(
        power != 0,
        lambda z: jnp.expm1(power * jnp.log1p(z)) / power,
        lambda z: jnp.log1p(z),
        z,
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
    if method != "pade":
        raise ValueError("wcosmo only supports pade integration with JAX")

    return _indefinite_integral_pade(z, Om0, w0, zpower)


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
