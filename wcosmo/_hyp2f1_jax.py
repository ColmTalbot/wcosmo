import jax
import jax.numpy as jnp
from scipy.special import hyp2f1 as sc_hyp2f1


def hyp2f1(a, b, c, z):
    a, b, c, z = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(z)

    # Promote the input to inexact (float/complex).
    # Note that jnp.result_type() accounts for the enable_x64 flag.
    z = z.astype(jnp.result_type(float, z.dtype))

    _scipy_hyp2f1 = lambda a, b, c, z: sc_hyp2f1(a, b, c, z).astype(z.dtype)

    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(a.shape, b.shape, c.shape, z.shape), dtype=z.dtype
    )

    return jax.pure_callback(
        _scipy_hyp2f1, result_shape_dtype, a, b, c, z, vectorized=True
    )


hyp2f1 = jax.custom_jvp(hyp2f1)


@hyp2f1.defjvp
def hyp2f1_jvp(primals, tangents):
    a, b, c, z = primals
    _, _, _, z_dot = tangents
    dhyp2f1_dz = a * b / c * hyp2f1(a + 1, b + 1, c + 1, z)
    return hyp2f1(a, b, c, z), z_dot * dhyp2f1_dz
