def maybe_jit(func, *args, **kwargs):
    """
    A decorator to jit the function if using jax.

    This also allows aribtrary arguments to be passed through,
    e.g., to specify static arguments.

    This function is pretty useful and so might make it into
    `gwpopulation` regardless of cosmology.
    """
    from .wcosmo import xp
    if "jax" in xp.__name__:
        from jax import jit

        return jit(func, *args, **kwargs)
    return func
