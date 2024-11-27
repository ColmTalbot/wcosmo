"""
Helper functions that are not directly relevant to cosmology.
"""

import functools
import os

import numpy as np
from array_api_compat import array_namespace as _array_namespace
from array_api_compat import is_array_api_obj

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
    zpower="""zpower: array_like
        The power of the redshift dependence of the distance integrand
        (:math:`k`)""",
)

__all__ = [
    "autodoc",
    "disable_units",
    "enable_units",
    "method_autodoc",
    "maybe_jit",
    "strip_units",
]

USE_UNITS = True


def autodoc(func):
    """
    Simple decorator to mark that a docstring needs formatting
    """
    func.__doc__ = func.__doc__.format(**_cosmology_docstrings_)
    return func


def disable_units():
    """
    Disable the use of astropy units throughout the package
    """
    _set_units(False)


def enable_units():
    """
    Enable the use of astropy units throughout the package
    """
    _set_units(True)


def _set_units(val):
    """
    Set the use of astropy units throughout the package
    """
    global USE_UNITS

    from . import astropy, constants

    constants.USE_UNITS = val
    astropy.USE_UNITS = val
    USE_UNITS = val


def method_autodoc(alt=None):
    """
    Simple decorator to mark that a docstring needs formatting.
    This will strip the class level attributes of :code:`FlatwCDM`
    from the dosctring and allow a docstring to be taken from
    another function.
    """

    def new_wrapper(func):
        def _strip_wcdm_parameters(doc):
            """
            Stripy the FlatwCDM parameters from the docstring and remove
            the entire parameters section if it is empty after.
            """
            for key in ["H0", "Om0", "w0"]:
                doc = doc.replace(_cosmology_docstrings_[key], "")
            doc = doc.replace(
                "Parameters\n    ----------\n    \n\n    Returns", "Returns"
            )
            return doc

        if alt is not None:
            doc = alt.__doc__
        else:
            doc = func.__doc__
        doc = _strip_wcdm_parameters(doc)
        func.__doc__ = doc
        return func

    return new_wrapper


def maybe_jit(func, *jit_args, **jit_kwargs):
    """
    A decorator to jit the function if using jax.

    This also allows arbitrary arguments to be passed through,
    e.g., to specify static arguments.

    This function is pretty useful and so might make it into
    :code:`gwpopulation` regardless of cosmology.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        xp = kwargs.pop("xp", np)
        if len(args) > 0 and is_array_api_obj(args[0]):
            xp = array_namespace(args[0])

        if "jax" in xp.__name__:
            from jax import jit

            return jit(func, *jit_args, **jit_kwargs)(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapped


def strip_units(value):
    """
    Strip units from a value if they are present
    also make sure that the result is not a
    numpy.float64.
    """
    if hasattr(value, "unit"):
        value = value.value
    if isinstance(value, np.float64):
        value = value.item()
    return value


def convert_quantity_if_necessary(arg, unit=None, xp=np):
    """
    Helper function to convert between :code:`astropy` and :code:`unxt`
    quantities and non-unitful values.

    The order of precedence is as follows:

    - If using :code:`jax.numpy` as the backend, the input is an
      :code:`astropy` or :code:`unxt` quantity or :code:`unit` is specified,
      convert to a :code:`unxt` quantity with the provided unit.
    - If using :code:`jax.numpy` as the backend, the input is not a quantiy
      and no unit is provided, return the input.
    - If a unit and an :code:`astropy` quantity are provided, convert the
      input to an :code:`astropy` quantity with the provided unit
    - If a unit is provided, convert the input to an :code:`astropy`
      quantity with the provided unit.
    - Else return the input as is.

    Parameters
    ==========
    arg: Union[astropy.units.Quantity, unxt.Quantity, array_like]
        The array to convert
    unit: Optional[astropy.units.Unit, str]
        The unit to convert to

    Returns
    =======
    Union[astropy.units.Quantity, unxt.Quantity, array_like]
        The converted array
    """
    from astropy.units import Quantity as _Quantity

    if not USE_UNITS:
        return strip_units(arg)

    if xp.__name__ in ["jax.numpy", "quaxed.numpy"] and (
        isinstance(arg, _Quantity) or unit is not None
    ):
        from unxt import Quantity

        if unit is None:
            return Quantity.from_(arg)
        return Quantity.from_(arg, unit)
    elif unit is not None and isinstance(arg, _Quantity):
        return arg.to(unit)
    elif unit is not None:
        return _Quantity(arg, unit)
    return arg


def array_namespace(obj):
    """
    Get the array namespace from an array object
    """
    if is_array_api_obj(obj):
        return _array_namespace(obj)
    else:
        return default_array_namespace()


def default_array_namespace():
    match os.environ.get("WCOSMO_ARRAY_API", "numpy"):
        case "numpy":
            import numpy as np
        case "jax":
            import jax.numpy as jnp

            return jnp
        case "cupy":
            import cupy as cp

            return cp
        case _:
            print(f"Unknown array API: {os.environ['WCOSMO_ARRAY_API']} for wcosmo")
            return np
    return np
