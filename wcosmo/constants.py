"""
Wrapper module to strip units from astropy constants is :code:`USE_UNITS` is
set to :code:`False`.

We include two additional constants that are useful for out default units:

- :code:`c_km_per_s` - the speed of light in km/s
- :code:`gyr_km_per_s_mpc` - the conversion factor from Gyr to km/s/Mpc
"""

from astropy import units

from .utils import convert_quantity_if_necessary, default_array_namespace

__all__ = ["USE_UNITS", "c_km_per_s", "gyr_km_per_s_mpc"]  # noqa F822


def __getattr__(name):
    value = _VALUES.get(name, None)
    if value is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if USE_UNITS:
        xp = default_array_namespace()
        value = convert_quantity_if_necessary(value, _UNITS[name], xp=xp)
    return value


def get(name, xp):
    if name not in _VALUES:
        raise KeyError(f"Invalid constant {name}")
    return convert_quantity_if_necessary(_VALUES[name], _UNITS.get(name, None), xp)


USE_UNITS = True
_VALUES = dict(
    c_km_per_s=299792.4580,
    gyr_km_per_s_mpc=977.7922216807891,
    steradian=1,
)
_UNITS = dict(
    c_km_per_s=units.Unit("km / s"),
    gyr_km_per_s_mpc=units.Unit("Gyr km / (s Mpc)"),
    steradian=units.Unit("sr"),
)
