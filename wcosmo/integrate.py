from functools import partial

from plum import dispatch

from .analytic import indefinite_integral_hypergeometric
from .taylor import indefinite_integral_pade
from .utils import array_namespace, autodoc, maybe_jit

__all__ = ["analytic_integral", "indefinite_integral"]


@autodoc
def analytic_integral(z, Om0, w0=-1, zpower=0, method="pade"):
    r"""
    Compute an integral of the form
    :math:`\int_{{\infty}}^z \frac{{(1+z)^k}}{{E(z)}}` using

    .. math::

        f(z; \Omega_{{m, 0}}, w_0) = \int_{{\infty}}^{{z}}
        \frac{{dz' (1 + z')^k}}{{E(z'; \Omega_{{m, 0}}, w_0)}}
        = \frac{{-2\Phi(z; \Omega_{{m, 0}}, w_0) (1+z)^k }}
        {{\sqrt{{\Omega_{{m, 0}}(1 + z)}}}}.

    The integral is approximated using the Pade approximation and is up
    to a factor the term in the braces in (1.1) of Adachi and Kasai.

    Parameters
    ----------
    {z}
    {Om0}
    {w0}

    Returns
    -------
    integral: array_like
        The integral of :math:`(1 + z)^k / E(z)` from :math:`\infty` to :math:`z`

    Examples
    --------
    We can use this to calculate many cosmological distances. For example, the
    comoving distance :math:`d_{{C}}` is given by

    .. math::

        d_{{C}} = d_{{H}} \int_{{0}}^{{z}}
        \frac{{dz'}}{{E(z'; \Omega_{{m,0}}, w_0)}}

    In this case, :math:`k=0`

    >>> from wcosmo.taylor import analytic_integral
    >>> import wcosmo

    >>> analytic_integral(z=2,Om0=0.3,zpower=0) * wcosmo.hubble_distance(H0=70)
    5179.8621

    We can check this against the comoving volume given by ``astropy`` and see
    that it gives the same answer.

    >>> from astropy.cosmology import FlatLambdaCDM
    >>> ap_cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    >>> ap_cosmo.comoving_distance(2)
    5179.8621 Mpc

    This is how :func:`comoving_distance` is implemented in ``wcosmo``, which gives the same result

    >>> wcosmo.comoving_distance(2,H0=70,Om0=0.3)
    5179.8621

    As another example, the lookback time :math:`t_L` is given by

    .. math::

        t_{{L}} = t_{{H}}\int_{{0}}^{{z}} \frac{{dz'}}{{(1+z)E(z')}}

    where :math:`t_L` is the Hubble distance. In this case, :math:`k=-1`

    >>> analytic_integral(z=2,Om0=0.3,zpower=-1) * wcosmo.hubble_time(70) # Gyr
    10.240357
    We can again compare this to ``astropy`` and ``wcosmo``'s built-in commands

    >>> ap_cosmo.lookback_time(2)
    10.240357 Gyr
    >>> wcosmo.lookback_time(2,H0=70,Om0=0.3)
    10.240357

    Finally, we can calculate the absorption distance, :math:`X`, commonly used
    in absorption line spectroscopy. This is not a physical distance, but is still
    a very useful quantity.
    See the book "Cosmological Absorption Line Spectroscopy" by Christopher W.
    Churchill for a full discussion.

    .. math::

        X(z, \Omega_m) = \int_0^z \frac{{dz' (1+z')^2}}{{
            \sqrt{{\Omega_m(1+z')^3 + (1 - \Omega_m)}}}}

    Here, :math:`k=2` and there are no factors of the Hubble distance.

    >>> print(analytic_integral(z=2,Om0=0.3,zpower=2))
    4.36995
    >>> print(ap_cosmo.absorption_distance(2))
    4.36995
    >>> wcosmo.absorption_distance(2,Om0=0.3)
    4.36995
    """
    xp = array_namespace(z)
    kwargs = dict(Om0=Om0, w0=w0, zpower=zpower, method=method)
    return indefinite_integral(z, **kwargs) - indefinite_integral(
        xp.array(0.0), **kwargs
    )


@partial(maybe_jit, static_argnames=("method",))
@dispatch
def indefinite_integral(z, Om0=None, w0=-1, zpower=0, method="pade"):
    if (Om0 == 0) | (Om0 == 1) | (w0 == 0):
        return indefinite_integral_one_component(z, Om0, w0, zpower)
    elif method == "pade":
        return indefinite_integral_pade(z, Om0, w0, zpower)
    else:
        return indefinite_integral_hypergeometric(z, Om0, w0, zpower)


def indefinite_integral_one_component(z, Om0, w0=-1, zpower=0):
    xp = array_namespace(z)
    power = zpower - 1 / 2 - (3 * w0 / 2) * (Om0 == 0)
    if power != 0:
        return (1 + z) ** power / power
    else:
        return xp.log1p(z)
