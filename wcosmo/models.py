import numpy as xp
from gwpopulation.experimental.jax import NonCachingModel
from gwpopulation.models.redshift import _Redshift

from .wcosmo import FlatwCDM, available, z_at_value

__all__ = ["CosmoModel", "PowerLawRedshift"]


class CosmologyMixin:

    def cosmology_variables(self, parameters):
        return {key: parameters[key] for key in self.cosmology_names}

    def cosmology(self, parameters):
        if isinstance(self._cosmo, FlatwCDM):
            return self._cosmo
        else:
            return self._cosmo(**self.cosmology_variables(parameters))

    def __init__(self, cosmo_model="Planck15"):

        self.cosmo_model = cosmo_model
        if self.cosmo_model == "FlatwCDM":
            self.cosmology_names = ["H0", "Om0", "w0"]
        elif self.cosmo_model == "FlatLambdaCDM":
            self.cosmology_names = ["H0", "Om0"]
        else:
            self.cosmology_names = []
        self._cosmo = available[cosmo_model]

    def detector_frame_to_source_frame(self, data):

        cosmo = self.cosmology(self.parameters)

        samples = dict()
        samples["redshift"] = z_at_value(
            cosmo.luminosity_distance,
            data["luminosity_distance"],
        )
        jacobian = cosmo.dDLdz(samples["redshift"])
        for key in data:
            if key.endswith("_detector"):
                samples[key.strip("_detector")] = data[key] / (1 + samples["redshift"])
                jacobian *= 1 + samples["redshift"]
            elif key != "luminosity_distance":
                samples[key] = data[key]
        return samples, jacobian

    def luminosity_distance_to_redshift_jacobian(self, redshift):

        """
        Calculates the luminosity distance to redshift jacobian

        Parameters
        ==========
        redshift: array-like

        Returns
        =======
        luminosity_distance: array-like
        """
        return self.cosmology(self.parameters).dDLdz(redshift)


class CosmoModel(NonCachingModel, CosmologyMixin):
    """
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """

    def __init__(self, model_functions, cosmo_model="Planck15"):
        super().__init__(model_functions=model_functions)
        CosmologyMixin.__init__(self, cosmo_model=cosmo_model)

    def prob(self, data, **kwargs):
        """
        Compute the total population probability for the provided data given
        the keyword arguments.

        Parameters
        ==========
        data: dict
            Dictionary containing the points at which to evaluate the
            population model.
        kwargs: dict
            The population parameters. These cannot include any of
            :code:`["dataset", "data", "self", "cls"]` unless the
            :code:`variable_names` attribute is available for the relevant
            model.
        """

        samples_in_source, jacobian = self.detector_frame_to_source_frame(data)
        probability = super().prob(samples_in_source, **kwargs)
        return probability / jacobian


class _Redshift(CosmologyMixin, _Redshift):
    """
    Base class for models which include a term like dVc/dz / (1 + z)
    """

    _variable_names = None

    @property
    def variable_names(self):
        vars = self.cosmology_names.copy()
        if self._variable_names is not None:
            vars += self._variable_names
        return vars

    def __init__(self, cosmo_model="Planck15", z_max=2.3):

        super().__init__(cosmo_model=cosmo_model)
        self.z_max = z_max
        self.zs = xp.linspace(1e-6, z_max, 2500)

    def __call__(self, dataset, **kwargs):
        return self.probability(dataset=dataset, **kwargs)

    def normalisation(self, parameters):
        normalisation_data = self.differential_spacetime_volume(
            dict(redshift=self.zs), bounds=True, **parameters
        )
        norm = xp.trapz(normalisation_data, self.zs)
        return norm

    def probability(self, dataset, **parameters):
        differential_volume = self.differential_spacetime_volume(
            dataset, bounds=True, **parameters
        )
        norm = self.normalisation(parameters)

        return differential_volume / norm

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

    def dvc_dz(self, redshift, **parameters):
        return (
            4
            * xp.pi
            * self.cosmology(parameters).differential_comoving_volume(redshift)
        )

    def differential_spacetime_volume(self, dataset, bounds=False, **parameters):
        r"""
        Compute the differential spacetime volume.

        .. math::
            d\mathcal{V} = \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        dataset: dict
            Dictionary containing entry "redshift"
        parameters: dict
            Dictionary of parameters
        Returns
        -------
        differential_volume: (float, array-like)
            Differential spacetime volume
        """
        differential_volume = (
            self.psi_of_z(redshift=dataset["redshift"], **parameters)
            / (1 + dataset["redshift"])
            * self.dvc_dz(redshift=dataset["redshift"], **parameters)
        )
        if bounds:
            differential_volume *= dataset["redshift"] <= self.z_max
        return differential_volume


class PowerLawRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= (1 + z)^\lambda

    Parameters
    ----------
    lamb: float
        The spectral index.
    """

    _variable_names = ["lamb"]

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]
