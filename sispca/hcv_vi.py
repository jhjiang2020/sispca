### Rewriting the HCV model in PyTorch and scvi-tools v1.2.0
### Adapted from https://github.com/romain-lopez/HCV/blob/master/scVI/scVIgenqc.py
from typing import List, Literal, Optional, Union
import warnings
import inspect

import torch
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal
from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput
from scvi.module import VAE
from scvi.model import SCVI
from scvi.model._utils import get_max_epochs_heuristic

from anndata import AnnData
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

from scvi.data import AnnDataManager
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    BaseAnnDataField,
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)

from sispca.utils import hsic_gaussian

class SupervisionAnnData():
    """Custom data class for supervision from an AnnData object."""
    def __init__(self, target_key, field_type, target_type, target_n_dim = 1):
        """
        Args:
            target_key (str): key name of the target variable.
            field_type (str): field type of the AnnData object where the target variable is stored.
            target_type (str): 'continuous' or 'categorical'. The type of the target variable.
            target_n_dim (int): The number of dimensions of the target variable.
                - If target_type == 'categorical', number of labels to predict.
                - If target_type == 'continuous', number of output dimensions in a regression model.
        """
        self.target_key = target_key # key name of the target variable
        self.field_type = field_type # field type of the AnnData object where the target variable is stored
        self.target_type = target_type # type of the target variable
        self.target_n_dim = target_n_dim # number of dimensions of the target variable

        self._sanity_check()

    def _sanity_check(self):
        assert self.field_type in ["obs", "obsm"], "field_type must be either 'obs' or 'obsm."
        assert self.target_type in ['continuous', 'categorical'], \
            "Currently only support 'continuous' or 'categorical' targets."


class HCV(VAE):
    """Add HSIC loss to the latent representation in the VAE model."""

    def __init__(
        self,
        n_input: int,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        n_hidden: int = 128,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "normal",
        hsic_scale: float = 1.0,
        n_latent_sub = [10, 10],
        predict_target_from_latent_sub: bool = False,
        target_supervision_list: list[SupervisionAnnData] | None = None,
        **model_kwargs
    ):
        """
        Args:
            hsic_scale (float): Scalar multiplier of the HSIC penalty.
            n_latent_sub (list of int): Number of latent dimensions for each subspace.
            predict_target_from_latent_sub (bool): Whether to predict target variables from the latent space.
            target_supervision_list (list of SupervisionAnnData): List of SupervisionAnnData objects if
                predict_target_from_latent_sub is True.
        """
        # remove redundant keys from model_kwargs
        for _keys in ['n_latent', 'latent_distribution', 'dispersion']:
            if _keys in model_kwargs.keys():
                warnings.warn(f"Removing redundant argument '{_keys}={model_kwargs[_keys]}'.")
                del model_kwargs[_keys]

        latent_distribution = "normal"
        dispersion = "gene"

        # initialize the VAE model
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=sum(n_latent_sub),
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            dispersion=dispersion,
            **model_kwargs
        )
        self.gene_likelihood = gene_likelihood

        # # after initialization we will have:
        # self.n_latent = sum(n_latent_sub)
        # self.latent_distribution = "normal"
        # self.dispersion = "gene"

        # HSIC parameters
        self.hsic_scale = hsic_scale

        # latent space parameters
        self.n_latent_sub = n_latent_sub # list of n_latent for each subspace
        self.n_subspace = len(n_latent_sub) # number of subspaces
        self.predict_target_from_latent_sub = predict_target_from_latent_sub
        self.target_supervision_list = target_supervision_list

        # if providing with target information
        if self.predict_target_from_latent_sub:
            assert target_supervision_list is not None, "Missing target_supervision_list. No supervision provided."

            self.n_target = len(target_supervision_list) # number of target variables

            # store the target register key
            self._target_register_key = [
                f"target_{s_i.target_type}_{s_i.target_key}"
                for s_i in self.target_supervision_list
            ]

            if self.n_target == (self.n_subspace - 1):
                print(
                    f"{self.n_target} supervision keys provided for {self.n_subspace} subspaces. " \
                    "The last subspace will be unsupervised."
                   )
            elif self.n_target != self.n_subspace:
                raise ValueError(
                    f"{self.n_subspace} subspaces supplied with {self.n_target} supervision keys."
                )

            # initialize the target predictors
            self.target_predictors = torch.nn.ModuleList()
            for i, s_i in enumerate(self.target_supervision_list):
                if s_i.target_type == "categorical": # classification
                    n_label_target = s_i.target_n_dim # number of labels to predict
                    target_predictor = torch.nn.Sequential(
                        torch.nn.Linear(n_latent_sub[i], 25, bias=True),
                        torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        torch.nn.ReLU(),
                        torch.nn.Linear(25, n_label_target, bias=False),
                        torch.nn.Softmax(dim=-1)
                    )
                else: # regression
                    n_dim_target = s_i.target_n_dim # number of output dimensions
                    target_predictor = torch.nn.Sequential(
                        torch.nn.Linear(n_latent_sub[i], 25, bias=True),
                        torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        torch.nn.ReLU(),
                        torch.nn.Linear(25, n_dim_target, bias=False)
                    )

                self.target_predictors.append(target_predictor)

            # initialize the loss function for each target prediction
            self.loss_target = torch.nn.ModuleList()
            for s_i in self.target_supervision_list:
                if s_i.target_type == "categorical":
                    self.loss_target.append(torch.nn.CrossEntropyLoss(reduction='sum'))
                else:
                    self.loss_target.append(torch.nn.MSELoss(reduction='sum'))

    def predict(self, x, use_posterior_mean=False):
        """Forward passing through the encoder and run prediction.

        Args:
            x (tensor): input tensor
            use_posterior_mean (bool): whether to use the posterior mean of the
                latent distribution for prediction
        """
        qz, z = self.z_encoder(x) # concatenated latent space, (n_batch_size, n_latent)
        z = qz.loc if use_posterior_mean else z # use posterior mean if specified

        # seperate the latent space into subspaces
        z_sub = torch.split(z, self.n_latent_sub, dim=-1) # list of (n_batch_size, n_latent_sub[i])

        # run prediction for each subspace
        predictions = []
        for i in range(self.n_target):
            predictions.append(self.target_predictors[i](z_sub[i]))

        return predictions

    def prediction_loss(self, labelled_dataset: dict[str, torch.Tensor]):
        """Calculate the mean squared error loss for the QC signal prediction."""
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_batch_size, n_vars)

        # run prediction
        predictions = self.predict(x) # list of n_target predictions

        loss = {}
        # loop through each target
        for s_i, pred_i, loss_fn in zip(
            self.target_supervision_list, predictions, self.loss_target
        ):
            # extract the target variable
            target_key = s_i.target_key
            value_type = s_i.target_type
            register_key = f"target_{value_type}_{target_key}"
            y = labelled_dataset[register_key]

            # reshape the target variable if categorical
            if value_type == "categorical":
                y = y.view(-1).long()

            # calculate the loss
            loss.update({target_key: loss_fn(pred_i, y)})

        return loss

    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Run the generative process."""
        if self.gene_likelihood != 'normal':
            return super().generative(
                z,
                library,
                batch_index,
                cont_covs=cont_covs,
                cat_covs=cat_covs,
                size_factor=size_factor,
                y=y,
                transform_batch=transform_batch,
            )

        # use Normal as the generative distribution for x
        _, px_r, px_rate, _ = self.decoder(
            self.dispersion,
            z,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        px = Normal(px_rate, px_r)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            'px': px,
            'pl': pl,
            'pz': pz,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
    ):
        # calculate the original VAE loss
        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=-1
        )
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # add the prediction loss
        if self.predict_target_from_latent_sub:
            prediction_loss = self.prediction_loss(tensors) # list of n_target losses
            prediction_loss_sum = sum(prediction_loss.values())
        else:
            prediction_loss = {}
            prediction_loss_sum = 0.0

        # for logging
        prediction_loss_dict = {"prediction_loss_sum": prediction_loss_sum} | prediction_loss

        # add the HSIC loss
        # seperate the latent space into subspaces
        z = inference_outputs["z"]
        z_sub = torch.split(z, self.n_latent_sub, dim=-1)

        hsic_loss = 0.0
        if self.hsic_scale > 0:
            for i in range(self.n_subspace):
                for j in range(i + 1, self.n_subspace):
                    # calculate the HSIC loss
                    z_1 = z_sub[i]
                    z_2 = z_sub[j]
                    hsic_loss += self.hsic_scale * hsic_gaussian(z_1, z_2)

        loss = torch.mean(reconst_loss + prediction_loss_sum + weighted_kl_local + hsic_loss)

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }
        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local, kl_global=hsic_loss,
            extra_metrics=prediction_loss_dict
        )


class HCVI(SCVI):
    """Training a HSIC-regulated VAE model with scvi-tools.
    """

    _module_cls = HCV

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "normal",
        hsic_scale: float = 1.0,
        n_latent_sub = [10, 10],
        predict_target_from_latent_sub: bool = False,
        target_supervision_list: list[SupervisionAnnData] | None = None,
        **model_kwargs,
    ):
        """
        Args:
            hsic_scale (float): Scalar multiplier of the HSIC penalty.
            n_latent_sub (list of int): Number of latent dimensions for each subspace.
            predict_target_from_latent_sub (bool): Whether to predict target variables from the latent space.
            target_supervision_list (list of SupervisionAnnData): List of SupervisionAnnData objects if
                predict_target_from_latent_sub is True.
        """
        super().__init__(adata, n_latent = sum(n_latent_sub))
        hcv_model_kwargs = dict(model_kwargs)

        # subspace parameters
        self.predict_target_from_latent_sub = predict_target_from_latent_sub
        self.n_latent_sub = n_latent_sub
        self.n_subspace = len(n_latent_sub) # number of subspaces
        self.target_key_and_type = target_supervision_list
        self.n_target = len(target_supervision_list) if target_supervision_list is not None else 0

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            n_hidden=n_hidden,
            hsic_scale=hsic_scale,
            n_latent_sub=n_latent_sub,
            predict_target_from_latent_sub=predict_target_from_latent_sub,
            target_supervision_list=target_supervision_list,
            gene_likelihood=gene_likelihood,
            **hcv_model_kwargs,
        )

        self.module.minified_data_type = self.minified_data_type

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None

        self._model_summary_string = (
            f"HCVI model with the following parameters: \n"
            "------------------------------------------\n"
            f"*(architecture): n_layers = {n_layers}, dropout_rate = {dropout_rate}, gene_likelihood = {gene_likelihood}\n"
            f"*(subspace) n_subspace = {self.n_subspace}, n_latent_sub = {self.n_latent_sub}, hsic_scale = {hsic_scale}\n"
            f"*(supervision) predict_target = {self.predict_target_from_latent_sub}, n_target = {self.n_target}"
        )
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        n_samples_per_label: float | None = None,
        check_val_every_n_epoch: int | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Trains the model using SCVI."""
        super().train(
            max_epochs = max_epochs,
            check_val_every_n_epoch = check_val_every_n_epoch,
            train_size = train_size,
            validation_size = validation_size,
            shuffle_set_split = shuffle_set_split,
            batch_size = batch_size,
            accelerator = accelerator,
            devices = devices,
            datasplitter_kwargs = datasplitter_kwargs,
                **trainer_kwargs
        )

    # TODO: add target information to fields
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        target_supervision_list: list[SupervisionAnnData] | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        # add target fields
        if target_supervision_list is not None:
            n_targets = len(target_supervision_list)
            for s_i in target_supervision_list:
                # extract information on the target variable
                key = s_i.target_key
                field_type = s_i.field_type
                value_type = s_i.target_type
                registry_key = f"target_{value_type}_{key}"

                # add to the fields
                if field_type == "obs":
                    if value_type == "categorical":
                        anndata_fields.append(CategoricalObsField(registry_key, key))
                    else:
                        anndata_fields.append(NumericalObsField(registry_key, key))
                else:
                    if value_type == "categorical":
                        raise ValueError("Currently, only 'obs' fields can be categorical.")
                    else:
                        anndata_fields.append(ObsmField(registry_key, key))

        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

