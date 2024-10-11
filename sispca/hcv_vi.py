### Rewriting the HCV model in PyTorch and scvi-tools v1.1.2
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

class HCV(VAE):
	"""Add HSIC loss to the latent representation in the VAE model.

	Parameters:
		hsic_scale: float, scalar multiplier of the HSIC penalty
		n_latent_desired: int, number of dimensions for the desired latent space
		n_latent_qc: int, number of dimensions for the batch space
		predict_qc: bool, whether to predict the batch qc signal from the latent space
		n_qcs: dimension of the batch qc signal to predict.
	"""

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
		target_key_and_type: list[dict] | None = None,
		**model_kwargs
	):
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
		# self.n_latent = sum(n_latent_sub)
		# self.latent_distribution = "normal"
		# self.dispersion = "gene"

		# HSIC parameters
		self.hsic_scale = hsic_scale

		# latent space parameters
		self.n_latent_sub = n_latent_sub # list of n_latent for each subspace
		self.n_subspace = len(n_latent_sub) # number of subspaces

		self.predict_target_from_latent_sub = predict_target_from_latent_sub
		self.target_key_and_type = target_key_and_type

		# if providing with target information
		if self.predict_target_from_latent_sub:
			assert target_key_and_type is not None, "Must provide target_key_and_type if predict_target_from_latent_sub is True."

			self._n_target = len(target_key_and_type)

			# store the target register key
			self._target_register_key = [
				f"target_{target_key_and_type[i]['value_type']}_{target_key_and_type[i]['key']}"
				for i in range(self._n_target)
			]

			if self._n_target == (self.n_subspace - 1):
				print(
					f"{self._n_target} supervision keys provided for {self.n_subspace} subspaces. " \
					"The last subspace will be unsupervised."
		   		)
			elif self._n_target != self.n_subspace:
				raise ValueError(
					f"{self.n_subspace} subspaces need at least {self.n_subspace - 1} supervision keys."
				)

			# initialize the target predictors
			self.target_predictors = torch.nn.ModuleList()
			for i in range(self._n_target):
				if target_key_and_type[i]["value_type"] == "categorical": # classification
					n_label_target = target_key_and_type[i]["n_label_target"]
					target_predictor = torch.nn.Sequential(
						torch.nn.Linear(n_latent_sub[i], 25, bias=True),
						torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
						torch.nn.ReLU(),
						torch.nn.Linear(25, n_label_target, bias=False),
						torch.nn.Softmax(dim=-1)
					)
				else: # regression
					n_dim_target = target_key_and_type[i]["n_dim_target"]
					target_predictor = torch.nn.Sequential(
						torch.nn.Linear(n_latent_sub[i], 25, bias=True),
						torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
						torch.nn.ReLU(),
						torch.nn.Linear(25, n_dim_target, bias=False)
					)

				self.target_predictors.append(target_predictor)

			# initialize the loss function for each target prediction
			self.loss_target = torch.nn.ModuleList()
			for i in range(self._n_target):
				if target_key_and_type[i]["value_type"] == "categorical":
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
		qz, z = self.z_encoder(x) # concatenated latent space
		z = qz.loc if use_posterior_mean else z # use posterior mean if specified

		# seperate the latent space into subspaces
		z_sub = torch.split(z, self.n_latent_sub, dim=-1)

		# run prediction for each subspace
		predictions = []
		for i in range(self._n_target):
			predictions.append(self.target_predictors[i](z_sub[i]))

		return predictions

	def prediction_loss(self, labelled_dataset: dict[str, torch.Tensor]):
		"""Calculate the mean squared error loss for the QC signal prediction."""
		x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)

		# run prediction
		predictions = self.predict(x)

		loss = {}
		# loop through each target
		for i in range(self._n_target):
			# extract the target variable
			target_key = self.target_key_and_type[i]["key"]
			value_type = self.target_key_and_type[i]["value_type"]
			register_key = f"target_{value_type}_{target_key}"
			y = labelled_dataset[register_key]

			# reshape the target variable if categorical
			if value_type == "categorical":
				y = y.view(-1).long()

			# calculate the loss
			loss.update({target_key: self.loss_target[i](predictions[i], y)})

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
		target_key_and_type: list[dict] | None = None,
		**model_kwargs,
	):
		super().__init__(adata, n_latent = sum(n_latent_sub))
		hcv_model_kwargs = dict(model_kwargs)

		self.predict_target_from_latent_sub = predict_target_from_latent_sub
		self.n_latent_sub = n_latent_sub
		self.n_subspace = len(n_latent_sub) # number of subspaces
		self.target_key_and_type = target_key_and_type

		self.module = self._module_cls(
			n_input=self.summary_stats.n_vars,
			n_layers=n_layers,
			dropout_rate=dropout_rate,
			n_hidden=n_hidden,
			hsic_scale=hsic_scale,
			n_latent_sub=n_latent_sub,
			predict_target_from_latent_sub=predict_target_from_latent_sub,
			target_key_and_type=target_key_and_type,
			gene_likelihood=gene_likelihood,
			**hcv_model_kwargs,
		)

		self.module.minified_data_type = self.minified_data_type

		self.unsupervised_history_ = None
		self.semisupervised_history_ = None

		self._model_summary_string = (
			f"HCVI Model with the following params: \n"
			f"hsic_scale: {hsic_scale}, predict_target: {predict_target_from_latent_sub}, "
			f"n_subspace: {self.n_subspace}, n_latent_sub: {self.n_latent_sub}, "
			f"n_layers: {n_layers}, dropout_rate: {dropout_rate}, gene_likelihood: {gene_likelihood}"
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
		"""Trains the model.
		"""
		if True: # train using SCVI
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
		else: # train using SCANVI
			if max_epochs is None:
				max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

				if self.was_pretrained:
					max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))

			logger.info(f"Training for {max_epochs} epochs.")

			plan_kwargs = {} if plan_kwargs is None else plan_kwargs
			datasplitter_kwargs = datasplitter_kwargs or {}

			data_splitter = SemiSupervisedDataSplitter(
				adata_manager=self.adata_manager,
				train_size=train_size,
				validation_size=validation_size,
				shuffle_set_split=shuffle_set_split,
				n_samples_per_label=n_samples_per_label,
				batch_size=batch_size,
				**datasplitter_kwargs,
			)

			training_plan = SemiSupervisedTrainingPlan(self.module, self.n_labels, **plan_kwargs)
			if "callbacks" in trainer_kwargs.keys():
				trainer_kwargs["callbacks"] + [sampler_callback]
			else:
				trainer_kwargs["callbacks"] = sampler_callback

			runner = TrainRunner(
				self,
				training_plan=training_plan,
				data_splitter=data_splitter,
				max_epochs=max_epochs,
				accelerator=accelerator,
				devices=devices,
				check_val_every_n_epoch=check_val_every_n_epoch,
				**trainer_kwargs,
			)
			return runner()

	# TODO: add target information to fields
	@classmethod
	@setup_anndata_dsp.dedent
	def setup_anndata(
		cls,
		adata: AnnData,
		layer: str | None = None,
		target_key_and_type: list[dict] | None = None,
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
		if target_key_and_type is not None:
			n_targets = len(target_key_and_type)
			for dict_target in target_key_and_type:
				# extract information on the target variable
				key = dict_target["key"]
				field_type = dict_target["field_type"]
				value_type = dict_target["value_type"]
				registry_key = f"target_{value_type}_{key}"

				# sanity check
				assert field_type in ["obs", "obsm"], "field_type must be either 'obs' or 'obsm."
				assert value_type in ["categorical", "continuous"], "value_type must be either 'categorical' or 'continuous'."

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

