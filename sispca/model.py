import warnings
import itertools
import lightning as L
import scipy
import numpy as np
from sklearn import cluster
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sispca.utils import normalize_col, tr_cov, gaussian_kernel, delta_kernel, hsic_gaussian, hsic_linear, gram_schmidt
from sispca.data import SISPCADataset

class LossHistoryCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.loss_history = {
            'train_loss_step': [],
            'train_loss_epoch': [],
            'reg_loss_step': [],
            'reg_loss_epoch': [],
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.loss_history['train_loss_step'].append(outputs['train_loss'].item())
        self.loss_history['reg_loss_step'].append(outputs['reg_loss'].item())

    def on_train_epoch_end(self, trainer, pl_module):
        # At the end of each epoch, extract the logged metrics
        metrics = trainer.callback_metrics
        self.loss_history['train_loss_epoch'].append(metrics['train_loss_epoch'].item())
        self.loss_history['reg_loss_epoch'].append(metrics['reg_loss_epoch'].item())


class PCA(L.LightningModule):
    """The unsupervised PCA model."""
    def __init__(self, n_feature, n_latent: int = 10, lr: float = 1.0):
        """
        Args:
            n_feature (int): Number of features of the data.
            n_latent (int): Number of principal components to compute.
            lr (float): Learning rate for updating U.
        """
        super().__init__()
        self.n_latent = n_latent # n_pc
        self.n_feature = n_feature # n_feature
        self._data = None # (n_sample, n_feature)
        self.automatic_optimization = False # solve PCA using eigen decomposition
        self.lr = lr # learning rate for updating U

        # configure U of size (n_feature, n_pc), the parameter to learn
        U_init = gram_schmidt(nn.init.uniform_(torch.zeros(self.n_feature, self.n_latent)))
        self.U = nn.parameter.Parameter(U_init, requires_grad = False)

    def forward(self, x):
        """Project data to the latent PC space.

        Args:
            x (n_sample, n_feature): Data to project.
        """
        return x @ self.U # (n_sample, n_latent)

    def loss(self, x):
        """Calculate the loss function."""
        z = self(x) # (n_sample, n_latent)
        # for PCA, we want to maximize the covariance in the latent space
        loss = - tr_cov(z) / z.shape[0]
        return loss

    def training_step(self, batch, batch_idx):
        """Training step for PCA."""
        lr = self.lr

        # update U using eigendecomposition
        _, _, V = torch.svd_lowrank(batch, self.n_latent)
        U_new = (1 - lr) * self.U + lr * V

        # orthonormalize U
        if lr < 1:
            U_new = gram_schmidt(U_new)

        self.U.copy_(U_new) # (n_feature, num_pc)

        loss = self.loss(batch)
        self.log('training_loss', loss)

        return loss

    def configure_optimizers(self):
        pass


class SISPCA(PCA):
    """The supervised independent subspace PCA (sisPCA) model."""

    def __init__(self,
            dataset: SISPCADataset,
            n_latent_sub = [10, 10],
            lambda_contrast = 1.0,
            kernel_subspace = 'linear',
            solver = 'eig',
        ):
        """
        Args:
            dataset (SISPCADataset): The dataset with supervision for sisPCA.
            n_latent_sub (list of int): Number of PCs in each subspace.
            lambda_contrast (float): Weight for the contrastive loss.
            kernel_subspace (str): The HSIC kernel for the contrastive loss between subspaces.
                Can be 'linear' or 'gaussian'.
            solver (str): Can be 'eig' or 'gd'.
        """
        super().__init__(n_feature = dataset.n_feature, n_latent=sum(n_latent_sub))
        self.n_sample = dataset.n_sample
        self.n_latent_sub = n_latent_sub # list of n_latent for each subspace
        self.n_subspace = len(n_latent_sub) # number of subspaces
        self.lambda_contrast = lambda_contrast # weight for the contrastive loss

        # HSIC kernel for the latent space
        # if linear, solve the HSIC using iterative eigendecomposition
        assert kernel_subspace in ['linear', 'gaussian'], "Invalid kernel_subspace."
        self.kernel_subspace = kernel_subspace

        # specify the solver
        assert solver in ['eig', 'gd']

        if kernel_subspace == 'gaussian':
            solver = 'gd' # only gradient descent solver for Gaussian kernel

        if self.n_feature > 2895 and solver == 'eig':
            warnings.warn(
                "torch.linalg.eigh has a numeric issue with more than 2895 features. " \
                "Switching to the gradient descent solver."
            )
            self.solver = 'gd'
        else:
            self.solver = solver

        # default learning rate
        self.lr = 1.0

        # if using gradient descent, set the automatic optimization
        self.automatic_optimization = (self.solver == 'gd')
        U_init = gram_schmidt(nn.init.uniform_(torch.zeros(self.n_feature, self.n_latent)))
        self.U = nn.parameter.Parameter(U_init, requires_grad = (self.solver == 'gd'))

        self._dataset = dataset

        # add the supervised target variables info
        self.n_target = dataset.n_target # number of target variables
        self.target_name_list = dataset.target_name_list.copy() # list of target names
        self.target_kernel_list = dataset.target_kernel_list.copy() # list of target kernels

        if self.n_target == (self.n_subspace - 1):
            print(
                f"{self.n_target} supervision variables provided for {self.n_subspace} subspaces. " \
                "The last subspace will be unsupervised."
            )
            self.target_kernel_list.append(torch.eye(dataset.n_sample))

        elif self.n_target != self.n_subspace:
            raise ValueError(
                f"{self.n_subspace} subspaces supplied with {self.n_target} supervision targets."
            )

    def _extract_batch_inputs(self, batch):
        """Helper function to extract batched inputs for training."""

        # extract the data
        index = batch['index'] # torch tensor
        x = batch['x']

        # center within the batch
        x = normalize_col(x, center = True, scale = False)

        # extract target kernels
        target_kernel_batch = [K[index, :][:, index] for K in self.target_kernel_list]

        return index, x, target_kernel_batch

    def loss(self, batch):
        """Calculate the loss function."""
        # extract batched inputs
        index, x, target_kernel_list = self._extract_batch_inputs(batch)

        # extract the latent representation
        z = self(x) # (batch_size, n_latent)
        n_sample = z.shape[0]
        H = torch.eye(n_sample) - 1/n_sample # centering matrix

        # center the representation
        z = normalize_col(z, center = True, scale = (self.kernel_subspace == 'gaussian'))

        # split z into subspaces
        z_sub = torch.split(z, self.n_latent_sub, dim = -1)

        recon_loss_list = []
        # the supervised PCA loss: maximize HSIC in each subspace
        for _z_i, _target_kernel in zip(z_sub, target_kernel_list):
            # center the subset target kernel
            _K = _target_kernel @ H

            if self.kernel_subspace == 'gaussian':
                recon_loss_i = - torch.trace(gaussian_kernel(_z_i) @ H @ _K) / (n_sample - 1) ** 2
            else:
                recon_loss_i = - torch.trace(_z_i.T @ _K @ _z_i) / (n_sample - 1) ** 2

            recon_loss_list.append(recon_loss_i)

        reg_loss = torch.tensor(0.0)
        # the contrastive loss: minimize HSIC between subspaces
        if self.lambda_contrast > 0:
            for i in range(self.n_subspace):
                for j in range(i + 1, self.n_subspace):
                    z_1 = z_sub[i]
                    z_2 = z_sub[j]

                    if self.kernel_subspace == 'gaussian':
                        reg_loss += self.lambda_contrast * hsic_gaussian(z_1, z_2)
                    else:
                        reg_loss += self.lambda_contrast * hsic_linear(z_1, z_2)

        return {
            'recon_loss': recon_loss_list,
            'reg_loss': reg_loss,
        }

    def _get_U_subspace_list(self):
        """Split U into subspaces."""
        U_sub = torch.split(self.U, self.n_latent_sub, dim = -1)
        return list(U_sub)

    def _reorthogonalize_U(self):
        """Reorthogonalize U per subspace."""
        with torch.no_grad():
            U_new_sub_list = [gram_schmidt(U_sub) for U_sub in self._get_U_subspace_list()]
            self.U.copy_(torch.concat(U_new_sub_list, dim = 1))

    def configure_optimizers(self):
        # if using gradient descent, return the optimizer
        if self.solver == 'gd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss_epoch"}
        else:
            return None

    def training_step(self, batch, batch_idx):
        """Training step for sisPCA."""
        # calculate and log the loss
        loss_dict = self.loss(batch)
        reg_loss = loss_dict['reg_loss'] # HSIC contrastive loss
        loss_sum = sum(loss_dict['recon_loss']) + reg_loss

        self.log('train_loss_step', loss_sum, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss_sum, on_step=False, on_epoch=True)
        self.log('reg_loss_step', reg_loss, on_step=True, on_epoch=False)
        self.log('reg_loss_epoch', reg_loss, on_step=False, on_epoch=True)

        # self.log_dict({
        #     'train_loss': loss_sum,
        #     'reg_loss': loss_dict['reg_loss']
        # }, on_step=False, on_epoch=True)

        # if using gradient descent, return only the loss
        if self.solver == 'gd':
            return {'loss': loss_sum, 'train_loss': loss_sum, 'reg_loss': reg_loss}

        # else, use iterative eigendecomposition
        # get the learning rate scheduler
        # lr = 0.1
        lr = self.lr

        # extract batched inputs
        index, x, target_kernel_list = self._extract_batch_inputs(batch)

        # split U into subspaces
        U_sub_list = self._get_U_subspace_list()

        # calculate representation and kernel in each subspace
        z_sub_list = [x @ U_sub for U_sub in U_sub_list]
        K_z_list = [_z_i @ _z_i.T for _z_i in z_sub_list]

        K_y_hat_list = []
        # update U using eigendecomposition for each subspace
        for i, _K_target in enumerate(target_kernel_list):
            K_y_hat = _K_target - self.lambda_contrast/2 * sum(
                K_z_list[:i] + K_z_list[(i+1):]
            )
            # centering the kernel
            K_y_hat = K_y_hat - K_y_hat.mean(dim=0).unsqueeze(0)
            K_y_hat = K_y_hat - K_y_hat.mean(dim=1).unsqueeze(1)
            K_y_hat_list.append(K_y_hat)

            # run eigendecomposition to get the U update
            mat = x.T @ K_y_hat @ x
            mat += 1e-3 * torch.eye(mat.shape[0])

            # # torch.svd_lowrank not stable with rank-one matrix
            # mat += 1e-1 * torch.eye(mat.shape[0])
            # _, _, U_sub_new = torch.svd_lowrank(mat, self.n_latent_sub[i] + 10)
            # U_sub_new = U_sub_new[:, :self.n_latent_sub[i]]

            # # scipy.sparse.linalg.svds similar behavior with torch.svd_lowrank
            # _, _, vh = scipy.sparse.linalg.svds(
            #     mat.numpy(), k = self.n_latent_sub[i], which='LM', return_singular_vectors='vh',
            # )
            # U_sub_new = torch.from_numpy(vh.T).float()
            # _, _, U_sub_new = torch.linalg.svd(mat)
            # U_sub_new = U_sub_new[:, :self.n_latent_sub[i]]

            # torch.linalg.eigh returns ascending eigenvalues
            _, V = torch.linalg.eigh(mat)
            U_sub_new = V[:, -self.n_latent_sub[i]:]
            U_sub_new = torch.flip(U_sub_new, [1]) # largest comes first

            # # torch.linalg.eig doesn't sort the eigenvectors
            # _lambda, V = torch.linalg.eig(mat)
            # idx = torch.argsort(torch.real(_lambda), descending = True)
            # V = torch.real(V)[:, idx]
            # U_sub_new = V[:, :self.n_latent_sub[i]]

            # update the subspace representation
            z_sub_new = x @ U_sub_new

            # update U, subspace representation and the kernel
            U_sub_list[i] = U_sub_new
            z_sub_list[i] = z_sub_new
            K_z_list[i] = z_sub_new @ z_sub_new.T

        # update U
        U_new = (1 - lr) * self.U + lr * torch.concat(U_sub_list, dim = 1)
        self.U.copy_(U_new)

        if lr < 1:
            self._reorthogonalize_U()

        return {'loss': loss_sum, 'train_loss': loss_sum, 'reg_loss': reg_loss}

    def on_train_epoch_end(self):
        """At the end of each epoch, reorthogonalize U (for gradient descent)."""
        if self.solver == 'gd':
            # self._reorthogonalize_U() # orthonormalize U per subspace
            # # need to enforce orthogonality of U across subspaces
            with torch.no_grad():
                self.U.copy_(gram_schmidt(self.U))

    def fit(
        self,
        batch_size = -1,
        shuffle = True,
        max_epochs = 100,
        early_stopping_patience = 5,
        accelerator = 'cpu',
        deterministic = True,
        lr = None,
        **kwargs
    ):
        """Fit the sisPCA model."""
        if batch_size == -1:
            batch_size = self.n_sample

        # update the learning rate for the eigendecomposition
        if lr is not None:
            if self.solver == 'eig':
                assert (lr <= 1.0), "lr should be less than 1 for eigendecomposition."
                if lr < 1:
                    warnings.warn("Untested partial update. Proceed with caution.")
        else:
            if self.solver == 'gd':
                lr = 100
            else:
                lr = 1.0
                # TODO: incremental SVD update with minibatch
                # lr = batch_size / self.n_sample
        self.lr = lr

        loss_history_callback = LossHistoryCallback()
        self._trainer = L.Trainer(
            accelerator = accelerator, max_epochs = max_epochs,
            enable_checkpointing=False,
            logger=False,
            callbacks = [
                loss_history_callback
            ] + (
                [L.pytorch.callbacks.EarlyStopping(monitor='train_loss_epoch', patience=early_stopping_patience)]
                if early_stopping_patience is not None else []
            ),
            deterministic = deterministic,
            **kwargs
        )
        train_dataloader = DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle)
        self._trainer.fit(self, train_dataloader)
        self.history = loss_history_callback.loss_history

    def get_latent_representation(self):
        """Get the latent representation."""
        return self(self._dataset.x).detach().numpy()


class SISPCAAuto():
    """A wrapper class for sisPCA for fitting with multiple lambda_contrast values.

    Adapted from contrastive PCA. Abid et al. (2018).
    https://github.com/abidlabs/contrastive/blob/master/contrastive/__init__.py
    """
    def __init__(
        self, dataset, n_latent_sub, max_log_lambda_c, n_lambda, n_lambda_clu,
        target_sdim_list = None, **kwargs
    ):
        """
        Args:
            dataset (SISPCADataset): The dataset with supervision for sisPCA.
            n_latent_sub (list of int): Number of PCs in each subspace.
            max_log_lambda_c (float): Maximum log10 lambda_contrast value.
            n_lambda (int): Number of lambda_contrast values to search.
            n_lambda_clu (int): Number of lambda_contrast clusters to return.
            target_sdim_list (list of int): Effective dimension of each target space.
                If None, will re-compute the rank of the target kernel.
            **kwargs: Additional keyword arguments for sisPCA
        """
        self.dataset = dataset
        self.n_latent_sub = n_latent_sub
        self.n_subspace = len(n_latent_sub)
        self.kwargs = kwargs

        # specify the lambda_contrast values to search
        self.max_log_lambda_c = max_log_lambda_c # maximum log10 lambda_contrast value
        self.n_lambda = n_lambda # number of lambda_contrast values to search
        self.n_lambda_clu = n_lambda_clu # number of lambda_contrast clusters to return
        self.lambda_contrast_list = np.concatenate(([0],np.logspace(-1, max_log_lambda_c, n_lambda - 1)))
        self.models = []

        # calculate the effective dimension of each target space
        if target_sdim_list is None:
            target_sdim_list = [torch.linalg.matrix_rank(K).item() for K in dataset.target_kernel_list]
        else:
            assert len(target_sdim_list) == dataset.n_target, \
                "The target_sdim_list should have the same length as the number of targets."
            target_sdim_list = target_sdim_list.copy()

        # add the dimension of the unsupervised subspace if necessary
        if len(target_sdim_list) == (self.n_subspace - 1):
            target_sdim_list.append(dataset.n_sample)
        elif len(target_sdim_list) != self.n_subspace:
            raise ValueError(
                f"{self.n_subspace} subspaces supplied with {len(target_sdim_list)} supervision targets."
            )

        self.target_sdim_list = [min(_sdim, _n_latent) for _sdim, _n_latent in zip(target_sdim_list, n_latent_sub)]

    def find_best_update_order(self):
        lambda_contrast = self.lambda_contrast_list[-1] # the largest lambda_contrast

        # generate indices for all possible permutations
        index_permutations = itertools.permutations(range(self.n_subspace))
        best_init_order = None
        best_dataset_perm = None
        current_loss = np.inf

        # screen over all possible subspace orders
        for subspace_order in index_permutations:
            dataset_perm = SISPCADataset(
                data = self.dataset.x,
                target_supervision_list = [
                    self.dataset.target_supervision_list[i] for i in subspace_order
                ]
            )
            model_perm = SISPCA(
                dataset = dataset_perm,
                n_latent_sub = self.n_latent_sub,
                lambda_contrast = lambda_contrast,
                **self.kwargs
            )
            model_perm.fit(batch_size=512, shuffle=True, max_epochs=5, lr=1, early_stopping_patience=None)
            loss = model_perm.history['train_loss_epoch'][-1]

            if loss < current_loss:
                current_loss = loss
                best_init_order = subspace_order
                best_dataset_perm = dataset_perm

        print(f"Best subspace order: {best_init_order}")

        # update the dataset and subspace order
        self.dataset = best_dataset_perm
        self.n_latent_sub = [self.n_latent_sub[i] for i in best_init_order]

    def fit(self, **kwargs):
        self.models = []
        self.final_loss = []
        for lambda_contrast in self.lambda_contrast_list:
            model = SISPCA(
                dataset = self.dataset,
                n_latent_sub = self.n_latent_sub,
                lambda_contrast = lambda_contrast,
                **self.kwargs
            )

            # initialize U from the previous model
            if len(self.models) > 0:
                with torch.no_grad():
                    model.U.copy_(self.models[-1].U.clone().detach())

            model.fit(**kwargs)
            self.models.append(model)

            # log the final loss
            self.final_loss.append({
                'lambda_contrast': lambda_contrast,
                'train_loss': model.history['train_loss_epoch'][-1],
                'reg_loss': model.history['reg_loss_epoch'][-1],
                'recon_loss': model.history['train_loss_epoch'][-1] - model.history['reg_loss_epoch'][-1]
            })

    def create_affinity_matrix(self, affinity_metric = 'determinant'):
        """Compute the pairwise affinity matrix between results of varying lambda_contrast.

        affinity_metric: str, the metric to compute the pairwise subspace affinity on the Grassmann manifold.
            Can be 'determinant' (Fubini-Study), 'asimov', or 'geodesic'.
            See https://uqpyproject.readthedocs.io/en/stable/utilities/distances/grassmann_distances.html.
        """
        from math import pi

        n_lambda = self.n_lambda

        assert affinity_metric in ['determinant', 'asimov', 'geodesic']
        if affinity_metric == 'asimov':
            affinity = pi/4 * np.identity(n_lambda)
        else:
            affinity = 0.5 * np.identity(n_lambda) # it gets doubled

        # extract the latent representation per subspace for each model
        solutions = [
            np.split(z, self.n_latent_sub[:-1], axis = -1)
            for z in [m.get_latent_representation() for m in self.models]
        ] # list of len n_lambda, each with list of len n_subspace

        # qr decomposition to orthonormalize the representation
        solutions = [
            [np.linalg.qr(s)[0] # orthonormalize the representation
             for s in sol]
            for sol in solutions
        ]

        # the effective dimension of each subspace
        sdim_list = self.target_sdim_list

        # calculate pairwise affinity as the determinant of cross-covariance matrix
        for i in range(n_lambda):
            for j in range(i + 1, n_lambda):
                # separate and compare subspaces independently
                for sdim, q0, q1 in zip(sdim_list, solutions[i], solutions[j]):
                    # compute the principal angles between the subspaces
                    u, s, v = np.linalg.svd(q0.T.dot(q1))
                    if affinity_metric == 'asimov':
                        angle = np.arccos(s[sdim - 1]) # the largest principal angle
                        affinity[i,j] += pi/2 - angle
                    elif affinity_metric == 'determinant':
                        # aka the Fubini-Study distance
                        affinity[i,j] += np.prod(s[:sdim])
                    else:
                        dist = np.linalg.norm(np.arccos(s[:sdim]))
                        affinity[i,j] += np.exp(- dist**2)
                affinity[i, j] /= len(self.n_latent_sub) # average over subspaces

        # make the affinity matrix symmetric
        affinity = affinity + affinity.T
        self.affinity_matrix = np.nan_to_num(affinity)

    def find_spectral_lambdas(self):
        """Aggregate solutions using spectral clustering."""
        affinity = self.affinity_matrix
        lambda_contrast_list = self.lambda_contrast_list

        # cluster the solutions based on the affinity matrix
        spectral = cluster.SpectralClustering(n_clusters=self.n_lambda_clu + 1, affinity='precomputed')
        spectral.fit(affinity)

        # log the exemplar lambda_contrast values for each cluster
        self.lambda_contrast_clusters = spectral.labels_

    def get_latent_representation(self, affinity_metric = 'determinant', exemplar_method='smallest'):

        # compute the affinity matrix and cluster the solutions
        self.create_affinity_matrix()
        self.find_spectral_lambdas()

        labels = self.lambda_contrast_clusters
        affinity = self.affinity_matrix

        # always include the sPCA result
        results = {
            'lambda_0.000 (sPCA)': self.models[0].get_latent_representation() # the sPCA solution
        }

        # keep the exemplar sisPCA solution for each cluster
        for i in range(self.n_lambda_clu + 1):
            idx = np.where(labels == i)[0]
            if not(0 in idx): # ignore the sPCA solution
                if exemplar_method=='smallest':
                    exemplar_idx = int(np.min(idx))
                elif exemplar_method=='medoid':
                    affinity_submatrix = affinity[idx][:, idx]
                    sum_affinities = np.sum(affinity_submatrix, axis=0)
                    exemplar_idx = idx[np.argmax(sum_affinities)]
                else:
                    raise ValueError("Invalid specification of exemplar method")

                results[
                    f'lambda_{self.lambda_contrast_list[exemplar_idx]:.3f}'
                ] = self.models[exemplar_idx].get_latent_representation()

        return results

