import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset
from sispca.utils import normalize_col, delta_kernel

class Supervision():
    """Custom data class for variable used as supervision."""
    def __init__(self, target_data, target_type, target_name = None, target_kernel = None):
        """
        Args:
            target_data (2D tensor or ndarray): (n_sample, n_dim_target). The target data used as supervision.
            target_type (str): One of ['continuous', 'categorical', 'custom']. The type of the target data.
                If 'custom', the target_kernel should be provided.
            target_name (str): The name of the target data.
            target_kernel (2D tensor): (n_sample, n_sample). Optional. The kernel matrix of the target data.
                Once provided, the target_data will be ignored.
        """
        self.target_data = target_data # (n_sample, n_dim_target) or None
        self.target_type = target_type
        self.target_name = target_name
        self.target_kernel = target_kernel # (n_sample, n_sample)

        self._sanity_check()

        # calculate the kernel matrix if not provided
        if self.target_kernel is None:
            self._calc_kernel()

        # save the number of samples
        self.n_sample = self.target_kernel.shape[0]

    def _sanity_check(self):
        # check the target type
        _valid_types = ['continuous', 'categorical', 'custom']
        assert self.target_type in _valid_types, \
            f"Currently only support type in {_valid_types}."

        if self.target_type == 'custom':
            # use pre-calculated kernel matrix and ignore the target data
            assert self.target_kernel is not None, \
                "If target_type is 'custom', the target_kernel should be provided."

            # convert the kernel matrix to tensor if ndarray
            if isinstance(self.target_kernel, np.ndarray):
                self.target_kernel = torch.from_numpy(self.target_kernel).float()

            # check the shape of the kernel matrix
            assert self.target_kernel.shape[0] == self.target_kernel.shape[1], \
                "The kernel matrix should be square."

            # set the target data to None
            self.target_data = None
        else:
            # preprocess the target data for continuous and categorical targets
            if len(self.target_data.shape) == 1:
                self.target_data = self.target_data[:, None] # (n_sample, 1)

            if isinstance(self.target_data, np.ndarray):
                # convert categorical string to integer
                if self.target_data.dtype.kind in {'S', 'U'}:
                    self.target_data = np.concatenate(
                        [np.unique(self.target_data[:, i], return_inverse = True)[1][:, None]
                        for i in range(self.target_data.shape[1])],
                        axis = 1
                    )

                self.target_data = torch.from_numpy(self.target_data).float()

            assert self.target_data.dim() == 2, \
                "The target data should be 2D tensor with (n_sample, n_dim_target)."

    def _calc_kernel(self):
        # TODO: low-rank approximation for large dataset
        """Calculate the kernel matrix of the target data."""
        if self.target_type == 'continuous':
            _y = normalize_col(self.target_data, center = True, scale = False).float()
            self.target_kernel = _y @ _y.t()
        elif self.target_type == 'categorical':
            self.target_kernel = delta_kernel(self.target_data)
        else:
            raise ValueError("Currently only support 'continuous' or 'categorical' targets.")

class SISPCADataset(Dataset):
    """Custom dataset for supervised independent subspace PCA (sisPCA)."""
    def __init__(self, data, target_supervision_list: List[Supervision]):
        """
        Args:
            data (2D tensor): (n_sample, n_feature). Data to run sisPCA on.
            target_supervision_list (list of Supervision): List of Supervision objects.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # the input data
        self.x = normalize_col(data, center = True, scale = False).float()
        self.n_sample = data.shape[0]
        self.n_feature = data.shape[1]

        # check supervision list
        for t in target_supervision_list:
            assert self.n_sample == t.n_sample, \
                "The number of samples in data and target supervision should be the same."

        # the supervised variable (target)
        self.target_supervision_list = target_supervision_list
        self.n_target = len(target_supervision_list)

        # extract target data and kernel
        self.target_data_list = [t.target_data for t in target_supervision_list]
        self.target_kernel_list = [t.target_kernel for t in target_supervision_list]

        # extract target names and replace None with default names
        self.target_name_list = [t.target_name for t in target_supervision_list]
        for i, name in enumerate(self.target_name_list):
            if name is None:
                self.target_name_list[i] = f"key_{i}"

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        sample = {
            'index': idx,
            'x': self.x[idx,:],
        }

        # append target data to the batch
        for (_name, _target_data) in zip(
            self.target_name_list, self.target_data_list
        ):
            if _target_data is not None:
                sample[f"target_data_{_name}"] = _target_data[idx,:]
            else:
                sample[f"target_data_{_name}"] = torch.empty(0)

        return sample