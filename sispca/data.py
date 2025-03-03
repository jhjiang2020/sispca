import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset
from sispca.utils import normalize_col, delta_kernel, Kernel
from sklearn.preprocessing import OneHotEncoder

class Supervision():
    """Custom data class for variable used as supervision."""
    def __init__(self, target_data, target_type, target_name = None, target_kernel_K = None, target_kernel_Q = None):
        """
        Args:
            target_data (2D tensor or ndarray): (n_sample, n_dim_target). The target data used as supervision.
            target_type (str): One of ['continuous', 'categorical', 'custom']. The type of the target data.
                If 'custom', the target_kernel should be provided.
            target_name (str): The name of the target data.
            target_kernel_K (2D tensor): (n_sample, n_sample). Optional. The kernel matrix of the target data.
                Once provided, the target_data will be ignored.
            target_kernel_Q (2D tensor): (n_sample, n_rank). Optional. The decomposed kernel matrix. 
                K = Q @ Q.T. Will be overriden by target_kernel_K if both are provided.
        """
        self.target_data = target_data # (n_sample, n_dim_target) or None
        self.target_type = target_type
        self.target_name = target_name
        self._target_kernel_K = target_kernel_K # (n_sample, n_sample) or None
        self._target_kernel_Q = target_kernel_Q # (n_sample, n_rank) or None

        self._sanity_check()

        # compute the kernel matrix from the target data or use the provided kernel matrix
        self.target_kernel = self._calc_kernel() # an object of Kernel

        # save the number of samples
        self.n_sample = self.target_kernel.shape[0]

    def _sanity_check(self):
        # check the target type
        _valid_types = ['continuous', 'categorical', 'custom']
        assert self.target_type in _valid_types, \
            f"Currently only support type in {_valid_types}."

        if self.target_type == 'custom':
            # will use pre-calculated kernel matrix and ignore the target data
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
        """Calculate the kernel matrix of the target data.
        
        Returns:
            Kernel: An object of Kernel. Use the realization() method to get the (n, n) kernel matrix.
        """
        if self.target_type == 'continuous':
            _y = normalize_col(self.target_data, center = True, scale = False).float()
            return Kernel(target_type='continuous', Q = _y)
        elif self.target_type == 'categorical':
            enc = OneHotEncoder()
            _y = torch.from_numpy(enc.fit_transform(self.target_data).toarray()).float() # get the one-hot encoding of the target data
            return Kernel(target_type='categorical', Q = _y)
        else: # target_type == 'custom'
            return Kernel(target_type='custom', target_kernel = self._target_kernel_K, Q = self._target_kernel_Q)

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
        self.target_kernel_list = [t.target_kernel for t in target_supervision_list] # list of Kernel objects

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
            if (_target_data is not None):
                sample[f"target_data_{_name}"] = _target_data[idx,:]
            else:
                sample[f"target_data_{_name}"] = torch.empty(0)

        return sample