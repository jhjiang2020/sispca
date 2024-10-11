import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset
from sispca.utils import normalize_col, delta_kernel

class Supervision():
    """Custom data class for variable used as supervision."""
    def __init__(self, target_data, target_type, target_name = None):
        self.target_data = target_data # (n_sample, n_dim_target)
        self.target_type = target_type
        self.target_name = target_name
        self.target_kernel = None # (n_sample, n_sample)

        self._sanity_check()
        self._calc_kernel()

    def _sanity_check(self):
        assert self.target_type in ['continuous', 'categorical'], \
            "Currently only support 'continuous' or 'categorical' targets."

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
        """Calculate the kernel matrix of the target data."""
        if self.target_type == 'continuous':
            _y = normalize_col(self.target_data, center = True, scale = False).float()
            self.target_kernel = _y @ _y.t()
        else: # 'categorical'
            self.target_kernel = delta_kernel(self.target_data)

class SISPCADataset(Dataset):
    """Custom dataset for supervised independent subspace PCA (sisPCA)."""
    def __init__(self, data, target_supervision_list: List[Supervision]):
        """Initialize the dataset.

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
            sample[f"target_data_{_name}"] = _target_data[idx,:]

        return sample