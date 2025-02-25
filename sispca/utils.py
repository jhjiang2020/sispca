import torch
import numpy as np

def normalize_col(x, center = True, scale = True):
    """Z-score normalization.

    Args:
        x (2D tensor): (n_sample, n_feature).

    Returns:
        (x - x.mean(dim=0)) / x.std(dim=0)
    """
    x_new = x
    if center:
        x_new = x_new - x_new.mean(dim=0, keepdim = True)
    if scale:
        x_new = x_new / x_new.std(dim=0, keepdim = True)
    return x_new

def tr_cov(x):
    """Calculate the trace of the covariance matrix of the hidden representation.

    Args:
        x (2D tensor): (n_sample, n_feature).

    Returns:
        tr(x @ x.T)
    """
    return torch.norm(x, dim = 1).pow(2).sum()

def gaussian_kernel(x, bw = None):
    """Calculate the Gaussian kernel matrix.

    Args:
        x (2D tensor): (n_sample, n_feature).
        bw: Bendwidth of the Gaussian kernel.
            If None, will set to the median distance.

    Returns
        K (2D tensor): (n_sample, n_sample).
    """
    dist = torch.cdist(x, x, p=2.0)
    if bw is None:
        bw = dist.median()
    K = torch.exp(- 0.5 * dist.pow(2) / bw ** 2)
    return K

def delta_kernel(x):
    """Calculate the delta kernel matrix.

    Args:
        x (2D array): Category labels. (n_sample, n_feature).

    Returns
        K (2D tensor): (n_sample, n_sample).
    """
    if len(x.shape) == 1:
        x = x[:, None] # (n_sample, 1)

    K = torch.zeros(x.shape[0], x.shape[0])
    for i in range(x.shape[1]): # sum over all features
        K_i = (x[:, i:(i+1)] == x[:, i:(i+1)].T) * 1.0
        if isinstance(K_i, np.ndarray):
            K += torch.from_numpy(K_i).float()
        else:
            K += K_i.float()

    return K

def hsic_gaussian(x, y, bw = None):
    """Calculate the HSIC between two tensors using Gaussian kernel.

    Args:
        x (2D tensor): (n_sample, n_feature_1).
        y (2D tensor): (n_sample, n_feature_2).
        bw: Bendwidth of the Gaussian kernel.
            If None, will set to the median distance.

    Returns:
        HSIC (float): HSIC between x and y.
    """
    n_sapmle = x.shape[0]

    # scale x and y to unit variance
    x_new = normalize_col(x, center = True, scale = True)
    y_new = normalize_col(y, center = True, scale = True)

    # calculate kernel matrices
    K = gaussian_kernel(x_new, bw=bw) # (n_sample, n_sample)
    L = gaussian_kernel(y_new, bw=bw) # (n_sample, n_sample)

    # center kernel matrices
    K_H = normalize_col(K, center = True, scale = False)
    L_H = normalize_col(L, center = True, scale = False)

    # calculate HSIC
    return torch.trace(K_H @ L_H) / ((n_sapmle - 1) ** 2)

def hsic_linear(x, y):
    """Calculate the HSIC between two tensors using linear kernel.

    Args:
        x (2D tensor): (n_sample, n_feature_1).
        y (2D tensor): (n_sample, n_feature_2).

    Returns:
        HSIC (float): HSIC between x and y.
    """
    n_sapmle = x.shape[0]

    # scale x and y to unit variance
    x_new = normalize_col(x, center = True, scale = False)
    y_new = normalize_col(y, center = True, scale = False)

    # # calculate kernel matrices
    # K = x_new @ x_new.T # (n_sample, n_sample)
    # L = y_new @ y_new.T # (n_sample, n_sample)

    # # calculate HSIC
    # return torch.trace(K @ L) / ((n_sapmle - 1) ** 2)

    return (torch.norm(x_new.T @ y_new, 'fro') / (n_sapmle - 1)) ** 2


def gram_schmidt(x):
    """Project the data to an orthogonal space using Gram-Schmidt process.

    Args:
        x (2D tensor).

    Returns:
        x_new (2D tensor): data with orthonormal columns.
    """
    # x_new.T @ x_new == I
    return torch.linalg.qr(x, mode = 'reduced')[0]

class Kernel:
    def __init__(self, Q):
        self.Q = Q
        self.shape = self._shape()
    
    def _shape(self): 
        if isinstance(self.Q, int):
            return (self.Q, self.Q)
        return (self.Q.shape[0], self.Q.shape[0])

    def realization(self):
        if isinstance(self.Q, int):
            return torch.eye(self.Q)
        return self.Q @ self.Q.T

    def xtKx(self, x):
        if isinstance(self.Q, int):
            return x.T @ x
        return x.T @ self.Q @ self.Q.T @ x