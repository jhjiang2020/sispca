# Supervised Independent Subspace Principal Component Analysis (sisPCA)
[![DOI](https://zenodo.org/badge/871005850.svg)](https://doi.org/10.5281/zenodo.13932661)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

<!-- ![Overview](/docs/img/sisPCA.png) -->

<!-- fig -->
<div align="center">
<img src="docs/img/sisPCA.png" alt="Overview" width="600px"/>
</div>

*sispca* is a Python package designed to learn linear representations capturing variations associated with factors of interest in high-dimensional data. It extends the Principal Component Analysis (PCA) to multiple subspaces and encourage subspace disentanglement by maximizing the Hilbert-Schmidt Independence Criterion (HSIC). The model is implemented in [PyTorch](https://pytorch.org/) and uses the [Lightning framework](https://lightning.ai/docs/pytorch/stable/) for training. See the [documentation](https://sispca.readthedocs.io/en/latest/index.html) for more details.

For more theoretical connections and applications, please refer to our paper [Disentangling Interpretable Factors with Supervised Independent Subspace Principal Component Analysis](https://openreview.net/forum?id=AFnSMlye5K).

## Installation
Via GitHub (latest version):
```bash
pip install git+https://github.com/JiayuSuPKU/sispca.git#egg=sispca
```

Via PyPI (stable version):
```bash
pip install sispca
```

## Getting Started
Basic usage:
```python
import numpy as np
import torch
from sispca import Supervision, SISPCADataset, SISPCA

# simulate random inputs
x = torch.randn(100, 20)
y_cont = torch.randn(100, 5) # continuous target
y_group = np.random.choice(['A', 'B', 'C'], 100) # categorical target
L = torch.randn(100, 20)
K_y = L @ L.T # custom kernel, (n_sample, n_sample)

# create a dataset with supervision
sdata = SISPCADataset(
    data = x.float(), # (n_sample, n_feature)
    target_supervision_list = [
        Supervision(target_data=y_cont, target_type='continuous'),
        Supervision(target_data=y_group, target_type='categorical'),
        Supervision(target_data=None, target_type='custom', target_kernel = K_y)
    ]
)

# fit the sisPCA model
sispca = SISPCA(
    sdata, 
    n_latent_sub=[3, 3, 3, 3], # the last subspace will be unsupervised 
    lambda_contrast=10,
    kernel_subspace='linear',
    solver='eig'
)
sispca.fit(batch_size = -1, max_epochs = 100, early_stopping_patience = 5)
```
Tutorials:
* [Feature selection using sisPCA on the Breast Cancer Wisconsin dataset](docs/source/tutorials/tutorial_brca.ipynb).
* [Learning unsupervised residual subspace in simulation](docs/source/tutorials/tutorial_donut.ipynb).
* [Learning interpretable infection subspaces in scRNA-seq data using sisPCA](docs/source/tutorials/tutorial_scrna_pca.ipynb).

For additional details, please refer to the [documentation](https://sispca.readthedocs.io/en/latest/index.html).


## Citation
If you find sisPCA useful in your research, please consider citing our paper:
```bibtex
@inproceedings{
  su2024disentangling,
  title={Disentangling Interpretable Factors of Variations with Supervised Independent Subspace Principal Component Analysis},
  author={Jiayu Su, David A. Knowles, and Raul Rabadan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=AFnSMlye5K}
}
```
