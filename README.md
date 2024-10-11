# Supervised Independent Subspace Principal Component Analysis (sisPCA)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

<!-- ![Overview](/docs/img/sisPCA.png) -->

<!-- fig -->
<div align="center">
<img src="docs/img/sisPCA.png" alt="Overview" width="600px"/>
</div>

*sispca* is a Python package designed to learn linear representations capturing variations associated with factors of interest in high-dimensional data. It extends the Principal Component Analysis (PCA) to multiple subspaces and encourage subspace disentanglement by maximizing the Hilbert-Schmidt Independence Criterion (HSIC). The model is implemented in [PyTorch](https://pytorch.org/) and uses the [Lightning framework](https://lightning.ai/docs/pytorch/stable/) for training. See the [documentation](https://sispca.readthedocs.io/en/latest/index.html) for more details.

For more theoretical connections and applications, please refer to our paper [Disentangling Interpretable Factors of Variations with Supervised Independent Subspace Principal Component Analysis](https://openreview.net/forum?id=AFnSMlye5K).

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
```python
from sispca import Supervision, SISPCADataset, SISPCA
```
Tutorials:
* [Feature selection using sisPCA on the Breast Cancer Wisconsin dataset](docs/source/tutorials/tutorial_brca.ipynb).

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
