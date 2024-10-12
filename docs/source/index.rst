Welcome
=======

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Getting Started

  self
  tutorials/index

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: API Reference

  autoapi/sispca/index

`SISPCA (Supervised Independent Subspace Principal Component Analysis) <https://github.com/JiayuSuPKU/sispca>`_ is a Python package designed to learn linear representations capturing variations associated with factors of interest in high-dimensional data. 
It extends the Principal Component Analysis (PCA) to multiple subspaces and encourage subspace disentanglement by maximizing the Hilbert-Schmidt Independence Criterion (HSIC). 
The model is implemented in `PyTorch <https://pytorch.org/>`_ and uses the `Lightning framework <https://lightning.ai/docs/pytorch/stable/>`_ for training. 

.. image:: ../img/sisPCA.png
  :alt: Overview
  :width: 800

For more theoretical connections and applications, please refer to our paper 
`Disentangling Interpretable Factors of Variations with Supervised Independent Subspace Principal Component Analysis <https://openreview.net/forum?id=AFnSMlye5K>`_.

.. note::

  This documentation is under active development.

Installation
------------
The package can be installed via pip:

.. code-block:: zsh

  # from PyPI (stable version)
  $ pip install sispca

  # or from github (latest version)
  $ pip install git+https://github.com/JiayuSuPKU/sispca.git#egg=sispca

Tutorials
---------

See the :ref:`tutorial gallery <gallery>` for examples on how to use the package.

Citation
-----------
If you find this work useful, please consider citing our paper:

.. code-block:: bibtex

  @inproceedings{
    su2024disentangling,
    title={Disentangling Interpretable Factors of Variations with Supervised Independent Subspace Principal Component Analysis},
    author={Jiayu Su, David A. Knowles, and Raul Rabadan},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=AFnSMlye5K}
  }


