import unittest
import warnings
import torch
import numpy as np
from scipy.linalg import subspace_angles
from sispca.utils import normalize_col
from sispca.model import PCA, SISPCA
from sispca.data import Supervision, SISPCADataset

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.n_feature = 100
        self.n_latent = 10
        self.lr = 0.1
        self.pca = PCA(self.n_feature, self.n_latent, self.lr)

    def test_forward(self):
        x = torch.randn(32, self.n_feature)
        output = self.pca.forward(x)
        self.assertEqual(output.shape, (32, self.n_latent))

    def test_loss(self):
        x = torch.randn(32, self.n_feature)
        loss = self.pca.loss(x)
        self.assertIsInstance(loss, torch.Tensor)

    def test_training_step(self):
        batch = torch.randn(32, self.n_feature)
        batch_idx = 0
        # suppress user warning
        with self.assertWarns(UserWarning):
            output = self.pca.training_step(batch, batch_idx)
            self.assertIsInstance(output, torch.Tensor)

class TestSISPCAEig(unittest.TestCase):
    def setUp(self):
        self.dataset = SISPCADataset(
            data=torch.randn(100, 20),
            target_supervision_list=[
                Supervision(
                    target_data=torch.randn(100, 10),
                    target_type='continuous'
                ),
                Supervision(
                    target_data=np.random.choice(['A', 'B', 'C'], 100),
                    target_type='categorical'
                )
            ]
        )
        self.n_latent_sub = [10, 10]
        self.lambda_contrast = 1.0
        self.kernel_subspace = 'linear'
        self.solver = 'eig'

        # initialize the SISPCA model
        self.sispca = SISPCA(
            self.dataset,
            self.n_latent_sub,
            self.lambda_contrast,
            self.kernel_subspace,
            solver = self.solver
        )

    def test_loss(self):
        batch = self.dataset[0:32]

        loss_dict = self.sispca.loss(batch)
        reg_loss = loss_dict['reg_loss'] # HSIC contrastive loss
        loss_sum = sum(loss_dict['recon_loss']) + reg_loss

        self.assertIsInstance(loss_sum, torch.Tensor)

    def test_training_step(self):
        batch = self.dataset[0:32]
        batch_idx = 0
        # suppress user warning
        with self.assertWarns(UserWarning):
            output = self.sispca.training_step(batch, batch_idx)
            self.assertIsInstance(output, dict)

    def test_fit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sispca.fit(batch_size=-1, shuffle=True, max_epochs=20)

    def test_get_latent_representation(self):
        latent_rep = self.sispca.get_latent_representation()
        self.assertEqual(latent_rep.shape, (100, 20))

    def test_pca_accuracy(self):
        pca_data = torch.randn(100, 20) * 10

        # (1) run PCA through svd
        u, s, _ = torch.linalg.svd(
            normalize_col(pca_data, center = True, scale = False), 
            full_matrices=False
        )
        pca_rep_1 = (u @ torch.diag(s))[:, :10]

        # (2) run PCA through SISPCA
        # initialize the PCA model as a special case of sPCA
        self.pca = SISPCA(
            dataset = SISPCADataset(
                data=pca_data,
                target_supervision_list=[]
            ),
            n_latent_sub = [10],
            lambda_contrast = 0.0,
            kernel_subspace = 'linear',
            solver = self.solver,
        )
        # fit the sispca model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pca.fit(batch_size=-1, shuffle=True, max_epochs=20, lr=1)
        pca_rep_2 = self.pca.get_latent_representation()

        # (3) compare PCA subspaces
        pca_accuracy = np.linalg.norm(subspace_angles(pca_rep_1, pca_rep_2))
        self.assertAlmostEqual(pca_accuracy, 0.0, places=3)

class TestSISPCAGd(TestSISPCAEig):
    def setUp(self):
        self.dataset = SISPCADataset(
            data=torch.randn(100, 20),
            target_supervision_list=[
                Supervision(
                    target_data=torch.randn(100, 10),
                    target_type='continuous'
                ),
                Supervision(
                    target_data=np.random.choice(['A', 'B', 'C'], 100),
                    target_type='categorical'
                )
            ]
        )
        self.n_latent_sub = [10, 10]
        self.lambda_contrast = 1.0
        self.kernel_subspace = 'linear'
        self.solver = 'gd'

        # initialize the SISPCA model
        self.sispca = SISPCA(
            self.dataset,
            self.n_latent_sub,
            self.lambda_contrast,
            self.kernel_subspace,
            solver = self.solver
        )

    def test_loss(self):
        batch = self.dataset[0:32]

        loss_dict = self.sispca.loss(batch)
        reg_loss = loss_dict['reg_loss'] # HSIC contrastive loss
        loss_sum = sum(loss_dict['recon_loss']) + reg_loss
        self.assertIsInstance(loss_sum, torch.Tensor)
        
        # check loss back propagation
        loss_sum.backward()
        self.assertTrue(self.sispca.U.grad is not None)

    def test_pca_accuracy(self):
        pass

if __name__ == '__main__':
    unittest.main()