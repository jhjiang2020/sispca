import unittest
import numpy as np
import torch
from sispca.data import Supervision, SISPCADataset

class TestSupervision(unittest.TestCase):
    def test_kernel(self):
        # Test case 1: categorical target
        supervision = Supervision(
            target_data=np.array([1, 2, 3]), target_type='categorical'
        )
        self.assertTrue(torch.equal(supervision.target_kernel.realization(), torch.eye(3)))

        # Test case 2: continuous target
        x = torch.randn(10, 2)
        supervision = Supervision(
            target_data=x, target_type='continuous'
        )
        x_new = x - x.mean(0, keepdim = True)
        K_x = x_new @ x_new.T

        self.assertTrue(torch.equal(supervision.target_kernel.realization(), K_x))

        # Test case 3: custom target
        s1 = Supervision(
            target_data=None, target_type='custom', target_kernel_K=K_x
        )
        s2 = Supervision(
            target_data=None, target_type='custom', target_kernel_Q=x_new
        )
        self.assertTrue(torch.equal(s1.target_kernel.realization(), s2.target_kernel.realization()))


class TestSISPCADataset(unittest.TestCase):
    def setUp(self):
        # Create a SisPCADataset instance
        data = np.random.randn(4, 3)
        target_supervision_list = [
            Supervision(target_data=np.array(['1', '1', '2', '3']), target_type='categorical'),
            Supervision(target_data=np.random.randn(4, 2), target_type='continuous'),
            Supervision(target_data=None, target_type='custom', target_kernel_Q=data),
        ]
        self.dataset = SISPCADataset(data=data, target_supervision_list=target_supervision_list)

    def test_len(self):
        # Call the __len__ method
        result = len(self.dataset)

        # Assert that the result is equal to the length of the data
        self.assertEqual(result, 4)

    def test_getitem(self):
        # Call the __getitem__ method
        result = self.dataset[0]

        # Assert that the result is a dictionary with the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('index', result)
        self.assertIn('x', result)
        self.assertIn('target_data_key_0', result)
        self.assertIn('target_data_key_1', result)

if __name__ == '__main__':
    unittest.main()