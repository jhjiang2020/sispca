import unittest
import numpy as np
import torch
from sispca.utils import gaussian_kernel, delta_kernel, hsic_gaussian, hsic_linear

class TestUtils(unittest.TestCase):
	def test_delta_kernel(self):
		group_labels = np.array(
			[[0, 0, 1, 1, 2, 2],
			 ['0', '1', '1', '2', '2', '0']]
		).T

		# Test case 1: single group feature
		self.assertEqual(delta_kernel(group_labels[:, 0]).sum(), 12)

		# Test case 2: two group features
		self.assertEqual(
			delta_kernel(group_labels).sum(),
			(delta_kernel(group_labels[:, 0]) + delta_kernel(group_labels[:, 1])).sum()
		)

	def test_hsic(self):
		x = torch.randn(10, 3)
		y = torch.randn(10, 2)
		hsic = hsic_linear(x, y)

		x_new = x - x.mean(0, keepdim = True)
		y_new = y - y.mean(0, keepdim = True)
		hsic_2 = torch.trace(x_new @ x_new.T @ y_new @ y_new.T) / 9 ** 2
		self.assertTrue(torch.isclose(hsic, hsic_2, 1e-5))

if __name__ == '__main__':
	unittest.main()