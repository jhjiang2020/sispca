import unittest
import numpy as np
import torch
from sispca.utils import gaussian_kernel, delta_kernel, hsic_gaussian, hsic_linear, Kernel

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

class TestKernel(unittest.TestCase):
	def setUp(self):
        # Create a Kernel instance
		Q = torch.randn(10, 3)
		Q2 = 10
		self.kernel = Kernel('continuous', Q)
		self.Q = Q
		self.kernel_id = Kernel('identity', Q2)

	def test_shape(self):
        # Call the shape attribute
		result = self.kernel.shape
        # Assert that the shape is equal to the kernel shape
		self.assertEqual(result, (10,10))

	def test_shape_id(self):
		# Make sure the shape of identity matrix is correct as well
		result2 = self.kernel_id.shape
		self.assertEqual(result2, (10,10))

	def test_real(self):
		# Call the realization function
		K = self.kernel.realization()
		K2 = self.Q @ self.Q.T
		self.assertTrue(torch.isclose(K, K2, 1e-5).all())

	def test_real_id(self):
		# Call the realization function
		K = self.kernel_id.realization()
		# Assert that the result is equal to the length of the data
		K2 = torch.eye(10)
		self.assertTrue(torch.isclose(K, K2, 1e-5).all())

	def test_xtKx(self):
		# Call the xtKx function
		x = torch.randn(10, 20)
		result = self.kernel.xtKx(x)
		result2 = x.T @ (self.Q @ self.Q.T) @ x
		self.assertTrue(torch.isclose(torch.norm(result - result2), torch.tensor(0.0), atol=1e-4))
		# self.assertTrue(torch.isclose(result, result2, 1e-5).all())

	def test_xtKx_id(self):
		# Call the xtKx function
		x = torch.randn(10, 20)
		result = self.kernel_id.xtKx(x)
		result2 = x.T @ torch.eye(10) @ x
		self.assertTrue(torch.isclose(result, result2, 1e-5).all())

	def test_rank(self):
		# Call the rank function
		result = self.kernel.rank()
		result2 = torch.linalg.matrix_rank(self.Q @ self.Q.T)
		self.assertEqual(result, result2)

	def test_custom(self):
		# Create a custom kernel
		data = torch.randn(10, 2)
		k1 = Kernel(target_type='custom', target_kernel=(data @ data.T))
		k2 = Kernel(target_type='custom', Q=data)
		k3 = Kernel(target_type='custom', target_kernel=(data @ data.T).to_sparse())

		# Call the realization function
		res1 = k1.subset(torch.tensor([0, 1, 2])).realization()
		res2 = k2.subset(torch.tensor([0, 1, 2])).realization()
		res3 = k3.subset(torch.tensor([0, 1, 2])).realization().to_dense()
		self.assertTrue(torch.isclose(res1, res2, 1e-5).all())
		self.assertTrue(torch.isclose(res1, res3, 1e-5).all())

		# Call the xtKx function
		x = torch.randn(10, 3)
		res4 = k1.xtKx(x)
		res5 = k2.xtKx(x)
		res6 = k3.xtKx(x)
		self.assertTrue(torch.isclose(res4, res5, 1e-5).all())
		self.assertTrue(torch.isclose(res4, res6, 1e-5).all())

if __name__ == '__main__':
	unittest.main()