"""CPU-only analytical tests; no CUDA rasterizer import."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from precomputed_v3 import PrecomputedSupervision, depth_to_camera_normals, losses_from_maps


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (9, 11)
        self.k = (12., 12., 5.5, 4.5)

    def target(self):
        depth = torch.full(self.shape, 2., dtype=torch.float64)
        normal = torch.zeros((*self.shape, 3), dtype=torch.float64)
        normal[..., 2] = -1
        return dict(depth=depth, depth_valid=torch.ones(self.shape, dtype=torch.bool),
                    normal=normal, normal_valid=torch.ones(self.shape, dtype=torch.bool))

    def test_plane_orientation_and_border(self):
        target = self.target()
        normal, valid = depth_to_camera_normals(target['depth'], torch.ones(self.shape), self.k)
        self.assertEqual(int(valid.sum()), 7 * 9)
        torch.testing.assert_close(normal[valid], target['normal'][valid])

    def test_absolute_metric_depth_gradient(self):
        target = self.target()
        predicted = (target['depth'] + .01).requires_grad_()
        loss, log = losses_from_maps(predicted, torch.ones(self.shape), target, self.k, depth_weight=1.)
        self.assertAlmostEqual(float(loss.detach()), .0025, places=10)
        self.assertEqual(log['depth_state'], 'ACTIVE')
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertTrue((predicted.grad > 0).all())

    def test_normal_backprop_on_tilted_plane(self):
        target = self.target()
        x = (torch.arange(11, dtype=torch.float64) + .5 - self.k[2]) / self.k[0]
        predicted = (2. / (1. - .15*x)).expand(self.shape).clone().requires_grad_()
        loss, log = losses_from_maps(predicted, torch.ones(self.shape), target, self.k, normal_weight=1.)
        self.assertGreater(float(loss.detach()), 0)
        self.assertEqual(log['normal_valid_pixels'], 63)
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad.abs().sum()), 0)
        next_loss, _ = losses_from_maps(predicted.detach() - .01*predicted.grad,
                                        torch.ones(self.shape), target, self.k, normal_weight=1.)
        self.assertLess(float(next_loss), float(loss.detach()))

    def test_invalid_cross_and_sparse_target(self):
        target = self.target()
        target['normal_valid'].zero_()
        target['normal_valid'][4, 5] = True
        alpha = torch.ones(self.shape)
        _, valid = depth_to_camera_normals(target['depth'], alpha, self.k)
        self.assertTrue(valid[4, 5])
        loss, log = losses_from_maps(target['depth'], alpha, target, self.k, normal_weight=1.)
        self.assertEqual(log['normal_valid_pixels'], 1)
        alpha[4, 6] = 0
        loss, log = losses_from_maps(target['depth'].clone().requires_grad_(), alpha, target, self.k, normal_weight=1.)
        self.assertEqual(log['normal_valid_pixels'], 0)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()

    def test_nonfinite_prediction_mask(self):
        target = self.target()
        predicted = target['depth'].clone()
        predicted[4, 5] = float('nan')
        predicted.requires_grad_()
        loss, _ = losses_from_maps(predicted, torch.ones(self.shape), target, self.k, depth_weight=1., normal_weight=1.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_provider_missing_explicit_skip_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            arrays = Path(folder) / 'arrays'
            arrays.mkdir()
            target = self.target()
            np.savez(arrays/'000123.npz', depth=target['depth'].numpy(),
                     point_ids=np.zeros(self.shape, dtype=np.int32),
                     normal=target['normal'].numpy(), normal_valid=target['normal_valid'].numpy())
            provider = PrecomputedSupervision(folder, {'real.png': 123}, cache_size=1, skip_images={'virtual.png'})
            one = provider.get('real.png', 'cpu', expected_shape=self.shape)
            self.assertIs(one, provider.get('real.png', 'cpu'))
            self.assertIsNone(provider.get('virtual.png', 'cpu'))
            with self.assertRaises(KeyError):
                provider.get('missing-real.png', 'cpu')
            with self.assertRaises(ValueError):
                provider.get('real.png', 'cpu', expected_shape=(2, 2))


if __name__ == '__main__':
    unittest.main()
