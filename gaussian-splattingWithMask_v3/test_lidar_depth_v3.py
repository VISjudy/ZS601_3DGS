import unittest

import numpy as np
import torch

from arguments_v3 import preset_features
from lidar_depth_v3 import (
    camera_to_world_np,
    conservative_fill,
    decode_depth_u16,
    distance_weights,
    encode_depth_u16,
    scatter_zbuffer_min,
    world_to_camera_np,
)


class LidarDepthTests(unittest.TestCase):
    def test_e_preset_enables_c_geometry_and_depth(self):
        e = preset_features('E')
        for name in (
            'surface_loss', 'tangent_loss', 'normal_loss',
            'flatten_loss', 'size_loss', 'scale_bounds',
            'lidar_depth_loss',
        ):
            self.assertTrue(e[name], name)
        self.assertNotIn('surface', e)

    def test_zbuffer_keeps_nearest_depth(self):
        buffer = torch.full((3,), float('inf'))
        index = torch.tensor([0, 0, 1], dtype=torch.long)
        depth = torch.tensor([3.0, 2.0, 5.0])
        scatter_zbuffer_min(buffer, index, depth)
        self.assertEqual(buffer[0].item(), 2.0)
        self.assertEqual(buffer[1].item(), 5.0)
        self.assertTrue(torch.isinf(buffer[2]))

    def test_conservative_fill_accepts_plane_and_rejects_edge(self):
        coherent = torch.full((3, 3), float('inf'))
        coherent[1, 0] = 2.00
        coherent[1, 2] = 2.01
        hit = torch.isfinite(coherent)
        filled, valid = conservative_fill(
            coherent, hit, radius=1, min_neighbors=2,
            edge_absolute=.02, edge_relative=.02,
        )
        self.assertTrue(valid[1, 1])
        self.assertAlmostEqual(filled[1, 1].item(), 2.0, places=5)

        edge = coherent.clone()
        edge[1, 2] = 4.0
        _, valid_edge = conservative_fill(
            edge, torch.isfinite(edge), radius=1, min_neighbors=2,
            edge_absolute=.02, edge_relative=.02,
        )
        self.assertFalse(valid_edge[1, 1])

    def test_camera_transform_round_trip(self):
        angle = .37
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.array([.3, -.4, 1.2])
        world = np.array([[1.0, 2.0, 3.0], [-2.0, .5, 4.0]])
        camera = world_to_camera_np(world, rotation, translation)
        recovered = camera_to_world_np(camera, rotation, translation)
        np.testing.assert_allclose(recovered, world, rtol=0, atol=1e-12)

    def test_depth_png_encoding_round_trip(self):
        depth = np.array([[0.0, .1], [7.55, 15.0]], dtype=np.float32)
        valid = np.array([[False, True], [True, True]])
        encoded = encode_depth_u16(depth, valid, .1, 15.0)
        decoded, decoded_valid = decode_depth_u16(encoded, .1, 15.0)
        np.testing.assert_array_equal(decoded_valid, valid)
        bound = (15.0 - .1) / 65534.0 + 1e-6
        self.assertLessEqual(np.abs(decoded[valid] - depth[valid]).max(), bound)
        self.assertEqual(encoded[0, 0], 0)

    def test_distance_weight_prefers_near_depth(self):
        depth = torch.tensor([1.0, 2.0, 4.0])
        weights = distance_weights(depth, power=1.0, minimum=.25, maximum=4.0)
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[2].item())
        self.assertAlmostEqual(weights[1].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
