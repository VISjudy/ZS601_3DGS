import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import preprocess_v3 as subject
from scripts import preprocess_dataset_v3 as cli


class PreprocessGeometryTests(unittest.TestCase):
    def camera(self):
        return subject.Camera(
            1, 1, "frame.png", 100, 100, "PINHOLE",
            np.array([50.0, 50.0, 50.0, 50.0]), np.eye(3), np.zeros(3)
        )

    def test_world_to_camera_center(self):
        camera = self.camera()
        camera.translation = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_allclose(camera.center, [1.0, 2.0, 3.0])

    def test_projection(self):
        xyz, uv, inside = subject.project(
            self.camera(), np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]))
        np.testing.assert_allclose(xyz[0], [0.0, 0.0, 2.0])
        np.testing.assert_allclose(uv[0], [50.0, 50.0])
        self.assertEqual(inside.tolist(), [True, False])

    @unittest.skipUnless(subject.cKDTree is not None, "requires scipy")
    def test_pca_and_camera_orientation_on_plane(self):
        grid = np.linspace(-0.5, 0.5, 5)
        points = np.array([[x, y, 2.0] for x in grid for y in grid])
        settings = subject.NormalSettings(knn=9, min_neighbors=6, max_curvature=0.1)
        normals, _, _, _, valid = subject.estimate_normals(points, settings)
        oriented, confidence, resolved, _, _ = subject.orient_normals_to_cameras(
            points, normals, [self.camera()], valid, settings)
        toward_camera = -points / np.linalg.norm(points, axis=1, keepdims=True)
        self.assertTrue(resolved.all())
        self.assertTrue((np.einsum("ij,ij->i", oriented, toward_camera) > 0).all())
        self.assertTrue((confidence > 0).all())

    def test_zbuffer_keeps_nearest_point(self):
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.1, 0.0, 1.0]])
        selected, _, camera_xyz, _ = subject._zbuffer(self.camera(), points, 0.01, 10.0)
        self.assertIn(0, selected.tolist())
        self.assertNotIn(1, selected.tolist())
        self.assertTrue((camera_xyz[:, 2] > 0).all())

    def test_images_txt_parser_keeps_blank_point_line(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text(
                "# Image list\n1 1 0 0 0 0 0 0 1 first.png\n\n"
                "2 1 0 0 0 0 0 0 1 second.png\n10 20 -1\n", encoding="utf-8")
            images = subject.read_colmap_images(Path(directory))
            self.assertEqual([images[key][3] for key in sorted(images)], ["first.png", "second.png"])

    @unittest.skipUnless(subject.cKDTree is not None, "requires scipy")
    def test_propagates_sign_from_resolved_neighbor(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        normals = np.array([[0., 0., 1.], [0., 0., -1.], [0., 0., -1.]])
        oriented, resolved = subject.propagate_normal_signs(
            points, normals, np.array([True, False, False]), np.ones(3, dtype=bool), 1)
        self.assertTrue(resolved.all())
        np.testing.assert_allclose(oriented[:, 2], 1.)

    def test_rejects_distorted_colmap_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras.txt").write_text("1 OPENCV 100 100 50 50 50 50 0 0 0 0\n", encoding="utf-8")
            (root / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already-undistorted"):
                subject.load_cameras(root)

    def test_depth_remains_valid_when_normal_is_invalid(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            subject.generate_supervision(
                np.array([[0., 0., 2.]]), np.array([[0., 0., -1.]]), np.array([False]),
                [self.camera()], output, near=.01, far=10., backprojection_samples=0)
            depth = np.load(output / "supervision" / "depth" / "frame.npy")
            normal_valid = np.asarray(subject.Image.open(
                output / "supervision" / "normal_valid" / "frame.png"))
            self.assertGreater(depth[50, 50], 0)
            self.assertEqual(int(normal_valid.sum()), 0)

    @unittest.skipUnless(subject.PlyData is not None, "requires plyfile")
    def test_disabled_pca_requires_input_normal_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "points.ply"
            subject.write_ply(source, {
                "x": np.array([0., 1., 0.]), "y": np.array([0., 0., 1.]),
                "z": np.array([1., 1., 1.]),
            })
            with self.assertRaisesRegex(ValueError, "PCA normal estimation is disabled"):
                subject.preprocess_lidar(
                    source, root / "processed", subject.NormalSettings(), [],
                    estimate_normals_enabled=False)

    def test_cli_records_preset_override_and_final_matrix(self):
        with TemporaryDirectory() as directory:
            scene = Path(directory)
            code = cli.main([
                "--stage", "supervision", "--scene-root", str(scene),
                "--experiment-group", "A", "--generate-supervision", "off",
            ])
            self.assertEqual(code, 0)
            config = json.loads((scene / "processed_v3" / "manifests" /
                                 "preprocess_run_config_supervision.json").read_text(encoding="utf-8"))
            self.assertEqual(config["resolved_preset"], "A")
            self.assertEqual(config["normalized_overrides"]["supervision_generation"], "off")
            self.assertFalse(config["resolved_feature_flags"]["supervision_generation"])
            self.assertTrue(config["resolved_feature_flags"]["pca_normal_estimation"])


if __name__ == "__main__":
    unittest.main()
