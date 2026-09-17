import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)


class StandardTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], text=True, capture_output=True)

    def test_create_run_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = root / "dataset.json"; manifest.write_text("{}", encoding="utf-8")
            args = ("--root", root / "runs", "--run-id", "A_001", "--group", "A",
                    "--git-commit", "abc123", "--dataset-manifest", manifest,
                    "--feature", "surface_loss=on")
            first = self.run_script("create_run.py", *args)
            self.assertEqual(first.returncode, 0, first.stderr)
            cfg = json.loads((root / "runs" / "A_001" / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["feature_overrides"]["surface_loss"], "on")
            second = self.run_script("create_run.py", *args)
            self.assertNotEqual(second.returncode, 0)

    def make_smoke(self, run):
        run.mkdir(); (run / "run_config.json").write_text(json.dumps({"experiment_group": "A"}), encoding="utf-8")
        (run / "run_manifest.json").write_text("{}", encoding="utf-8")
        (run / "environment.json").write_text("{}", encoding="utf-8")
        write_csv(run / "loss_log.csv", ["iteration", "total_loss"], [{"iteration": 1, "total_loss": .5}])
        write_csv(run / "training_progress.csv", ["iteration", "elapsed_seconds", "gaussian_count"], [{"iteration": 200, "elapsed_seconds": 10, "gaussian_count": 20}])
        write_csv(run / "geometry_metrics.csv", ["iteration", "gaussian_count"], [{"iteration": 200, "gaussian_count": 20}])
        write_csv(run / "val_metrics.csv", ["iteration", "image_name", "psnr"], [{"iteration": i, "image_name": "MEAN", "psnr": 20} for i in (0, 200)])
        for i in (0, 200):
            folder = run / "val" / f"iteration_{i:06d}"; folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(json.dumps({"camera_count": 10}), encoding="utf-8")
        (run / "checkpoints").mkdir(); (run / "checkpoints" / "iteration_200.pth").write_bytes(b"x")
        model = run / "point_cloud" / "iteration_200"; model.mkdir(parents=True); (model / "point_cloud.ply").write_bytes(b"ply")

    def test_smoke_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; self.make_smoke(run)
            result = self.run_script("verify_run_outputs.py", run, "--profile", "smoke", "--iterations", "200", "--output", "verification.json")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((run / "verification.json").read_text(encoding="utf-8"))
            self.assertTrue(report["verified"])

    def test_summary_uses_real_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"; run.mkdir()
            (run / "run_config.json").write_text(json.dumps({"run_id": "B_001", "experiment_group": "B", "git_commit": "abc", "resolved_feature_flags": {"surface_loss": True}}), encoding="utf-8")
            write_csv(run / "training_progress.csv", ["iteration", "elapsed_seconds", "gaussian_count"], [{"iteration": 150000, "elapsed_seconds": 50, "gaussian_count": 100}])
            test = run / "test_final" / "iteration_150000"; test.mkdir(parents=True)
            write_csv(test / "test_metrics_per_camera.csv", ["image_name", "psnr", "ssim", "mae"], [{"image_name": "a", "psnr": 20, "ssim": .8, "mae": .1}, {"image_name": "b", "psnr": 22, "ssim": .9, "mae": .05}])
            result = self.run_script("summarize_experiment.py", run)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("21", (run / "results_table.csv").read_text(encoding="utf-8"))
            self.assertIn("surface_loss", (run / "experiment_summary.md").read_text(encoding="utf-8"))


    def test_formal_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "formal"; run.mkdir()
            (run / "run_config.json").write_text(json.dumps({"experiment_group": "C"}), encoding="utf-8")
            (run / "run_manifest.json").write_text("{}", encoding="utf-8")
            (run / "environment.json").write_text("{}", encoding="utf-8")
            for name in ("loss_log.csv", "training_progress.csv", "geometry_metrics.csv"):
                write_csv(run / name, ["iteration", "value"], [{"iteration": 150000, "value": 1}])
            iterations = list(range(0, 150001, 5000))
            write_csv(run / "val_metrics.csv", ["iteration", "image_name", "psnr"], [{"iteration": i, "image_name": "MEAN", "psnr": 20} for i in iterations])
            for i in iterations:
                folder = run / "val" / f"iteration_{i:06d}"; folder.mkdir(parents=True)
                (folder / "manifest.json").write_text(json.dumps({"camera_count": 10}), encoding="utf-8")
            for i in (50000, 100000, 150000):
                (run / "checkpoints").mkdir(exist_ok=True); (run / "checkpoints" / f"iteration_{i}.pth").write_bytes(b"x")
                model = run / "point_cloud" / f"iteration_{i}"; model.mkdir(parents=True); (model / "point_cloud.ply").write_bytes(b"ply")
            final = run / "test_final" / "iteration_150000"; final.mkdir(parents=True)
            write_csv(final / "test_metrics_per_camera.csv", ["image_name", "psnr"], [{"image_name": "a", "psnr": 20}])
            (final / "test_summary.json").write_text("{}", encoding="utf-8")
            for i in range(10): (final / f"worst_{i:02d}_rgb.png").write_bytes(b"png")
            for name in ("results_table.csv", "results_table.md", "results_table.tex", "experiment_summary.md"):
                (run / name).write_text("ok", encoding="utf-8")
            result = self.run_script("verify_run_outputs.py", run, "--profile", "formal", "--iterations", "150000")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_adapter_driver_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); adapter = root / "adapter.py"
            adapter.write_text("def render_validation(**kw):\n p=kw['output_dir']/'rgb.png';p.write_bytes(b'x');return {'camera_count':10,'artifacts':['rgb.png']}\n", encoding="utf-8")
            config = root / "run_config.json"; config.write_text("{}", encoding="utf-8")
            model = root / "model.ply"; model.write_bytes(b"ply")
            output = root / "diagnostics"
            result = self.run_script("render_val_diagnostics.py", "--adapter", adapter, "--run-config", config,
                                     "--model", model, "--output", output, "--iteration", "0")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["result"]["camera_count"], 10)
if __name__ == "__main__": unittest.main()
