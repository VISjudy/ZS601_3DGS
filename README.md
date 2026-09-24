# ZS601 v018: depth / normal / virtual-view ablation

One shared trainer, six parameter presets, one pinned Colab notebook per experiment.
This branch contains source code, CUDA extension source, configuration and reproduction notebooks only.
Images, point clouds, supervision arrays, checkpoints and results remain in Google Drive.

| Preset | Input | Depth loss | Normal loss | Virtual RGB views |
|---|---|---|---|---|
| A | Synthetic, 3cm initialization | off | off | 0 |
| B | Same | on | off | 0 |
| C | Same | off | on | 0 |
| D | Same | on | on | 0 |
| E | Same | on | on | 200 |
| F-real | Real LiDAR / acquired RGB | on | on | 200, generated from real LiDAR |

All groups use 50,000 iterations, validation every 5,000 steps and a final checkpoint at 50,000.
For consistency with the already-running A, `position_lr_max_steps=150000` and the existing
densification / warm-up schedule are unchanged. This is a common truncated schedule, not a
new 50k learning-rate schedule. Depth weight is 0.1, normal weight 0.05, ramp starts at 1,000
and spans 4,000 steps. `paper_arm` controls the losses; all groups use `experiment=original`.
Do not use legacy geometry presets with the same A/B/C letters.

## Reproduce

1. Open the desired notebook in `notebooks/` on Colab; choose a GPU (reference run: L4).
2. Mount your own Drive. Set its dataset root and a **new** output name in the configuration cell.
3. The notebook checks out an immutable Git commit, builds the vendored CUDA extensions,
   stages shared inputs on Colab local disk, validates inputs, and runs the selected preset.
4. Outputs go directly to Drive. Each newly started run saves its notebook and resolved config
   in the output directory. Existing output directories cause a hard failure instead of overwrite.

`scripts/run_experiment.py` prints and validates the command unless `--execute` is supplied.
Do not run these notebooks against the current formal output while its queue is active.

## What is actually reproduced

`provenance/trainer-source-sha256.json` matches the active Colab trainer after the authorized
50k final-test change. The original pre-change hashes are retained as lineage metadata.
A was started with 150k, then shortened: the controller selects its complete 50k checkpoint,
stops the old process, and evaluates that checkpoint's PLY. A fresh notebook uses the same
first-50k optimization behavior, including the final optimizer update. It does not reenact
the storage migration. B/C/D are queued; E and F-real were not run at publication.

Only the historical trainer source has GPU smoke evidence. The notebook packaging has
syntax, configuration, source-hash and preflight tests; no duplicate 50k training was launched
to test the notebooks. GPU kernels are not promised bitwise deterministic across runtimes.
The supplied environment lock records the observed runtime; Colab may require a new runtime
after changing PyTorch. Its Python reference version is 3.13.15.

The normal loss compares a normal derived from rendered depth against the camera-space PCA
normal of LiDAR points. It is not a Gaussian-axis loss, nor a direct implementation of the full
2DGS objective. Virtual RGB views use valid alpha masks and are excluded from LiDAR supervision.

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for exact dataset prerequisites and real-data limitations.
Third-party sources retain their original licenses in `LICENSE.md` and `submodules/`.
