# ZS601 2DGS project memory

This document is maintained inside the clean `2dgs-zs601-mask-init` branch. It records practical failures, successful steps, and recovery notes for the ZS601 mask-aware 2DGS Colab workflow.

## Current branch contract

- Branch: `2dgs-zs601-mask-init`
- Top-level branch content: only `2d-gaussian-splattingWithMask/`
- Source fork: official `hbb1/2d-gaussian-splatting` imported from commit `f3e3b9fa67bbd1c75e05167ff37391d8dab2a678`
- ZS601 additions: nested COLMAP image-path loading, `--masks`, `--mask_valid_value black|white`, and masked RGB/SSIM/regularization/validation metrics.

## Success log

### 2026-09-11: clean GitHub source branch

- Created and pushed a clean root commit so GitHub branch `2dgs-zs601-mask-init` contains only `2d-gaussian-splattingWithMask/`.
- Verified with `git ls-tree --name-only origin/2dgs-zs601-mask-init`: only `2d-gaussian-splattingWithMask` appears at the top level.
- Verified full tree filter: no paths outside `2d-gaussian-splattingWithMask/*` remain.

### 2026-09-11: Colab Pro+ smoke notebook prepared

- Added `colab/ZS601_2DGS_ProPlus_smoke.ipynb` under the 2DGS source folder.
- The notebook is designed to check GPU evidence, mount Drive, clone this branch, compile CUDA extensions, stage ZS601 data under `/content`, validate image-mask matching, and run a 200-step mask-aware COLMAP-sparse smoke test.
- Formal 150k training is intentionally gated behind smoke success.


### 2026-09-11: Colab tab deployment attempt

Success:
- Pushed `colab/ZS601_2DGS_ProPlus_smoke.ipynb` to branch `2dgs-zs601-mask-init`.
- Opened the notebook URL in a Chrome tab: `https://colab.research.google.com/github/VISjudy/ZS601_3DGS/blob/2dgs-zs601-mask-init/2d-gaussian-splattingWithMask/colab/ZS601_2DGS_ProPlus_smoke.ipynb`.

Failure / current blocker:
- Browser automation through the available CUA layer failed with `windows sandbox failed: helper_unknown_error: setup refresh had errors`.
- Colab MCP connection returned true, but only exposed the browser-connection handshake tool and did not expose notebook cell execution tools.
- Therefore runtime selection, Drive authorization, and cell execution still require user interaction in the opened Colab tab.
- Do not record smoke success until the notebook prints fresh `nvidia-smi`, builds both CUDA extensions, runs 200 training steps, and saves/read-backs `verification.json`.

## Failure / risk log

### Browser and authorization boundary

- Google sign-in, OAuth consent, and Drive authorization are user authorization actions. If Colab or Drive prompts for final approval, the user must click it manually.
- Browser automation is useful for opening the notebook and non-auth UI operations, but should not be treated as proof that GPU code ran. Runtime evidence must come from notebook outputs such as `nvidia-smi`, CUDA extension import checks, training logs, and saved verification files.

### Dataset staging risk

- ZS601 Drive data is treated as read-only input.
- The notebook copies the unzipped dataset into a timestamped `/content/<run_id>/ZS601meetingroom_data_staged` directory.
- If the original dataset has `sparse/*.txt` instead of `sparse/0/*.txt`, the notebook copies those files into the staged `sparse/0/` folder. It does not modify the original Drive zip.


### 2026-09-11: WSL / Colab CLI probe

Failure / current blocker:
- `wsl -l -v` and `wsl -e bash -lc "command -v colab && colab version"` failed because no usable WSL Linux distribution is currently installed on this Windows host.
- Therefore the official Colab CLI route cannot be used from this machine yet.
- If unattended Colab runs are required later, install WSL2 Ubuntu, install `google-colab-cli` inside Ubuntu, and complete one manual `colab --auth=oauth2 sessions` OAuth handoff before asking Codex to run experiments.

## Next run checklist

1. Open `colab/ZS601_2DGS_ProPlus_smoke.ipynb` from the GitHub branch in Colab.
2. Select a GPU runtime, preferably T4 or better under Colab Pro+.
3. Run the notebook from top to bottom.
4. If Drive authorization appears, the user approves it manually.
5. Confirm smoke success only after `verification.json` is saved and read back from Drive.
6. Start formal COLMAP-sparse and LiDAR-initialized runs only after smoke success.
