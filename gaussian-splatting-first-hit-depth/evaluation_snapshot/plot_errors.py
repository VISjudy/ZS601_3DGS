"""Publication-style error figures for the five-method v008 CPU evaluation.

All methods use the same fixed colour limits. Plots expose missing data and colour
saturation; they never change the numeric metrics or fill invalid pixels.
"""
from pathlib import Path
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "python-deps"))
sys.path.insert(0, str(PROJECT / ".runtime/synthetic-dataset"))


def render_figures(evaluation, figures, representatives, nn_max_mm=30.0, z_limit_mm=150.0):
    # Delay Matplotlib import/config creation until rendering is explicitly run.
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "matplotlib-config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    evaluation, figures = Path(evaluation), Path(figures)
    reuse_empty_directory = figures.exists()
    if reuse_empty_directory:
        expected_children = {"nn_heatmaps", "signed_mesh_Z", "representative_views"}
        if any(p.is_file() for p in figures.rglob("*")) or any(p.name not in expected_children for p in figures.iterdir()):
            raise FileExistsError(f"Refusing to replace populated figure directory: {figures}")
    report = json.loads((evaluation / "geometry_metrics.json").read_text(encoding="utf-8"))
    cameras = report["protocol"]["cameras"]
    methods = report["protocol"]["methods"]
    camera_map = {v["image_id"]: v for v in cameras}
    if not set(representatives).issubset(camera_map):
        raise ValueError("Representative view is outside the fixed ten cameras")
    if nn_max_mm <= 0 or z_limit_mm <= 0:
        raise ValueError("Colour limits must be positive")
    figures.mkdir(parents=True, exist_ok=reuse_empty_directory)
    for child in ("nn_heatmaps", "signed_mesh_Z", "representative_views"):
        (figures / child).mkdir(exist_ok=reuse_empty_directory)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.dpi": 120, "savefig.dpi": 160, "pdf.fonttype": 42})
    nn_cmap = matplotlib.colormaps["viridis"].copy()
    z_cmap = matplotlib.colormaps["RdBu_r"].copy()
    for cmap in (nn_cmap, z_cmap):
        cmap.set_bad("#d9d9d9")
    records = []

    def finish(fig, path, pdf=False):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        frame = fig.bbox
        checked_text = 0
        failures = []
        for artist in fig.findobj(matplotlib.text.Text):
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            extent = artist.get_window_extent(renderer)
            checked_text += 1
            if extent.x0 < frame.x0 - 2 or extent.y0 < frame.y0 - 2 or extent.x1 > frame.x1 + 2 or extent.y1 > frame.y1 + 2:
                failures.append(artist.get_text())
        if failures:
            plt.close(fig)
            raise RuntimeError(f"Text extends outside figure canvas: {failures}")
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
        with Image.open(path) as image:
            size = list(image.size)
            image.verify()
        pdf_path = None
        if pdf:
            pdf_path = path.with_suffix(".pdf")
            if pdf_path.exists():
                raise FileExistsError(pdf_path)
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        return {"path": path.relative_to(evaluation).as_posix(), "PNG_size": size, "text_extents_inside_canvas": True, "text_artists_checked": checked_text, "PNG_readback": "PASS", "PDF_path": pdf_path.relative_to(evaluation).as_posix() if pdf_path else None}

    def load_maps(method, stem):
        folder = evaluation / "error_arrays" / method["id"]
        nn = np.load(folder / f"{stem}_nn_m.npy", allow_pickle=False) * 1000
        z = np.load(folder / f"{stem}_signed_mesh_Z_m.npy", allow_pickle=False) * 1000
        return np.ma.masked_invalid(nn), np.ma.masked_invalid(z)

    for camera in cameras:
        stem = Path(camera["name"]).stem
        for method in methods:
            nn, signed = load_maps(method, stem)
            for kind, values, cmap, low, high, legend, folder in (
                ("nn", nn, nn_cmap, 0, nn_max_mm, "Source-cloud NN (mm)", "nn_heatmaps"),
                ("signed_mesh_Z", signed, z_cmap, -z_limit_mm, z_limit_mm, "Z error (mm); +behind / -front", "signed_mesh_Z"),
            ):
                fig, ax = plt.subplots(figsize=(5.1, 8.0), layout="constrained")
                artist = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
                ax.set_title(f"{method['plot_label']}\nView {stem}")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_axis_off()
                fig.colorbar(artist, ax=ax, fraction=0.07, pad=0.03, extend="max" if kind == "nn" else "both", label=legend)
                fig.supxlabel("Gray = not evaluated; colour limits are shared across all five methods", fontsize=8)
                path = figures / folder / f"{stem}_{method['id']}.png"
                record = finish(fig, path)
                data = values.compressed()
                record.update(method=method["id"], image_id=camera["image_id"], kind=kind, colour_limits_mm=[low, high], eligible_pixels=len(data), below_colour_limit_count=int((data < low).sum()), above_colour_limit_count=int((data > high).sum()))
                records.append(record)
        print(f"HEATMAPS_COMPLETE view={camera['image_id']}", flush=True)

    for image_id in representatives:
        camera = camera_map[image_id]
        stem = Path(camera["name"]).stem
        fig, axes = plt.subplots(3, len(methods), figsize=(16.5, 16.8), layout="constrained")
        nn_artist = z_artist = None
        for column, method in enumerate(methods):
            with Image.open(Path(method["input_root"]) / "images" / camera["name"]) as image:
                rgb = np.asarray(image).copy()
            nn, signed = load_maps(method, stem)
            # Explicitly shade invalid depth pixels even if a renderer emitted RGB there.
            rgb[~np.isfinite(nn.data)] = 217
            axes[0, column].imshow(rgb, interpolation="nearest")
            nn_artist = axes[1, column].imshow(nn, cmap=nn_cmap, vmin=0, vmax=nn_max_mm, interpolation="nearest")
            z_artist = axes[2, column].imshow(signed, cmap=z_cmap, vmin=-z_limit_mm, vmax=z_limit_mm, interpolation="nearest")
            axes[0, column].set_title(method["plot_label"], fontsize=11, pad=7)
            for row in range(3):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                for spine in axes[row, column].spines.values():
                    spine.set_visible(False)
        axes[0, 0].set_ylabel("Saved RGB on valid depth pixels", fontsize=11, labelpad=12)
        axes[1, 0].set_ylabel("One-sided NN to source cloud", fontsize=11, labelpad=12)
        axes[2, 0].set_ylabel("Signed mesh first-surface Z error", fontsize=11, labelpad=12)
        fig.colorbar(nn_artist, ax=list(axes[1, :]), fraction=0.016, pad=0.012, extend="max", label="NN (mm)")
        fig.colorbar(z_artist, ax=list(axes[2, :]), fraction=0.016, pad=0.012, extend="both", label="Z error (mm)")
        fig.suptitle(f"View {stem}: five depth methods, saved-PNG backprojection\nGray = not evaluated. NN measures source proximity, not visibility.\nPositive Z error = behind first surface; negative = in front. Shared colour limits.", fontsize=13)
        records.append(finish(fig, figures / "representative_views" / f"view_{stem}_five_methods.png", pdf=True))
        print(f"SCIENTIFIC_COMPARISON_COMPLETE view={image_id}", flush=True)
    manifest = {"status": "PASS_FILES_AND_TEXT_EXTENTS_PENDING_HUMAN_VISUAL_REVIEW", "representative_views": representatives, "nn_max_mm": nn_max_mm, "signed_Z_limits_mm": [-z_limit_mm, z_limit_mm], "colour_limit_selection": "Fixed display parameters; no values excluded from numeric metrics", "invalid_colour": "#d9d9d9", "method_mask": "Each method's primary valid depth mask; signed mesh Z additionally requires reference validity", "records": records}
    with (figures / "figure_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=PROJECT / "3dgsResult/zs601-firsthit-depth-v008/evaluation")
    parser.add_argument("--figures", type=Path, help="Fresh figure directory under evaluation; default evaluation/figures")
    parser.add_argument("--representatives", nargs=3, type=int, default=[3202, 3376, 3481])
    parser.add_argument("--nn-max-mm", type=float, default=30)
    parser.add_argument("--signed-z-limit-mm", type=float, default=150)
    args = parser.parse_args()
    figures = args.figures or args.evaluation / "figures"
    if not figures.resolve().is_relative_to(args.evaluation.resolve()):
        raise ValueError("Figure output must stay under this evaluation directory")
    render_figures(args.evaluation, figures, args.representatives, args.nn_max_mm, args.signed_z_limit_mm)


if __name__ == "__main__":
    main()
