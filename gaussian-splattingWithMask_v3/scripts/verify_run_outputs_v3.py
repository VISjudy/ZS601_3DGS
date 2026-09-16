#!/usr/bin/env python3
"""Verify the standardized output contract of a completed v3 run."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def csv_rows(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--iterations",type=int,default=150000)
    p.add_argument("--val-interval",type=int,default=5000)
    p.add_argument("--val-cameras",type=int,default=10)
    a=p.parse_args(); run=a.run
    expected=list(range(0,a.iterations+1,a.val_interval))
    if expected[-1]!=a.iterations: expected.append(a.iterations)
    errors=[]
    cfg=json.loads((run/"run_config.json").read_text(encoding="utf-8"))
    val=csv_rows(run/"val_metrics.csv")
    actual=sorted({int(r["iteration"]) for r in val if r.get("image_name")=="MEAN"})
    if actual!=expected: errors.append({"val_iterations":{"expected":expected,"actual":actual}})
    for it in expected:
        folder=run/"val_v3"/f"iteration_{it:06d}"
        if not folder.is_dir():
            errors.append({"missing_val_dir":str(folder)}); continue
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if len(manifest.get("cameras",[]))!=a.val_cameras:
            errors.append({"camera_count":{"iteration":it,"actual":len(manifest.get("cameras",[]))}})
        if list(folder.glob("*_geometry.npz")):
            errors.append({"unexpected_geometry_npz":str(folder)})
    if a.iterations==150000:
        for it in (50000,100000,150000):
            for path in (run/"checkpoints"/f"iteration_{it}.pth",
                         run/"point_cloud"/f"iteration_{it}"/"point_cloud.ply"):
                if not path.is_file(): errors.append({"missing_checkpoint_artifact":str(path)})
        final=run/"test_final"/"iteration_150000"
        for name in ("test_metrics_per_camera.csv","test_summary.json"):
            if not (final/name).is_file(): errors.append({"missing_final_test":str(final/name)})
        if len(list(final.glob("worst_*_rgb.png")))!=10:
            errors.append({"worst10_rgb_count":len(list(final.glob("worst_*_rgb.png")))})
        for name in ("results_table.csv","results_table.md","results_table.tex","experiment_summary.md"):
            if not (run/name).is_file(): errors.append({"missing_summary_artifact":name})
    report={"run":str(run.resolve()),"config_experiment":cfg.get("experiment"),
            "expected_val_timepoints":len(expected),"errors":errors,"verified":not errors}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)

if __name__=="__main__":
    main()
