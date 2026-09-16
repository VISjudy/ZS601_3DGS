#!/usr/bin/env python3
"""Render the standard val10 diagnostics from an existing v3 point-cloud PLY.

This script does not train or mutate the source run. It reconstructs the
cameras and Gaussian model from run_config.json and writes diagnostics to a new
output directory.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config",type=Path,required=True)
    parser.add_argument("--model-ply",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--iteration",type=int,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    cfg=json.loads(args.run_config.read_text(encoding="utf-8"))
    cfg["model_path"]=str(args.output)
    a=SimpleNamespace(**cfg)
    from data_v3 import load_data
    from scene.gaussian_model import GaussianModel
    from render_v3 import export_val
    _,_,_,val_cameras,_,_,_,_=load_data(a)
    g=GaussianModel(a.sh_degree,"default",False,-10.,False)
    g.load_ply(str(args.model_ply),False)
    pipe=SimpleNamespace(
        debug=bool(cfg.get("debug",False)),
        antialiasing=bool(cfg.get("antialiasing",False)),
        convert_SHs_python=bool(cfg.get("convert_SHs_python",False)),
        compute_cov3D_python=bool(cfg.get("compute_cov3D_python",False)),
    )
    args.output.mkdir(parents=True,exist_ok=False)
    print("[DIAGNOSTICS]",json.dumps({
        "iteration":args.iteration,"model_ply":str(args.model_ply),
        "output":str(args.output),"camera_count":len(val_cameras),
        "rgb":a.val_rgb,"depth":a.val_depth,"normal":a.val_normal,
        "ellipsoid_1sigma":a.val_ellipsoids,"raw_npz":a.val_npz},indent=2))
    export_val(a,args.iteration,val_cameras,g,pipe)

if __name__=="__main__":
    main()
