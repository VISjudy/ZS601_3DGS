#!/usr/bin/env python3
"""Capture a reproducible Python/GPU/CUDA environment record."""
from __future__ import annotations
import argparse, importlib, importlib.metadata, json, platform, subprocess, sys
from pathlib import Path


def command(args, cwd=None):
    try:
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return type("Result", (), {"returncode": 127, "stdout": "", "stderr": repr(exc)})()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--require-gpu-substring", default="")
    p.add_argument("--require-module", action="append", default=[])
    p.add_argument("--package", action="append", default=[])
    a = p.parse_args(argv)
    errors = []
    smi = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    gpu_line = smi.stdout.strip()
    if a.require_gpu_substring and a.require_gpu_substring.lower() not in gpu_line.lower():
        errors.append(f"GPU does not contain {a.require_gpu_substring!r}: {gpu_line or smi.stderr.strip()}")
    torch_info = {"available": False}
    try:
        import torch
        torch_info = {
            "available": True, "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info["error"] = repr(exc)
    modules = {}
    for name in a.require_module:
        try:
            importlib.import_module(name); modules[name] = "ok"
        except Exception as exc:
            modules[name] = repr(exc); errors.append(f"module {name} failed: {exc!r}")
    packages = {}
    for name in a.package:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    git = command(["git", "-C", str(a.project_root), "rev-parse", "HEAD"])
    dirty = command(["git", "-C", str(a.project_root), "status", "--porcelain"])
    record = {
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(), "nvidia_smi_query": gpu_line,
        "nvidia_smi_error": smi.stderr.strip() or None, "torch": torch_info,
        "modules": modules, "packages": packages,
        "git": {"commit": git.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())},
        "errors": errors, "verified": not errors,
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    print(payload)
    if a.output:
        if a.output.exists() and not a.overwrite:
            raise FileExistsError(f"Refusing to overwrite {a.output}")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(payload + "\n", encoding="utf-8")
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()
