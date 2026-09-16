#!/usr/bin/env python3
"""Validate a processed_v3 directory and emit machine/human-readable reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess_v3 import validate_processed  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ZS601 processed_v3 artifacts")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace prior validation reports")
    args = parser.parse_args(argv)
    processed = args.processed.resolve()
    if not processed.exists():
        raise SystemExit(f"Processed directory does not exist: {processed}")
    report, errors = validate_processed(processed, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
