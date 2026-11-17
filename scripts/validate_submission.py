#!/usr/bin/env python3
"""Validate take-home submission layout (generated from submission_schema.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takehome.validation import validate_tree  # noqa: E402


def main() -> int:
    report = validate_tree(ROOT)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
