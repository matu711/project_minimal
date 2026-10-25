#!/usr/bin/env python3
"""Compatibility entry point; use sh scripts/convert.sh."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversion.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
