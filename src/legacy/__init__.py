"""Preserved Part 1 prototypes, outside the current execution path.

These files are historical source, not the selected research configuration.
Some training scripts overwrite their original output paths. Do not run them
against existing results without reviewing their settings and output locations.
"""
from pathlib import Path
import sys

_legacy_dir = str(Path(__file__).resolve().parent)
if _legacy_dir not in sys.path:
    sys.path.insert(0, _legacy_dir)
