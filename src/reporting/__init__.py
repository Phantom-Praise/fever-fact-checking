"""Report generation, kept separate from retrieval and model inference.

Run these tools from the repository root with
``.venv/bin/python -m src.reporting.<module>``. Existing experiment modules
retain their original flat imports so frozen experiment sources stay intact.
"""
from pathlib import Path
import sys

_source_dir = str(Path(__file__).resolve().parents[1])
if _source_dir not in sys.path:
    sys.path.insert(0, _source_dir)
