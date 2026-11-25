"""Resolve historical module names without keeping duplicate implementations."""
import importlib
from pathlib import Path
import sys


def load(module):
    source = str(Path(__file__).resolve().parents[2] / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module(module)
