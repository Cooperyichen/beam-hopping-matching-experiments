"""Compatibility path for the package now maintained under src/."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[3] / "src/synthetic_traffic_validation")]
__version__ = "0.1.0"
