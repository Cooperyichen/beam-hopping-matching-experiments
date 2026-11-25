"""Compatibility launcher; see docs/history.md."""
import sys
from _compat import load

_implementation = load('beam_hopping.visualization.presentation')
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
