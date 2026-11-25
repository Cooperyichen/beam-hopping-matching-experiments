"""Content fingerprints, protocol details and source inventory."""
import hashlib
from pathlib import Path
from .resources import write_json


def record(root):
    source=Path(__file__).parent
    files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(source.rglob('*.py')) if 'runs' not in p.parts}
    write_json(Path(root)/'source_manifest.json',dict(source_sha256=files,
        original_scripts='pareto_old.py, gen_onoff_pareto.py, farima.py, farima_f.py supplied by the user',
        implementation_notes=['GPU interval integration adapts the stationary residual-life and interval-bin approach.',
          'FARIMA is reimplemented using Python profile Whittle fitting, not the original R likelihood.',
          'Trust interpolation is an empirical adaptation; no borrowed online matching competitive ratio is asserted.'],
        fixed_protocol=dict(cells=8,slots=12,slot_duration_s=.002,cycles_per_snapshot_experiment=24,
          monte_carlo_draws_per_cycle=512,operational_cycles=120,seed=20261026,
          calibration_only_parameter_selection=True,
          correction_note='Development checks may be repeated after correctness fixes; final test outcomes do not select parameters.')))


if __name__=='__main__':
    import sys
    record(sys.argv[1])
