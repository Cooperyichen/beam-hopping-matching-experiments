"""Discover features without importing optional scientific dependencies."""
import argparse
import importlib
import importlib.metadata
import json
import sys

from .catalog import entries

COMMANDS = {
    'single-slot': ('beam_hopping.matching.single_slot', 'optimization', 'Exact single slot LP and binary comparison'),
    'benchmarks': ('beam_hopping.matching.heuristics', 'optimization', 'Greedy and genetic algorithm comparisons'),
    'online': ('beam_hopping.scheduling.online', 'optimization', 'Earlier synthetic horizon prototype'),
    'learning': ('beam_hopping.learning.reinforcement', 'learning', 'Queue based reinforcement learning prototype'),
    'forecast': ('synthetic_traffic_validation.cli', 'experiments', 'Historical data and forecasting pipeline'),
    'forecast-revision': ('synthetic_traffic_validation.revision_code.revision', 'experiments', 'PDF revision protocol'),
    'deficit': ('synthetic_traffic_validation.deficit_demo', None, 'Toy or consecutive traffic deficit replay'),
    'verify-evidence': ('synthetic_traffic_validation.verify_evidence', None, 'Read-only historical archive checks'),
    'plots': ('beam_hopping.visualization.presentation', 'plots', 'Historical figure generator; writes figures'),
}


def doctor(argv):
    parser = argparse.ArgumentParser(prog='beam-hopping doctor')
    parser.add_argument('--check-mosek', action='store_true', help='Run a small MOSEK LP; requires a license')
    args = parser.parse_args(argv)
    versions = {}
    for distribution in ('numpy', 'scipy', 'cvxpy', 'Mosek', 'torch', 'pandas', 'matplotlib', 'pytest'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    result = {'python': sys.version.split()[0], 'packages': versions,
              'mosek_license': 'not checked; use --check-mosek'}
    status = 0
    if args.check_mosek:
        try:
            import cvxpy as cp
            variable = cp.Variable()
            problem = cp.Problem(cp.Minimize(variable), [variable >= 1])
            problem.solve(solver='MOSEK')
            if problem.status != 'optimal' or abs(float(problem.value) - 1.) > 1e-6:
                raise RuntimeError(f'unexpected status or objective: {problem.status}, {problem.value}')
            result['mosek_license'] = 'small LP passed'
        except Exception as error:
            result['mosek_license'] = f'check failed: {error}'
            status = 1
    print(json.dumps(result, indent=2))
    return status


def missing_dependency(command, extra, error):
    if command == 'deficit' and error.name == 'torch':
        extra = 'experiments'
    feature = f'.[{extra}]' if extra else '.'
    print(f"Feature '{command}' needs module '{error.name}'. Install with:\n"
          f"  python -m pip install -e '{feature}'\n"
          'See docs/getting_started.md. No solver was substituted.', file=sys.stderr)
    return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='beam-hopping', description=__doc__)
    parser.add_argument('command', nargs='?', choices=['strategies', 'doctor', *COMMANDS])
    parser.epilog = 'Use COMMAND --help for feature options. Start with docs/getting_started.md.'
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        print('\nCommands:')
        print('  strategies        List policy names and source entry points')
        print('  doctor            Report dependencies; optional MOSEK check')
        for name, (_, _, description) in COMMANDS.items():
            print(f'  {name:17} {description}')
        return 0
    command, rest = argv[0], argv[1:]
    if command == 'strategies':
        subparser = argparse.ArgumentParser(prog='beam-hopping strategies')
        subparser.add_argument('--json', action='store_true')
        args = subparser.parse_args(rest)
        if args.json:
            print(json.dumps(entries(), indent=2))
        else:
            for entry in entries():
                print(f"{entry['name']}\n  {entry['module']}:{entry['entry']}\n  {entry['summary']}")
        return 0
    if command == 'doctor':
        return doctor(rest)
    if command not in COMMANDS:
        parser.error(f'unknown command: {command}')
    module_name, extra, _ = COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        return missing_dependency(command, extra, error)
    original = sys.argv
    sys.argv = [f'beam-hopping {command}', *rest]
    try:
        return module.main() or 0
    except ModuleNotFoundError as error:
        return missing_dependency(command, extra, error)
    except (FileNotFoundError, FileExistsError) as error:
        print(f'{error}\nCheck input paths and choose a fresh output path; see docs/experiments.md.', file=sys.stderr)
        return 2
    finally:
        sys.argv = original
