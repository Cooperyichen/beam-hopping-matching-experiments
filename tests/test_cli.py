import json
import subprocess
import sys

import pytest

from beam_hopping import cli
from beam_hopping.catalog import entries


def test_help_and_catalog_do_not_import_optional_dependencies():
    code = "from beam_hopping.cli import main; main(['strategies', '--json']); import sys; assert not any(k in sys.modules for k in ['torch','cvxpy','matplotlib','pandas'])"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    assert len(rows) == 16
    assert len({row['name'] for row in rows}) == 16
    assert any(row['name'] == 'certified_static_matching' for row in rows)


def test_cli_json_and_global_help(capsys):
    assert cli.main(['strategies', '--json']) == 0
    assert json.loads(capsys.readouterr().out) == entries()
    assert cli.main(['--help']) == 0
    assert 'forecast-revision' in capsys.readouterr().out


def test_missing_feature_dependency_has_actionable_error(monkeypatch, capsys):
    def absent(name):
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    monkeypatch.setattr(cli.importlib, 'import_module', absent)
    assert cli.main(['learning', '--help']) == 2
    assert ".[learning]" in capsys.readouterr().err


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit) as error:
        cli.main(['does-not-exist'])
    assert error.value.code == 2


def test_lazy_solver_import_failure_is_actionable(monkeypatch, capsys):
    from types import SimpleNamespace
    def run():
        raise ModuleNotFoundError("No module named 'cvxpy'", name='cvxpy')
    monkeypatch.setattr(cli.importlib, 'import_module', lambda _: SimpleNamespace(main=run))
    assert cli.main(['single-slot']) == 2
    assert '.[optimization]' in capsys.readouterr().err


def test_doctor_is_informational_without_solver_call(capsys):
    assert cli.main(['doctor']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mosek_license'].startswith('not checked')


def test_all_catalog_entries_resolve_from_source_ast():
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'src'
    for entry in entries():
        path = source / (entry['module'].replace('.', '/') + '.py')
        definitions = {node.name for node in ast.parse(path.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        assert entry['entry'] in definitions, entry


def test_plot_help_does_not_generate_figures(tmp_path):
    import importlib.util
    if not all(importlib.util.find_spec(name) for name in ['matplotlib', 'pandas']):
        pytest.skip('optional plot dependencies are not installed')
    result = subprocess.run([sys.executable, '-m', 'beam_hopping', 'plots', '--help'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert '--output-dir' in result.stdout
    assert not list(tmp_path.iterdir())
