import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def test_current_guides_have_valid_relative_links():
    guides = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md', ROOT / 'results/README.md']
    guides += [p for p in (ROOT / 'docs').glob('*.md') if p.name != 'PROJECT_JOURNAL.md']
    for guide in guides:
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', guide.read_text()):
            if raw.startswith(('https:', 'http:', '#', 'mailto:')):
                continue
            target = unquote(raw.split('#')[0].strip('<>'))
            assert (guide.parent / target).exists(), (guide.relative_to(ROOT), raw)


def test_current_text_is_english():
    chinese = re.compile('[\u3400-\u9fff]')
    for base in ['src', 'docs', 'tests', 'examples']:
        for path in (ROOT / base).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.md', '.json', '.toml'):
                assert not chinese.search(path.read_text()), str(path.relative_to(ROOT))
    assert not chinese.search((ROOT / 'README.md').read_text())


def test_examples_run_without_creating_repository_outputs():
    matching = subprocess.run([sys.executable, 'examples/matching_quickstart.py'], cwd=ROOT,
                              capture_output=True, text=True, check=True)
    assert '198.0 bits' in matching.stdout
    deficit = subprocess.run([sys.executable, 'examples/deficit_quickstart.py'], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    assert deficit.stdout.count('Cycle') == 4


def test_old_module_names_resolve_to_current_implementation():
    legacy = ROOT / 'beam_hopping_experiment_artifacts/code'
    code = (
        f"import sys; sys.path.insert(0, {str(legacy)!r}); "
        "import single_slot_matching as old; "
        "from beam_hopping.matching import single_slot as current; "
        "assert old is current; "
        "import synthetic_traffic_validation.deficit_matching as deficit; "
        "assert hasattr(deficit, 'DeficitMatchingController')"
    )
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_layout_archive_preserves_scientific_evidence():
    import hashlib
    manifest = json.loads((ROOT / 'docs/validation/layout_manifest_20261125.json').read_text())
    for entry in manifest['files']:
        if entry['preserve_bytes']:
            path = ROOT / entry['new_path']
            assert path.is_file(), entry
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['before_sha256'], entry
