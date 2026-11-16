"""Git checkout settings must never change the checksum-pinned manifest bytes."""
import hashlib
import subprocess
from pathlib import Path

import pytest

from defectops.history_provenance import _MIGRATION_SHA256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('src/defectops/history_migrations/privacy-20261116.json')


@pytest.mark.parametrize('autocrlf,eol', [('true', 'crlf'), ('false', 'lf'), ('input', 'lf')])
def test_checkout_preserves_pinned_manifest(tmp_path, autocrlf, eol):
    source, target = tmp_path / 'source', tmp_path / 'checkout'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / '.gitattributes').write_bytes((ROOT / '.gitattributes').read_bytes())
    (source / MANIFEST).parent.mkdir(parents=True)
    original = (ROOT / MANIFEST).read_bytes()
    (source / MANIFEST).write_bytes(original)
    subprocess.run(['git', '-C', str(source), '-c', 'core.autocrlf=false',
                    'add', '.gitattributes', MANIFEST.as_posix()], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm',
                    'Fixed manifest fixture'], check=True)
    subprocess.run(['git', 'clone', '-q', '--no-local', '--config', f'core.autocrlf={autocrlf}',
                    '--config', f'core.eol={eol}', str(source), str(target)], check=True)
    payload = (target / MANIFEST).read_bytes()
    assert payload == original
    assert hashlib.sha256(payload).hexdigest() == _MIGRATION_SHA256
    attr = subprocess.check_output(['git', '-C', str(target), 'check-attr', 'text', '--',
                                    MANIFEST.as_posix()], text=True)
    assert attr.strip().endswith(': text: unset')
    assert not subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain']).strip()
