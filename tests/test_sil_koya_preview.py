"""A preview must leave every canonical ingestion artifact untouched."""
import csv
import importlib.util
import json
from pathlib import Path


def test_preview_does_not_write_canonical_files(tmp_path, monkeypatch):
    package = Path(__file__).parents[1] / 'data/other/forms/raw_data/sil_koya_2021'
    spec = importlib.util.spec_from_file_location('koya_preview_importer', package / 'import_koya.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'REPO', tmp_path)
    canonical = []
    for name in ['INSTALLED', 'AUDIT', 'MANIFEST']:
        path = tmp_path / name
        path.write_bytes(b'canonical sentinel\n')
        monkeypatch.setattr(module, name, path)
        canonical.append(path)
    forms = [['Gondi', '', 'test']]
    audit = [{'Status': 'installed'}]
    metadata = {'preview': True}
    module.write(forms, audit, metadata, install=False)
    assert all(p.read_bytes() == b'canonical sentinel\n' for p in canonical)
    preview = tmp_path / 'tmp/20260828-sil-koya-preview.csv'
    assert list(csv.reader(preview.open())) == forms
    assert list(csv.DictReader(preview.with_name('20260828-sil-koya-preview-audit.csv').open()))[0]['Status'] == 'installed'
    assert json.loads(preview.with_name('20260828-sil-koya-preview-manifest.json').read_text()) == metadata
