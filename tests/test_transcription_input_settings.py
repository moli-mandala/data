"""Configured pronunciation input must not overwrite bibliographic spelling."""
import csv
import io
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import source_meta


def test_configured_pronunciation_preserves_original_and_hyphens(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    import make_cldf
    settings = tmp_path / 'sample.yaml'
    settings.write_text('''defaults:
  transcription:
    profile: bhattacharya-ollari
    input: phonemic
    preserve_hyphens: true
sources:
  test-ollari:
    forms:
      split_alternates: false
''')
    meta = source_meta.SourceMeta([settings])
    monkeypatch.setattr(source_meta, 'load', lambda: meta)
    folder = tmp_path / 'other/forms'
    folder.mkdir(parents=True)
    file = folder / 'sample.csv'
    with file.open('w') as out:
        writer = csv.writer(out)
        for i, (form, phonemic) in enumerate([('jir er-', 'dzir er-'), ('tanḍ jir', 'tanḍ zir'), ('kã·j-', ''), ('neliṅ', '')]):
            writer.writerow(['OllariGadaba', '', form, 'test', '', phonemic, '', 'test-ollari[p. 60]', '', '', f'test:{i}', '', '', '', ''])
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(file), errors)
    assert errors.getvalue() == '' and stats['converted'] == 4
    assert [r.form for r in rows] == ['ʣir er-', 'tanḍ zir', 'kā̃j-', 'neliŋ']
    assert [r.old_form for r in rows] == ['jir er-', 'tanḍ jir', 'kã·j-', 'neliṅ']
    assert [r.ipa for r in rows] == ['dzir er-', 'tanḍ zir', '', '']
    # Default behavior remains source-spelling conversion and boundary stripping.
    meta.files['sample']['transcription'][0].pop('input')
    meta.files['sample']['transcription'][0].pop('preserve_hyphens')
    rows, _ = make_cldf.parse_file(str(file), io.StringIO())
    assert [r.form for r in rows] == ['jir er', 'tanḍ jir', 'kā̃j', 'neliŋ']


@pytest.mark.parametrize('rule', [{'input': 'typo'}, {'preserve_hyphens': 'false'}])
def test_reject_invalid_conversion_options(rule):
    with pytest.raises(source_meta.SourceMetaError):
        source_meta._rules(rule, 'test')
