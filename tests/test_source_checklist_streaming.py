"""Bounded audit rendering preserves the prior CSV payload exactly."""
import csv
import gzip
import hashlib
import io
from types import SimpleNamespace

import audit_source_ingestions as audit


def test_streamed_audit_matches_previous_payload(tmp_path, monkeypatch):
    rows = [
        ['L', '1', 'ā, word', 'a\nmeaning', '', '', '', 's[1]', '', '', 'key1'],
        ['L', '', '', 'blank'],
        ['L', '', '�', 'bad'],
        [],
    ]
    path = tmp_path / 'input.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows(rows)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    unit = SimpleNamespace(id='tiny', installed_file='input.csv')
    expected = io.StringIO(newline='')
    writer = csv.writer(expected, lineterminator='\n')
    writer.writerow(['Unit_ID', 'Installed_File', 'Row_Number', 'Status', 'Reason',
                     'Language_ID', 'Parameter_ID', 'Form', 'Gloss', 'Source',
                     'Entry_Key', 'Row_SHA256'])
    for number, row in enumerate(rows, 1):
        form = row[2] if len(row) > 2 else ''
        reason = 'blank form' if not form.strip() else 'replacement character' if '�' in form else ''
        writer.writerow(['tiny', 'input.csv', number, 'excluded' if reason else 'installed', reason,
                         row[0] if row else '', row[1] if len(row)>1 else '', form,
                         row[3] if len(row)>3 else '', row[7] if len(row)>7 else '',
                         row[10] if len(row)>10 else '', hashlib.sha256('\x1f'.join(row).encode()).hexdigest()])
    result = audit.render_installed_record_audit([unit])
    assert gzip.decompress(result) == expected.getvalue().encode('utf-8')
    assert result == audit.render_installed_record_audit([unit])
    assert result[4:8] == b'\0' * 4


def test_compiled_index_retains_only_consumed_fields(tmp_path, monkeypatch):
    (tmp_path / 'cldf').mkdir()
    with (tmp_path / 'cldf/forms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['ID','Language_ID','Tags','Source','Form'])
        writer.writeheader()
        writer.writerow(dict(ID='f1', Language_ID='L', Tags='n dialect:L-site', Source='a[1];b[2]', Form='large discarded payload'))
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    result = audit.compiled_source_rows()
    assert result == {'a':[dict(ID='f1',Language_ID='L',Tags='n dialect:L-site')],
                      'b':[dict(ID='f1',Language_ID='L',Tags='n dialect:L-site')]}
    assert result['a'][0] is result['b'][0]


def test_profiles_follow_actual_yaml_row_routing(tmp_path, monkeypatch):
    import source_meta
    conversion = tmp_path / 'conversion'
    conversion.mkdir()
    for name in ('default', 'specific'):
        (conversion / f'{name}.txt').write_text('Grapheme\tIPA\na\ta\n')
    yaml = tmp_path / 'fixture.yaml'
    yaml.write_text('''file: fixture.csv
defaults:
  transcription:
    - profile: default
sources:
  cited:
    transcription:
      - languages: [Special]
        profile: specific
      - languages: [Literal]
        convert: false
''')
    meta = source_meta.SourceMeta([yaml])
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit.source_meta, 'load', lambda: meta)
    path = tmp_path / 'fixture.csv'
    def row(lang): return [lang, '', 'a', '', '', '', '', 'cited[1]']
    # Same API and precedence used by make_cldf.parse_file.
    assert meta.transcription('cited', path, 'Special') == ('specific', True)
    assert meta.transcription('cited', path, 'Other') == ('default', True)
    assert meta.transcription('cited', path, 'Literal') == (None, False)
    assert audit.infer_profiles(path, 'fixture', [row('Special')]) == ['conversion/specific.txt']
    assert audit.infer_profiles(path, 'fixture', [row('Other')]) == ['conversion/default.txt']
    assert audit.infer_profiles(path, 'fixture', [row('Literal')]) == ['fixture.yaml']
    assert audit.infer_profiles(path, 'fixture', [row('Other'), row('Special'), row('Literal')]) == ['conversion/default.txt', 'conversion/specific.txt', 'fixture.yaml']
