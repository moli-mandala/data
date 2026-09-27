"""Installed full Sansi stage; no database construction."""
import csv
import io
import json
from test_grierson_sansi_full_stage import DATA, PACKAGE, source


def test_installed_source_equals_reviewed_proposal():
    rows,audit=source.generate()
    path=DATA/'data/other/forms/20260925-grierson-sansi.csv'
    assert rows==list(csv.reader(path.open()))
    assert audit==[json.loads(x) for x in (PACKAGE/'audit.jsonl').read_text().splitlines()]
    assert len(rows)==1943
    continuity=json.loads((PACKAGE/'independent-full-audit-20260926-pass1-continuity.json').read_text())
    assert continuity['csv_byte_identical'] and len(continuity['changed_units'])==27


def test_scoped_parser_preserves_all_full_source_keys():
    import make_cldf
    path=DATA/'data/other/forms/20260925-grierson-sansi.csv'
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(path),errors,name='20260925-grierson-sansi')
    assert not errors.getvalue()
    original={r[10]:r for r in csv.reader(path.open())}
    assert len(parsed)==stats['converted']==1943
    assert {r.entry_key for r in parsed}==set(original)
    assert all(r.old_form==original[r.entry_key][2] and r.lang=='Sansi' for r in parsed)
