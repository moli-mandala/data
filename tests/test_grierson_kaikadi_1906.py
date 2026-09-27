"""Installed complete Kaikadi source-stage checks, without a database build."""
import csv,hashlib,io,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_kaikadi_1906'
DATA=P.parents[4]
CSV=DATA/'data/other/forms/20260925-grierson-kaikadi.csv'

def test_installed_source_inputs_match_full_independent_approval():
    report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
    assert report['status']=='passed' and report['material_errors']==0
    targets={'proposal.csv':CSV,'proposal-audit.jsonl':P/'audit.jsonl','proposal-profile.txt':DATA/'conversion/grierson-kaikadi-1906.txt','20260925-grierson-kaikadi.yaml':DATA/'data/other/forms/20260925-grierson-kaikadi.yaml'}
    for name,path in targets.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==report['hashes'][name]
    rows=list(csv.reader(CSV.open()));assert len(rows)==841
    registry={r['Tag']:r for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    for tag in {t for row in rows for t in row[14].split()if t.startswith('dialect:')}:
        assert registry[tag]['Language_ID']=='Kaikadi'
        assert not registry[tag]['Latitude'] and not registry[tag]['Longitude']
    legacy=json.loads((P/'legacy164-preservation-manifest.json').read_text())
    assert len(legacy['entry_keys'])==164 and set(legacy['entry_keys'])<={r[10]for r in rows}

def test_actual_installed_profile_route_parser_and_bibliography():
    import make_cldf,source_meta,make_refs,pybtex.database
    assert source_meta.SourceMeta().transcription('grierson1906kaikadi',CSV,'Kaikadi')[0]=='grierson-kaikadi-1906'
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-grierson-kaikadi')
    rows=list(csv.reader(CSV.open()))
    assert len(parsed)==stats['converted']==841 and not errors.getvalue()
    assert {r.entry_key:r.old_form for r in parsed}=={r[10]:r[2]for r in rows}
    assert all(not r.ipa and not r.native for r in parsed)
    bib=pybtex.database.parse_file(str(DATA/'cldf/sources.bib'))
    assert all(set(make_refs.source_ids(r[7]))<=set(bib.entries)for r in rows)
