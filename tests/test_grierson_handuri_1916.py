"""Full-source Handuri scope, reconciliation and conversion regressions."""
import csv,importlib.util,io,json
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/grierson_handuri_1916'
CSV=DATA/'data/other/forms/20260925-grierson-handuri.csv'
PROFILE=DATA/'conversion/grierson-handuri-1916.txt'
spec=importlib.util.spec_from_file_location('handuri',PACKAGE/'import_source.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
def installed():return list(csv.reader(CSV.open()))
def audited():return [json.loads(x) for x in (PACKAGE/'audit.jsonl').read_text().splitlines()]
def test_full_scope_and_legacy_keys():
    rows,audit=source.generate()
    assert rows==installed() and audit==audited()
    assert len(rows)==587 and len(audit)==670
    assert Counter(a.get('section','table') for a in audit)=={'table':241,'grammar':86,'specimen':343}
    assert sum(bool(a.get('reuse_entry_keys')) for a in audit)==100
    assert Counter(a['status'] for a in audit)=={'ingested':557,'ingested_uncertain':7,'source_blank':2,'inventory_only_morphological_ending':4,'reused_same_source_attestation':100}
    keys={r[10] for r in rows};assert len(keys)==587
    assert {f'grierson1916handuri:p628:item:{i}' for i in [1,2,3,4,6,7,8,10,11,13]}<=keys

def test_independent_twenty_still_matches():
    report=json.loads((PACKAGE/'independent-audit-20260926-pass1.json').read_text())
    assert report['result']=='pass' and report['entries_with_errors']==0 and len(report['sample'])==20
    before={r[10]:r for r in csv.reader((PACKAGE/'full-preview.csv').open())};after={r[10]:r for r in installed()}
    for cell in report['sample']:
        for key in cell['entry_keys']+cell.get('reuse_entry_keys',[]):
            assert [after[key][i] for i in (0,2,3,7,10)]==[before[key][i] for i in (0,2,3,7,10)]

def test_citations_grammar_and_column_repair():
    rows={r[10]:r for r in installed()}
    for a in audited():
        for key in a.get('reuse_entry_keys',[]):assert f"p. {a['page']}, specimen 4, line {a['line']}, aligned unit {a['aligned_unit']}" in rows[key][7]
    def table(p,i):return rows[f'grierson1916handuri:p{p}:item:{i}']
    assert {'pron','pl','gen','first-person'}<=set(table(628,18)[14].split())
    assert {'f','pl'}<=set(table(638,141)[14].split())
    assert {'verb','2pl','pret'}<=set(table(644,215)[14].split())
    assert 'participle' in table(644,219)[14]
    assert table(632,56)[2]=='Chhōṭī' and table(636,133)[2]=='(Tĕs-tē) kharā'
    assert all(not r[13] and source.DIALECT in r[14] for r in rows.values())
    assert all('not installed' not in r[6] and 'pending review' not in r[6] for r in rows.values())

def test_profile_metadata_and_scoped_parse():
    import make_cldf,profile_policy,source_meta,tags
    assert source_meta.SourceMeta().transcription(source.SOURCE,CSV,'Hinduri')[0]=='grierson-handuri-1916'
    assert 'grierson-handuri-1916' not in profile_policy.audit({})
    tok=Tokenizer(str(PROFILE))
    for row in installed():
        assert '�' not in tok(row[2],column='IPA')
        assert set(t for t in row[14].split() if not t.startswith('dialect:')) <= tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-grierson-handuri')
    assert not errors.getvalue()
    assert len(parsed)==stats['converted']==587
    raw={r[10]:r for r in installed()}
    assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
