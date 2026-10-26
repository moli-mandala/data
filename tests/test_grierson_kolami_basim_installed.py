"""Whole chapter source-stage regressions; compiled identity/build gates are separate."""
import csv,hashlib,io,json,os
from pathlib import Path

DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/grierson_kolami_basim_1906'
CANONICAL=DATA/'data/other/forms/20260927-grierson-kolami-basim.csv'
FINAL=PACKAGE/'integration/revision6'

def rows(path):
    with path.open(newline='') as stream:return list(csv.reader(stream))

def test_installed_whole_scope_and_accepted_audit_preserve_physical_reuses():
    rr=rows(CANONICAL)
    assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest()=='747f12ae281fbb7b6054ec3a22614ff962adc5e283bbe26dd79a2c6501e1d223'
    assert rr==rows(FINAL/'combined-review-candidate.csv')
    assert len(rr)==len({r[10] for r in rr})==827
    audit=[json.loads(s) for s in (PACKAGE/'installed-audit.jsonl').read_text().splitlines()]
    raw=[json.loads(s) for s in (FINAL/'combined-review-audit.jsonl').read_text().splitlines()]
    assert len(audit)==len(raw)==1570
    for a,b in zip(audit,raw):
        assert a['accepted_for_installation'] is True
        assert all(a[k]==v for k,v in b.items() if k!='accepted_for_installation')
    new={r[10] for r in rr}
    represented={k for a in audit for k in a['entry_keys']}
    reused=represented-new
    assert reused=={'grierson1906kui:p566:outside-comparand:06',
                    'grierson1906kui:p623:outside-comparand:15',
                    'grierson1906naiki:p623:genitive-comparison:naiki'}
    assert sum(a['kind']=='physical_atom_accounting' for a in audit)==561
    assert sum(a['kind']=='atomic' for a in audit)==560
    assert sum(a['kind']=='expression' for a in audit)==51

def test_whole_source_object_scope_suffix_notes_and_closed_ordered_graph():
    rr=rows(CANONICAL);by={r[10]:r for r in rr}
    assert not any('obj' in r[14].split() for r in rr)
    plural=by['grierson1906kolamibasim:p562:noun-plural:1:independent:w1']
    case=by['grierson1906kolamibasim:p562:accusative-dative:1:independent:w1']
    assert (plural[2],plural[6])==('l','Grierson identifies l as the plural suffix.')
    assert (case[2],case[6])==('n','Grierson identifies n as the accusative and dative suffix.')
    assert sum(bool(r[11]) for r in rr)==15
    assert sum(bool(r[13]) for r in rr)==2
    assert sum(len(r[13].split('|')) for r in rr if r[13])==4
    adjacency={r[10]:[k for k in [r[11],*r[13].split('|')] if k] for r in rr}
    assert all(p in by for parents in adjacency.values() for p in parents)
    def visit(k,path):
        assert k not in path
        for parent in adjacency[k]:visit(parent,path|{k})
    for key in by:visit(key,set())

def test_actual_canonical_parser_preserves_all_source_fields_and_initial_identity():
    import make_cldf,source_meta
    fields=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
    old_cwd=os.getcwd()
    try:
        os.chdir(DATA)
        meta=source_meta.load();source='grierson1906kolamibasim'
        assert meta.flag(source,'forms','split_alternates',True) is False
        assert meta.flag(source,'gloss','grammar_tags',True) is False
        assert meta.file_flag(CANONICAL.name,'identity','dedupe_by_entry_key',False) is True
        errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CANONICAL),errors,name=CANONICAL.stem)
        assert not errors.getvalue() and len(parsed)==stats['converted']==827
        assert [[getattr(row,f) for f in fields] for row in parsed]==rows(CANONICAL)
        assert len({make_cldf.source_entry_dedupe_key(row,meta) for row in parsed})==827
    finally:os.chdir(old_cwd)
