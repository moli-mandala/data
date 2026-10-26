"""Actual canonical source-stage checks only; never a build or ID allocation."""
import csv,hashlib,io,json,os,sys,tempfile,unicodedata as ud
from pathlib import Path
from pybtex.database import parse_string
from pybtex import PybtexEngine
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
def main():
    approval=json.loads((HERE/'root-acceptance.json').read_text())
    assert approval['installation_approved'] is True and approval['result']=='pass'
    own={role:DATA/a['target'] for role,a in approval['assets'].items() if role in ['csv','yaml','audit','profile']}
    assert all(sha(path)==approval['assets'][role]['output_sha256'] for role,path in own.items())
    identity_before=sha(DATA/'data/form-identities.csv');assert identity_before==approval['durable_identity_sha256']
    original=os.getcwd();os.chdir(DATA);sys.path.insert(0,str(DATA))
    import make_cldf,source_meta
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    meta=source_meta.load();source='grierson1906kolamibasim'
    with own['csv'].open() as stream:expected=list(csv.reader(stream))
    assert len(expected)==len({r[10] for r in expected})==827
    assert meta.flag(source,'forms','split_alternates',True) is False
    assert meta.flag(source,'gloss','grammar_tags',True) is False
    assert meta.file_flag(own['csv'].name,'identity','dedupe_by_entry_key') is True
    refs=parse_string((HERE/'staged-references.bib').read_text(),'bibtex').entries
    installed=parse_string((DATA/'cldf/sources.bib').read_text(),'bibtex').entries
    previous=parse_string((HERE/'baseline-bibliography.bib').read_text(),'bibtex').entries
    assert all(installed[k]==v for k,v in previous.items()) and all(installed[k]==v for k,v in refs.items())
    formatted={k:PybtexEngine().format_from_string(v.to_string('bibtex'),'plain',output_backend='markdown') for k,v in refs.items()}
    assert all(formatted.values())
    for key in refs:
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','ocr')==(key==source)
        assert meta.flag(key,'reference','etymology_provenance')==('source' if key==source else 'none')
    language_ids={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    dialects={r['ID']:r for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    proposed=list(csv.DictReader((HERE/'staged-dialects.csv').open()))
    assert all(dialects[r['ID']]==r and all(not r[k] for k in ['Glottocode','Latitude','Longitude','Quality']) for r in proposed)
    dialect_tags={r['Tag'] for r in dialects.values()}
    assert all(r[0] in language_ids and set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS|dialect_tags for r in expected)
    checks={};terminal='1234⁴5⁵67⁷,;.'
    with tempfile.TemporaryDirectory(dir=HERE,prefix='canonical-parser-') as temp:
        for mode in ['canonical','NFC','NFD']:
            if mode=='canonical':path=own['csv'];wanted=expected
            else:
                path=Path(temp)/own['csv'].name
                wanted=[[ud.normalize(mode,v) if i==2 else v for i,v in enumerate(r)] for r in expected]
                with path.open('w',newline='') as stream:csv.writer(stream).writerows(wanted)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=own['csv'].stem)
            assert not errors.getvalue() and len(parsed)==stats['converted']==827,errors.getvalue()
            dedupe=[]
            for r,w in zip(parsed,wanted):
                assert [getattr(r,f) for f in FIELDS]==w,r.entry_key
                display=ud.normalize('NFC',w[2].strip(terminal).lower()).replace('ṅ','ŋ')
                assert ud.normalize('NFC',r.form)==display,(r.entry_key,r.form,display)
                dedupe.append(make_cldf.source_entry_dedupe_key(r,meta))
            assert len(set(dedupe))==827
            checks[mode]={'rows':827,'all15fields_exact':True,'orderedkeys':True,'dedupe_keys':827,'unintended_splits':0}
    audit=[json.loads(s) for s in own['audit'].read_text().splitlines()];assert len(audit)==1570
    assert all(r['accepted_for_installation'] is True for r in audit)
    keys={r[10] for r in expected};assert {k for a in audit for k in a['entry_keys']}==keys|set(approval['reused_existing_keys'])
    assert all(k in keys for r in expected for k in [r[11],*r[13].split('|')] if k)
    adjacent={r[10]:[k for k in [r[11],*r[13].split('|')] if k] for r in expected}
    active=set();done=set()
    def visit(key):
        assert key not in active,('source-local graph cycle',key)
        if key in done:return
        active.add(key)
        for parent in adjacent[key]:visit(parent)
        active.remove(key);done.add(key)
    for key in adjacent:visit(key)
    assert sha(DATA/'data/form-identities.csv')==identity_before
    # The actual public guarded entrypoint must be an exact no-op after install.
    import importlib.util
    spec=importlib.util.spec_from_file_location('actual_kolami_guarded_entrypoint',HERE/'install_source.py')
    installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    assert installer.main(install=True)['canonical_mutations'] is False
    os.chdir(original)
    report={'status':'Canonical source-stage checks passed; compiled/browser gates deferred','checks':checks,
      'own_assets':{role:sha(p) for role,p in own.items()},'accepted_audit_records':1570,'closed_variants':sum(bool(r[11]) for r in expected),
      'ordered_component_links':sum(len(r[13].split('|')) for r in expected if r[13]),'prior_bibliography_entries_preserved':len(previous),
      'formatted_references':formatted,'two_registered_source_dialects_verified':True,'durable_identity_sha256':identity_before,
      'identity_unchanged':True,'guarded_repeat_noop':True,'monkeypatch':False,'full_build_invoked':False,
      'source_local_graph_acyclic':True,
      'reuse':'Three existing keys/rows remain source-owned; new scope mappings are audit-only. Compiled alias/citation/persistent-ID proof deferred.'}
    (PACKAGE/'CANONICAL_SOURCE_STAGE_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':checks,'prior_bib_preserved':len(previous),'variants':report['closed_variants'],'components':report['ordered_component_links'],'identity_unchanged':True,'noop':True}))
if __name__=='__main__':main()
