"""Validate the complete review CSV through the unmodified actual parser."""
import argparse,csv,hashlib,io,json,os,shutil,sys,tempfile,unicodedata as ud
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
DATA=ROOT/'data'
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
BOUNDARIES='1234⁴5⁵67⁷,;.'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def display(v):return ud.normalize('NFC',v.lower()).replace('ṅ','ŋ')
def main(directory=None):
    global HERE
    if directory is not None:HERE=directory.resolve()
    status=json.loads((HERE/'status.json').read_text())
    for name,pin in status['outputs'].items():assert sha(HERE/name)==pin,(name,sha(HERE/name))
    csvpath=HERE/'combined-review-candidate.csv'
    with csvpath.open() as f:rr=list(csv.reader(f))
    assert len(rr)==827 and all(len(r)==15 for r in rr)
    original=Path.cwd();checks={};deltas=[]
    settings=yaml.safe_load((HERE/'combined-review-candidate.yaml').read_text())
    settings['file']='combined-review-candidate.csv'
    with tempfile.TemporaryDirectory(prefix='actual-combined-parser-',dir=HERE) as temp:
        scratch=Path(temp);target=scratch/'data/other/forms';target.mkdir(parents=True)
        (scratch/'conversion').mkdir()
        shutil.copyfile(DATA/'make_cldf.py',scratch/'make_cldf.py')
        shutil.copyfile(DATA/'source_meta.py',scratch/'source_meta.py')
        shutil.copyfile(HERE/'grierson-kolami-basim-1906.txt',scratch/'conversion/grierson-kolami-basim-1906.txt')
        (target/'combined-review-candidate.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
        os.chdir(scratch);sys.path[:0]=[str(scratch),str(DATA)]
        import make_cldf,source_meta
        meta=source_meta.load()
        assert meta.flag('grierson1906kolamibasim','forms','split_alternates',True) is False
        assert meta.flag('grierson1906kolamibasim','gloss','grammar_tags',True) is False
        for language in {r[0] for r in rr}:
            assert meta.transcription('grierson1906kolamibasim','combined-review-candidate.csv',language)==('grierson-kolami-basim-1906',True)
        for normalization in ['NFC','NFD']:
            expected=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rr]
            path=target/'combined-review-candidate.csv'
            with path.open('w',newline='') as f:csv.writer(f).writerows(expected)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name='combined-review-candidate')
            assert not errors.getvalue(),errors.getvalue()
            assert len(parsed)==827 and stats['converted']==827, {'parsed':len(parsed),'stats':stats,'missing_keys':sorted({r[10] for r in expected}-{r.entry_key for r in parsed})}
            keys=[]
            for got,wanted in zip(parsed,expected):
                assert [getattr(got,f) for f in FIELDS]==wanted,got.entry_key
                final=display(wanted[2].strip(BOUNDARIES))
                assert ud.normalize('NFC',got.form)==ud.normalize('NFC',final),(got.entry_key,got.form,final)
                assert got.ipa==got.native==''
                keys.append(make_cldf.source_entry_dedupe_key(got,meta))
                if normalization=='NFC' and wanted[2].strip(BOUNDARIES)!=wanted[2]:
                    deltas.append({'entry_key':got.entry_key,'Original':wanted[2],'display':got.form,'terminal_punctuation_only_before_approved_profile':True})
            assert len(set(keys))==827
            checks[normalization]={'rows':827,'all15fields_exact':True,'orderedkeys':True,'initial_dedupe_keys':827,'unintended_splits':0,'phonemic_native_blank':True}
        os.chdir(original)
    lookup={r[10]:r for r in rr}
    adjacency={k:[] for k in lookup}
    for r in rr:
        adjacency[r[10]]=[k for k in [r[11],*r[13].split('|')] if k]
        assert all(k in lookup for k in adjacency[r[10]])
    active=set();done=set()
    def visit(k):
        assert k not in active,('cycle',k)
        if k in done:return
        active.add(k)
        for p in adjacency[k]:visit(p)
        active.remove(k);done.add(k)
    for k in lookup:visit(k)
    audit=[json.loads(s) for s in (HERE/'combined-review-audit.jsonl').read_text().splitlines()]
    represented={k for r in audit for k in r['entry_keys']}
    assert set(lookup)<=represented
    report={'status':'Actual unmodified parser and source-local graph closure pass; final output acceptance still pending',
        'candidate_sha256':sha(csvpath),'profile_sha256':sha(HERE/'grierson-kolami-basim-1906.txt'),
        'code_pins':{n:sha(DATA/n) for n in ['make_cldf.py','source_meta.py','form_grammar.py']},
        'checks':checks,'source_local_graph':{'variant_links':sum(bool(r[11]) for r in rr),'component_sets':sum(bool(r[13]) for r in rr),'ordered_component_links':sum(len(r[13].split('|')) for r in rr if r[13]),'closed':True,'acyclic':True},
        'all_emitted_keys_in_audit':True,'terminal_display_deltas':deltas,'terminal_display_delta_count':len(deltas),
        'current_source_literals_not_mutated':True,'monkeypatch':False,'canonical_changes':False,
        'limits':['Proposed dialects remain source-local; canonical registry installation is deferred.','Source-stage independent final actual-row audit and metadata acceptance remain required.','No compiled graph/build/browser validation invoked.']}
    (HERE/'actual-parser-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':checks,'graph':report['source_local_graph'],'terminal_display_deltas':len(deltas)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path);a=p.parse_args();main(a.directory)
