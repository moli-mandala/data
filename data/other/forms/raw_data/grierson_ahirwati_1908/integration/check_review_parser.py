"""Isolated actual-parser check of a pinned, unaccepted Ahirwati review revision."""
import argparse,csv,hashlib,io,json,os,shutil,sys,tempfile,unicodedata as ud
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
ROOT=DATA.parent
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
SOURCE='grierson1908ahirwati'
PROFILE='grierson-ahirwati-1908-literal-preliminary'
BOUNDARIES='1234⁴5⁵67⁷,;.'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(revision,report):
    revision=revision.resolve();report=report.resolve()
    assert revision.parent==HERE and report.parent==HERE
    status=json.loads((revision/'status.json').read_text())
    assert status['accepted_for_installation'] is False
    for name,pin in status['outputs'].items():assert sha(revision/name)==pin,name
    rr=list(csv.reader((revision/'combined-review-candidate.csv').open()))
    assert len(rr)==len({r[10] for r in rr})==653
    settings=yaml.safe_load((revision/'combined-review-candidate.yaml').read_text())
    checks={};deltas=[];oldcwd=Path.cwd()
    # Fixtures live in Jambu tmp; frozen source revision is never altered.
    with tempfile.TemporaryDirectory(prefix='ahirwati-parser-',dir=ROOT/'tmp/jambu-next-source') as tmp:
        scratch=Path(tmp);target=scratch/'data/other/forms';target.mkdir(parents=True)
        (scratch/'conversion').mkdir()
        for name in ['make_cldf.py','source_meta.py']:shutil.copyfile(DATA/name,scratch/name)
        shutil.copyfile(revision/(PROFILE+'.txt'),scratch/'conversion'/(PROFILE+'.txt'))
        (target/'combined-review-candidate.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
        try:
            os.chdir(scratch);sys.path[:0]=[str(scratch),str(DATA)]
            import make_cldf,source_meta
            meta=source_meta.load()
            assert meta.flag(SOURCE,'forms','split_alternates',True) is False
            assert meta.flag(SOURCE,'gloss','grammar_tags',True) is False
            assert meta.file_flag('combined-review-candidate.csv','identity','dedupe_by_entry_key',False) is True
            for language in {r[0] for r in rr}:
                assert meta.transcription(SOURCE,'combined-review-candidate.csv',language)==(PROFILE,True)
            for normalization in ['NFC','NFD']:
                expected=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rr]
                path=target/'combined-review-candidate.csv'
                with path.open('w',newline='') as stream:csv.writer(stream).writerows(expected)
                errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name='combined-review-candidate')
                assert not errors.getvalue(),errors.getvalue()
                assert len(parsed)==stats['converted']==653,stats
                for got,wanted in zip(parsed,expected):
                    assert [getattr(got,f) for f in FIELDS]==wanted,got.entry_key
                    display=ud.normalize('NFC',wanted[2].strip(BOUNDARIES).lower()).replace('ṅ','ŋ')
                    assert ud.normalize('NFC',got.form)==display,(got.entry_key,got.form,display)
                    if normalization=='NFC' and got.form!=wanted[2]:
                        deltas.append({'entry_key':got.entry_key,'Original':wanted[2],'display':got.form})
                assert len({make_cldf.source_entry_dedupe_key(r,meta) for r in parsed})==653
                checks[normalization]={'rows':653,'all15fields_exact':True,'ordered_keys':True,'initial_key_dedupe':653,'unintended_splits':0}
        finally:os.chdir(oldcwd)
    result={'status':'pass','review_revision':str(revision.relative_to(HERE)),
      'candidate_sha256':sha(revision/'combined-review-candidate.csv'),'settings_sha256':sha(revision/'combined-review-candidate.yaml'),
      'profile_sha256':sha(revision/(PROFILE+'.txt')),'code_pins':{n:sha(DATA/n) for n in ['make_cldf.py','source_meta.py','form_grammar.py']},
      'checks':checks,'display_deltas':deltas,'display_delta_count':len(deltas),
      'accepted_for_installation':False,'canonical_changes':False,
      'limits':['This validates a review scaffold, including pending grammar/No37 metadata.','Profile display decisions remain preliminary; exact source Original is retained.','Proposed dialect registration and final original-output audit remain pending.','No compiled ID/graph/full-build/browser gate invoked.']}
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'pass','checks':checks,'display_deltas':len(deltas)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,default=HERE/'review-assembly-r1');p.add_argument('--report',type=Path,default=HERE/'review-assembly-r1-parser-check.json');a=p.parse_args();main(a.revision,a.report)
