"""Exercise unmodified parse_file with proposed metadata in a small isolated tree."""
from pathlib import Path
import csv, hashlib, io, json, os, shutil, sys, unicodedata
import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[4]
SANDBOX = DATA.parent / 'tmp/rohru-whole-review/parser-sandbox'
STEM = '20260925-bailey-rohru'

def main():
    target = SANDBOX / 'data/other/forms'
    target.mkdir(parents=True, exist_ok=True)
    (SANDBOX/'conversion').mkdir(exist_ok=True)
    hashes = {}
    for name in ('make_cldf.py','source_meta.py'):
        shutil.copy2(DATA/name,SANDBOX/name)
        hashes[name] = hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    meta=yaml.safe_load((DATA/f'data/other/forms/{STEM}.yaml').read_text())
    meta['defaults']['transcription'].update(input='form',preserve_hyphens=True)
    (target/f'{STEM}.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))
    shutil.copy2(HERE/'whole-proposed-profile.txt',SANDBOX/'conversion/bailey-rohru-1920.txt')
    raw=list(csv.reader((HERE/'whole-proposed.csv').open()))
    os.chdir(SANDBOX)
    sys.path[:0]=[str(SANDBOX),str(DATA)]
    import make_cldf
    results={}
    for normalization in ('NFC','NFD'):
        path=target/f'{STEM}.csv'
        rows=[[unicodedata.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in raw]
        with path.open('w',newline='') as f: csv.writer(f).writerows(rows)
        errors=io.StringIO()
        parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
        assert errors.getvalue()=='',errors.getvalue()
        assert len(parsed)==stats['converted']==723
        assert [r.entry_key for r in parsed]==[r[10] for r in raw]
        for r,source in zip(parsed,rows):
            assert (r.old_form,r.notes,r.source,r.tags)==(source[2],source[6],source[7],source[14]),r.entry_key
            expected=unicodedata.normalize('NFC',source[2].lower().replace('w','v'))
            assert unicodedata.normalize('NFC',r.form)==expected,(r.entry_key,r.form,expected)
        results[normalization]=dict(rows=len(parsed),converted=stats['converted'],errors=0,keys='exact ordered match',original='exact input retained',display='literal lowercase and w→v; NFC-equivalent')
    report=dict(status='pass',method='Unmodified make_cldf.py and source_meta.py copied byte-exact; isolated YAML adds intended input:form/preserve_hyphens:true and proposed profile. No monkeypatch, canonical writes or CLDF build.',code_sha256=hashes,proposal_sha256=hashlib.sha256((HERE/'whole-proposed.csv').read_bytes()).hexdigest(),checks=results)
    (HERE/'whole-parser-validation-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
