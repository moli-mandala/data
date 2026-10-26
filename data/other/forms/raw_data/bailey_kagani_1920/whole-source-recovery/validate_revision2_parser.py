"""Validate the draft with unmodified parser code in a small local source tree."""
import csv,hashlib,io,json,os,shutil,sys,unicodedata
from pathlib import Path
import yaml

P=Path(__file__).resolve().parent
D=P.parent.parents[4]
STEM='20260925-bailey-kagani'
SANDBOX=D.parent/'tmp/kagani-whole/parser-sandbox'


def main():
    target=SANDBOX/'data/other/forms'
    target.mkdir(parents=True,exist_ok=True)
    (SANDBOX/'conversion').mkdir(exist_ok=True)
    hashes={}
    for name in ['make_cldf.py','source_meta.py']:
        shutil.copyfile(D/name,SANDBOX/name)
        hashes[name]=hashlib.sha256((D/name).read_bytes()).hexdigest()
    meta=yaml.safe_load((D/f'data/other/forms/{STEM}.yaml').read_text())
    meta['defaults']['transcription'].update(input='form',preserve_hyphens=True)
    (target/f'{STEM}.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))
    shutil.copyfile(P/'revision2-working/profile-proposal.txt',SANDBOX/'conversion/bailey-kagani-1920.txt')
    rows=list(csv.reader((P/'revision2-working/proposal.csv').open()))
    os.chdir(SANDBOX)
    sys.path[:0]=[str(SANDBOX),str(D)]
    import make_cldf
    checks={}
    for normalization in ['NFC','NFD']:
        inputs=[[unicodedata.normalize(normalization,v)if i==2 else v for i,v in enumerate(r)]for r in rows]
        path=target/f'{STEM}.csv'
        with path.open('w',newline='')as f:csv.writer(f).writerows(inputs)
        errors=io.StringIO()
        parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
        assert not errors.getvalue(),errors.getvalue()
        assert len(parsed)==len(rows)==stats['converted']
        assert [r.entry_key for r in parsed]==[r[10]for r in rows]
        for actual,expected in zip(parsed,inputs):
            assert (actual.old_form,actual.notes,actual.source,actual.tags)==(expected[2],expected[6],expected[7],expected[14]),actual.entry_key
            display=unicodedata.normalize('NFC',expected[2]).lower().replace('w','v').replace('ṅ','ŋ')
            assert unicodedata.normalize('NFC',actual.form)==display,(actual.entry_key,actual.form,display)
        checks[normalization]=dict(rows=len(parsed),errors=0,ordered_keys_preserved=True,originals_notes_sources_tags_preserved=True)
    report=dict(status='passed draft parser route; source/metadata acceptance separate',checks=checks,code_sha256=hashes,proposal_sha256=hashlib.sha256((P/'revision2-working/proposal.csv').read_bytes()).hexdigest(),profile_sha256=hashlib.sha256((P/'revision2-working/profile-proposal.txt').read_bytes()).hexdigest(),method='Unmodified parser and metadata loader copied into a small local source tree; intended input:form/preserve_hyphens:true route. No canonical or compiled writes.')
    (P/'revision2-working/parser-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
