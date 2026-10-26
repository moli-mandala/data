"""Actual parser validation in a small isolated source tree, no canonical writes."""
import csv,hashlib,io,json,os,shutil,sys,unicodedata
from pathlib import Path
import yaml

P=Path(__file__).resolve().parent
D=P.parents[4]
SANDBOX=D.parent/'tmp/rohru-whole-review/paired-parser-sandbox'


def main():
    target=SANDBOX/'data/other/forms';target.mkdir(parents=True,exist_ok=True)
    (SANDBOX/'conversion').mkdir(exist_ok=True)
    hashes={}
    for name in ['make_cldf.py','source_meta.py']:
        shutil.copyfile(D/name,SANDBOX/name);hashes[name]=hashlib.sha256((D/name).read_bytes()).hexdigest()
    sets={}
    for name in ['rampur','baghi']:
        package=P.parent/f'bailey_{name}_1920';stem=f'20260925-bailey-{name}'
        meta=yaml.safe_load((D/f'data/other/forms/{stem}.yaml').read_text())
        meta['defaults']['transcription'].update(input='form',preserve_hyphens=True)
        (target/f'{stem}.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))
        shutil.copyfile(package/'whole-proposed-profile.txt',SANDBOX/f'conversion/bailey-{name}-1920.txt')
        sets[name]=(package,stem,list(csv.reader((package/'whole-proposed.csv').open())))
    os.chdir(SANDBOX);sys.path[:0]=[str(SANDBOX),str(D)]
    import make_cldf
    for name,(package,stem,rows) in sets.items():
        checks={}
        for norm in ['NFC','NFD']:
            inputs=[[unicodedata.normalize(norm,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
            path=target/f'{stem}.csv'
            with path.open('w',newline='') as f:csv.writer(f).writerows(inputs)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=stem)
            assert not errors.getvalue(),errors.getvalue()
            assert len(parsed)==len(rows)==stats['converted']
            assert [r.entry_key for r in parsed]==[r[10] for r in rows]
            for actual,expected in zip(parsed,inputs):
                assert (actual.old_form,actual.notes,actual.source,actual.tags)==(expected[2],expected[6],expected[7],expected[14]),actual.entry_key
                display=unicodedata.normalize('NFC',expected[2]).lower().replace('w','v').replace('ṅ','ŋ')
                assert unicodedata.normalize('NFC',actual.form)==display,(actual.entry_key,actual.form,display)
            checks[norm]=dict(rows=len(parsed),errors=0,ordered_keys_preserved=True,originals_notes_sources_tags_preserved=True,all_display_forms_match_literal_profile_policy=True)
        report=dict(status='passed draft actual-parser route; source acceptance separate',checks=checks,code_sha256=hashes,proposal_sha256=hashlib.sha256((package/'whole-proposed.csv').read_bytes()).hexdigest(),profile_sha256=hashlib.sha256((package/'whole-proposed-profile.txt').read_bytes()).hexdigest(),method='Unmodified parser and metadata loader copied into bounded local source tree with intended input:form and preserve_hyphens:true. No canonical or compiled writes.')
        (package/'whole-parser-validation-20260927.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,json.dumps(report))


if __name__=='__main__':main()
