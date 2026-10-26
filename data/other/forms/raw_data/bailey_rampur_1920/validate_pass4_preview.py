"""Validate pass4 previews while preserving all earlier frozen files."""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path
import yaml

P=Path(__file__).resolve().parent
D=P.parents[4]
SANDBOX=D.parent/'tmp/rohru-whole-review/pass4-parser-sandbox'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=SANDBOX/'data/other/forms';target.mkdir(parents=True,exist_ok=True)
    (SANDBOX/'conversion').mkdir(exist_ok=True)
    code={}
    for name in ['make_cldf.py','source_meta.py']:
        shutil.copyfile(D/name,SANDBOX/name);code[name]=sha(D/name)
    sets={}
    for name in ['rampur','baghi']:
        package=P.parent/f'bailey_{name}_1920';stem=f'20260925-bailey-{name}'
        with (package/'pass4-preview.csv').open(newline='') as stream: rows=list(csv.reader(stream))
        with (package/'whole-proposed-profile.txt').open(newline='') as stream:
            mapping={r['Grapheme']:r['IPA'] for r in csv.DictReader(stream,delimiter='\t')}
        added={}
        for char in sorted({c for row in rows for c in unicodedata.normalize('NFC',row[2])}-set(mapping)):
            assert char not in ['\ufffd','\n','\t']
            added[char]=char.lower().replace('w','v').replace('ṅ','ŋ')
        mapping.update(added)
        for key,value in list(mapping.items()):
            decomposed=unicodedata.normalize('NFD',key)
            assert decomposed not in mapping or mapping[decomposed]==value,(key,value)
            mapping[decomposed]=value
        profile=package/'pass4-profile.txt'
        with profile.open('w',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t');writer.writerow(['Grapheme','IPA']);writer.writerows(mapping.items())
        meta=yaml.safe_load((D/f'data/other/forms/{stem}.yaml').read_text())
        meta['defaults']['transcription'].update(input='form',preserve_hyphens=True)
        (target/f'{stem}.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))
        shutil.copyfile(profile,SANDBOX/f'conversion/bailey-{name}-1920.txt')
        sets[name]=(package,stem,rows,added)
    os.chdir(SANDBOX);sys.path[:0]=[str(SANDBOX),str(D)]
    import make_cldf
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    for name,(package,stem,rows,added) in sets.items():
        checks={}
        for row in rows:
            assert not(set(row[14].split())-(GRAMMATICAL_TAGS|GENDER_TAGS)),(row[10],row[14])
        for norm in ['NFC','NFD']:
            inputs=[[unicodedata.normalize(norm,v) if i==2 else v for i,v in enumerate(row)] for row in rows]
            path=target/f'{stem}.csv'
            with path.open('w',newline='') as stream:csv.writer(stream).writerows(inputs)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=stem)
            assert not errors.getvalue(),errors.getvalue()
            assert len(parsed)==len(rows)==stats['converted']
            assert [r.entry_key for r in parsed]==[r[10] for r in inputs]
            for actual,expected in zip(parsed,inputs):
                assert (actual.old_form,actual.notes,actual.source,actual.tags)==(expected[2],expected[6],expected[7],expected[14]),actual.entry_key
                display=unicodedata.normalize('NFC',expected[2]).lower().replace('w','v').replace('ṅ','ŋ')
                assert unicodedata.normalize('NFC',actual.form)==display,(actual.entry_key,actual.form,display)
            checks[norm]=dict(rows=len(parsed),errors=0,ordered_keys_originals_notes_sources_tags_preserved=True)
        report=dict(status='pass4 preview parser checks passed; acceptance pending',checks=checks,
                    added_profile_symbols=added,code_sha256=code,preview_sha256=sha(package/'pass4-preview.csv'),
                    profile_sha256=sha(package/'pass4-profile.txt'),canonical_changes=False)
        (package/'pass4-parser-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(name,json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
