"""Actual unchanged parser on the precise1832-row correction in isolated Jambu."""
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unicodedata as ud
from pathlib import Path
from apply_correction import HERE,DATA,ROOT,validate,file_hash

def main():
    _,plan=validate(draft=True)
    rows=list(csv.reader((HERE/'stage/corrected.csv').open(newline='')))
    baseline=list(csv.reader((DATA/'data/other/forms/20260927-grierson-kui.csv').open(newline='')))
    assert len(rows)==len(baseline)==1832
    changed=[(i,a,b) for i,(a,b) in enumerate(zip(baseline,rows)) if a!=b]
    assert len(changed)==1 and changed[0][1][2]=='and' and changed[0][2][2]=='amd'
    assert changed[0][2][10]=='grierson1906kui:p566:outside-comparand:06'
    fields=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology',
        'entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
    checks={};oldcwd=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='kui-amd-parser-',dir=ROOT/'tmp') as name:
        scratch=Path(name);forms=scratch/'data/other/forms';forms.mkdir(parents=True)
        (scratch/'conversion').mkdir()
        for filename in ['make_cldf.py','source_meta.py']:shutil.copyfile(DATA/filename,scratch/filename)
        stem='20260927-grierson-kui'
        shutil.copyfile(DATA/f'data/other/forms/{stem}.yaml',forms/f'{stem}.yaml')
        shutil.copyfile(DATA/'conversion/grierson-kui-1906.txt',scratch/'conversion/grierson-kui-1906.txt')
        os.chdir(scratch);sys.path[:0]=[str(scratch),str(DATA)]
        import make_cldf
        for normalization in ['NFC','NFD']:
            inputs=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
            path=forms/f'{stem}.csv'
            with path.open('w',newline='') as stream:csv.writer(stream).writerows(inputs)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=stem)
            assert not errors.getvalue(),errors.getvalue()
            assert len(parsed)==stats['converted']==1832
            assert [r.entry_key for r in parsed]==[r[10] for r in inputs]
            for actual,raw in zip(parsed,inputs):
                assert [getattr(actual,f) for f in fields]==raw,actual.entry_key
                expected=ud.normalize('NFC',raw[2]).lower().replace('ṅ','ŋ').strip('1234⁴5⁵67⁷,;.')
                assert ud.normalize('NFC',actual.form)==expected,actual.entry_key
            checks[normalization]=dict(rows=1832,all15fields_exact=True,ordered_keys=True,errors=0,
                corrected_key_original_and_display='amd',unintended_splits=0)
        os.chdir(oldcwd)
    report=dict(status='Precise staged correction actualparserPASS; no canonical writes',checks=checks,
        draft_sha256=file_hash(HERE/'root-acceptance-draft.json'),csv_sha256=file_hash(HERE/'stage/corrected.csv'),
        rows=1832,unchanged_other_raw_records=1831,key_order_unchanged=True,
        parser_sha256={name:file_hash(DATA/name) for name in ['make_cldf.py','source_meta.py']},
        deferred=['Root plan activation','Guarded source-stage correction transaction','All full-build/compiled/browser gates'])
    (HERE/'staged-parser-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
