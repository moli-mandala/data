"""Actual generic parser with staged YAML only; no canonical acceptance claim."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import make_cldf
import source_meta
from source_meta import SourceMeta

FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
STEM='20260927-nair-mullu-kurumba'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths=[PACKAGE/'proposed.csv',PACKAGE/'proposal-audit.jsonl',HERE/(STEM+'.yaml'),HERE/'staged-references.bib',HERE/'accepted-audit-proposal.jsonl']
    pins={str(p.relative_to(PACKAGE)):sha(p) for p in paths}
    rows=list(csv.reader((PACKAGE/'proposed.csv').open(newline='')))
    assert len(rows)==len({r[10] for r in rows})==135
    assert sum(bool(r[11]) for r in rows)==11
    assert all(not r[11] or r[11] in {a[10] for a in rows} for r in rows)
    settings=HERE/(STEM+'.yaml')
    meta=SourceMeta([settings])
    assert set(meta.sources)=={'nair2013mullukurumba','robertsathyajoseph1982mullukurumba'}
    rule=meta.transcription_rule('nair2013mullukurumba',STEM+'.csv','Mullu Kurumba')
    assert rule['convert'] is False and rule['preserve_hyphens'] is True
    for key in meta.sources:
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','ocr') is False
        assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013mullukurumba' else 'none')
    original=source_meta.yaml_files
    checks={}
    try:
        source_meta.yaml_files=lambda:[settings]
        source_meta.load.cache_clear()
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            for normalization in ('NFC','NFD'):
                expected=[[unicodedata.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
                path=Path(temporary)/(STEM+'.csv')
                with path.open('w',newline='') as stream: csv.writer(stream).writerows(expected)
                errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
                assert not errors.getvalue(),errors.getvalue()
                assert len(parsed)==135 and stats=={'converted':0,'for_conversion':0}
                for actual,wanted in zip(parsed,expected):
                    assert [getattr(actual,f) for f in FIELDS]==wanted,actual.entry_key
                    assert unicodedata.normalize('NFC',actual.form)==unicodedata.normalize('NFC',wanted[2])
                assert len({(r.lang,r.param,r.form,make_cldf.source_entry_dedupe_key(r,meta)) for r in parsed})==135
                checks[normalization]=dict(rows=135,all15fields_exact=True,initial_dedupe_keys=135,zero_conversion=True)
    finally:
        source_meta.yaml_files=original
        source_meta.load.cache_clear()
    assert pins=={str(p.relative_to(PACKAGE)):sha(p) for p in paths}
    report=dict(status='Staged YAML and actual parser checks pass; no canonical installation',inputs=pins,checks=checks,
        source_reference_flags=2,closed_variants=11,canonical_changes=False,
        isolation='Only YAML discovery patched; actual parse_file unchanged. This is not a canonical SourceMeta or compiled acceptance claim.')
    (HERE/'actual-parser-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return report


if __name__=='__main__':main()
