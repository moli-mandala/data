"""Emit the source-checked Shina/Brokskat donor-head supplement.

Selected comparative dictionary heads, not a full source import. Exact source
spellings/prose and deduplication decisions live in the audit. Curated parameter
heads preserve the reviewed transcription; no new attested paradigms or loan
routes are inferred. Run without --install to check deterministic regeneration.
"""
import argparse,csv,io,json,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).with_name('20260910-sh-bro-donors-audit.json')
OUTPUT=ROOT/'data/other/params/20260910-sh-bro-donors.csv'
def render():
    records=json.loads(AUDIT.read_text())
    assert len(records)==123
    selected=[r for r in records if r['Status']=='install']
    assert len(selected)==94 and len({r['ID'] for r in records})==123
    for r in records:
        assert r['Language_ID'] in {'H','Pers','Psht','Bur','D'}
        assert r['Evidence'] and r['Source'] and r['Gloss'] and r['Original']
        assert unicodedata.normalize('NFC',r['Form'])==r['Form']
        assert '\ufffd' not in r['Form']
        if r['Status']!='install':assert r['ResolvedParameter']
    b=io.StringIO(newline='');csv.writer(b).writerows([[r[k] for k in ['ID','Language_ID','Form','Gloss','Source']] for r in selected])
    return b.getvalue()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');args=p.parse_args();out=render().encode()
    if args.install:OUTPUT.write_bytes(out)
    else:assert OUTPUT.read_bytes()==out
    print('123 donor heads audited: 94 emitted, 29 existing supplement heads reused')
