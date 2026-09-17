"""Reproduce nine selected dictionary heads for approved central-survey links."""
import argparse, csv, io, json, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).with_name('20260911-central-surveys-donors-audit.json')
OUTPUT=ROOT/'data/other/params/20260911-central-surveys-donors.csv'
def render():
    rows=json.loads(AUDIT.read_text())
    assert len(rows)==9 and len({r['Entry_Key'] for r in rows})==9
    buf=io.StringIO(newline='')
    for r in rows:
        assert r['Language_ID'] in {'Ar','Pers','H'}
        assert all(r[k] for k in ['Original','Native','Source','SourceExcerpt','URLs','Transcription'])
        assert r['Form']==unicodedata.normalize('NFC',r['Form']) and '�' not in r['Form']
        csv.writer(buf).writerow([r[k] for k in ['ID','Language_ID','Form','Gloss','Source']])
    return buf.getvalue().encode()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');a=p.parse_args()
    out=render()
    if a.install:OUTPUT.write_bytes(out)
    else:assert OUTPUT.read_bytes()==out
    print('Nine selected dictionary heads reproduced.')
