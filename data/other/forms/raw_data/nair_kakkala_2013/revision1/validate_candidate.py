"""Small real-parser/config/formatter fixture, without canonical writes."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata as ud
from pathlib import Path
from unittest.mock import patch
import pybtex
from pybtex.database import parse_file

HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import make_cldf
import source_meta
from tags import GENDER_TAGS,GRAMMATICAL_TAGS
STEM='20260927-nair-kakkala'
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 rows=list(csv.reader((HERE/'proposed.csv').open(newline='')))
 audit=[json.loads(x) for x in (HERE/'proposal-audit.jsonl').read_text().splitlines()]
 assert len(rows)==202 and len(audit)==41 and all(len(r)==15 for r in rows)
 assert sum(len(a['physical_witnesses']) for a in audit)==207
 assert sum(not w['entry_keys'] for a in audit for w in a['physical_witnesses'])==7
 assert {r[10] for r in rows}=={k for a in audit for k in a['entry_keys']}
 assert len({r[10] for r in rows})==202
 assert all(not any(ud.category(c)=='Co' or c=='\ufffd' for c in ''.join(r)) for r in rows)
 assert all(not r[4] and not r[5] for r in rows)
 assert all(t in GRAMMATICAL_TAGS or t in GENDER_TAGS or t.startswith('dialect:') for r in rows for t in r[14].split())
 langs={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
 assert {r[0] for r in rows}<=langs
 assert {r[2] for r in rows}>={'ñaaŋka','n̲aamaacci','el̲umaacci','pal̲ak-icci','–un̲n̲','unis´i+n̲n̲+en','kol̲iñcicci','kalamb-un̲n̲-aaŋk','tūňkiren','a','ooṭ-aṭṭaa-'}
 bykey={r[10]:r for r in rows}
 assert 'pret' not in bykey['nair2013kakkala:p28:past-neuter-marker:w1'][14].split()
 assert not any(r[2]=='-ø' for r in rows)
 results=[]
 with tempfile.TemporaryDirectory(dir=HERE/'metadata') as t:
  t=Path(t);y=t/f'{STEM}.yaml';y.write_bytes((HERE/'metadata/proposed-settings.yaml').read_bytes())
  meta=source_meta.SourceMeta(paths=[y])
  for lang in ('Kakkala','Malayalam','Tamil'):
   assert meta.transcription('nair2013kakkala',f'{STEM}.csv',lang)==(None,False)
  for key in ('nair2013kakkala','azad1980kakkala','sheela1980kakkala'):
   assert meta.flag(key,'reference','ocr') is False
   assert meta.flag(key,'reference','editor')=='OpenAI Codex'
   assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013kakkala' else 'none')
  for norm in ('NFC','NFD'):
   raw=[[ud.normalize(norm,x) for x in r] for r in rows];p=t/f'{norm}.csv'
   with p.open('w',newline='') as f:csv.writer(f).writerows(raw)
   with patch.object(make_cldf.source_meta,'load',return_value=meta):
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(p),errors,name=STEM)
   assert not errors.getvalue(),errors.getvalue()
   assert len(parsed)==len(rows) and stats=={'converted':0,'for_conversion':0},stats
   for got,want in zip(parsed,raw):
    assert [getattr(got,x) for x in FIELDS]==want,want[10]
    assert got.form==want[2]  # convert:false preserves caller Unicode normalization
   results.append(dict(normalization=norm,rows=len(parsed),exact_fields=15,errors=0))
 bib=parse_file(HERE/'metadata/proposed-references.bib')
 assert set(bib.entries)=={'nair2013kakkala','azad1980kakkala','sheela1980kakkala'}
 engine=pybtex.PybtexEngine();formatted=[]
 for k,v in bib.entries.items():
  s=engine.format_from_string(v.to_string('bibtex'),'plain',output_backend='markdown');assert s.strip();formatted.append(dict(key=k,formatted=s))
 report=dict(status='pass: isolated actual parser all15fields NFC/NFD and real SourceMeta/formatter; not independent lexical approval',
  csv_sha256=sha(HERE/'proposed.csv'),audit_sha256=sha(HERE/'proposal-audit.jsonl'),settings_sha256=sha(HERE/'metadata/proposed-settings.yaml'),
  bib_sha256=sha(HERE/'metadata/proposed-references.bib'),checks=results,references=formatted,canonical_changes=False)
 (HERE/'candidate-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(report['status'])
if __name__=='__main__':main()
