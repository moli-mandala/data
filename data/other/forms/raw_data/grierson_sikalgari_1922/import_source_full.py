"""Reproduce the complete LSI XI Sikalgari source stage; no database construction."""
from pathlib import Path
import argparse,csv,json,unicodedata
PACKAGE=Path(__file__).resolve().parent
DATA=PACKAGE.parents[4]
SOURCE='grierson1922lsi11'
DIALECT='dialect:Sik:sik_belgaum:Belgaum'
OUTPUT=DATA/'data/other/forms/20260925-grierson-sikalgari.csv'
INPUTS=[PACKAGE/f'{s}-transcription.tsv' for s in ('table','grammar','specimen')]
LEGACY_PROMPTS=set(range(1,14))|set(range(32,101))
def generate():
 units=[]
 for path in INPUTS:
  with path.open() as f:units.extend(csv.DictReader(f,delimiter='\t'))
 assert len(units)==790
 assert [int(r['unit'][5:]) for r in units if r['unit'].startswith('item:')]==list(range(1,242))
 out=[];audit=[];identity={}
 for r in units:
  p=int(r['page']);u=r['unit'];form=unicodedata.normalize('NFC',r['forms']);gloss=r['gloss']
  if not form or r['decision']!='accepted':raise ValueError('Unexpected blank or unaccounted unit')
  table=u.startswith('item:');prompt=int(u[5:]) if table else None
  key=f'{SOURCE}:sikalgari_belgaum:{prompt}' if table else f'{SOURCE}:sikalgari_belgaum:p{p}:{u}'
  loc=f'p. {p}, item {prompt}' if table else f"p. {p}, {r['stratum']}, {u}"
  citation=f'{SOURCE}[{loc}]';tags=list(dict.fromkeys([DIALECT,*r['tags'].split(),*(['multiword-expression'] if ' ' in form else [])]))
  # Every table prompt stays a distinct record. Repeated interlinear tokens may
  # share only an identical literal form, gloss, tags and residual source note.
  sig=(form,gloss,tuple(tags),r['notes'])
  reuse=not table and sig in identity
  if reuse:
   target=out[identity[sig]]
   if citation not in target[7].split(';'):target[7]+=';'+citation
   targetkey=target[10]
  else:
   targetkey=key;identity.setdefault(sig,len(out))
   out.append(['Sik','',form,gloss,'','',r['notes'],citation,'','',key,'','','',' '.join(tags)])
  audit.append(dict(source_cell_key=key,printed_page=p,pdf_page=p+12,section=r['stratum'],unit=u,english_prompt=gloss,visual_reading=form,source_notes=r['notes'],source_tags=r['tags'],status='exact_reuse' if reuse else 'ingested',reason='Exact literal form, gloss, grammar and source note; citation retained.' if reuse else '',entry_keys=[targetkey],language_id='Sik',dialect_tag=DIALECT,citation_locator=loc,typed_uncertainty=r['notes'] if 'uncertainty' in r['notes'].lower() else ''))
 assert len({r[10] for r in out})==len(out)
 old=list(csv.reader((PACKAGE/'legacy-installed-before-full.csv').open()))
 assert len(old)==82 and {r[10] for r in old}<={r[10] for r in out}
 return out,audit

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--install',action='store_true');args=ap.parse_args()
 rows,audit=generate()
 if args.install:
  with OUTPUT.open('w',newline='') as f:csv.writer(f).writerows(rows)
  (PACKAGE/'audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in audit))
 with (PACKAGE/'full-staged.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (PACKAGE/'full-staged-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in audit))
 print(len(audit),'source units;',len(rows),'forms;',sum(a['status']=='exact_reuse' for a in audit),'exact reuses')
if __name__=='__main__':main()
