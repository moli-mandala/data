"""Whole-source LSI Simla Siraji draft importer; explicit source-stage installation only."""
from pathlib import Path
import csv,json,unicodedata,argparse
PACKAGE=Path(__file__).resolve().parent
DATA=PACKAGE.parents[4]
SOURCE='grierson1916simlasiraji'
OUTPUT=DATA/'data/other/forms/20260925-grierson-simla-siraji.csv'
INPUTS=[PACKAGE/'full-transcription.tsv',PACKAGE/'grammar-transcription.tsv']
def generate():
 rows=[]
 for path in INPUTS:
  with path.open() as f: rows.extend(csv.DictReader(f,delimiter='\t'))
 table=[r for r in rows if r['unit'].startswith('item:')]
 if [int(r['unit'].split(':')[1]) for r in table]!=list(range(1,242)):raise ValueError('Incomplete 241-cell list')
 if len(rows)!=341:raise ValueError('Expected whole341 lexical source units')
 out=[];audit=[]
 for r in rows:
  if r['decision'] not in {'accepted','source_blank'}:raise ValueError('Invalid decision')
  p=int(r['page']);base=f"{SOURCE}:p{p}:{r['unit']}";loc=f"p. {p}, {r['stratum']}, {r['unit']}"
  forms=r['forms'].split(';') if r['decision']=='accepted' else []
  if forms and not all(forms):raise ValueError('Empty accepted form')
  keys=[]
  for n,form in enumerate(forms,1):
   key=base if n==1 else f'{base}:answer{n}';keys.append(key)
   tag=r['tags'];note=r['notes']
   if ' ' in form:tag=(tag+' multiword-expression').strip()
   out.append(['ShimlaSiraji','',unicodedata.normalize('NFC',form),r['gloss'],'','',note,f'{SOURCE}[{loc}]','','',key,'','','',tag])
  audit.append(dict(source_cell_key=base,printed_page=p,scan_page=p+16,section=r['stratum'],unit=r['unit'],english_prompt=r['gloss'],visual_reading=r['forms'],source_pattern=r['printed_form'],source_notes=r['notes'],status='ingested' if forms else r['decision'],entry_keys=keys,language_id='ShimlaSiraji',citation_locator=loc))
 if len(set(r[10] for r in out))!=len(out):raise ValueError('Duplicate keys')
 return out,audit

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--install',action='store_true');a=ap.parse_args()
 rows,audit=generate()
 if a.install:
  with OUTPUT.open('w',newline='') as f:csv.writer(f).writerows(rows)
  (PACKAGE/'audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in audit))
 with (PACKAGE/'full-staged.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (PACKAGE/'full-staged-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in audit))
 print(len(audit),'source units,',len(rows),'forms (source stage; full build deferred)')
if __name__=='__main__':main()
