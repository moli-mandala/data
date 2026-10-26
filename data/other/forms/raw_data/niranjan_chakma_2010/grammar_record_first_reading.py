"""Reproduce grammar first reading from source-local editorial transcription.

No OCR or canonical write. Draft records preserve unresolved glyphs as explicit
candidate readings, never as accepted phonemic data or invented translations.
"""
import json,hashlib,collections,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 records=json.loads((P/'grammar-literal-draft.json').read_text())
 keys=[r['source_unit_key'] for r in records]
 assert len(keys)==len(set(keys))
 for r in records:
  assert 7<=r['pdf_page']<=20 and r['printed_page']==r['pdf_page']-2
  assert r['block_id'].startswith(f"grammar:pdf{r['pdf_page']}:")
  assert r['disposition'] in {'target-first-reading','control','source-blank','notation-uncertain'}
  assert not r.get('phonemic') and not r.get('romanization')
  if r['disposition']=='target-first-reading':assert r['form']
  assert r['review_status']!='accepted-final'
  assert r['form']==unicodedata.normalize('NFC',r['form']),r['source_unit_key']
 (P/'grammar-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records))
 census=json.loads((P/'grammar-structure-census-proposal.json').read_text())
 expected={b['id'] for p in census['pages'] for b in p['blocks']}
 seen={r['block_id'] for r in records}
 assert not seen-expected
 # Whole paradigms span page breaks; count actual printed main-column
 # predicates, not a rectangular grid filled from grammatical expectations.
 paradigms={
  'present_indefinite':([(15,3)],24),
  'present_continuous':([(15,4),(16,1)],24),
  'present_perfect':([(16,2)],18),
  'present_imperative':([(16,3),(17,1)],20),
  'past_indefinite':([(17,2)],24),
  'past_habitual':([(17,3),(18,1)],24),
  'past_substantial':([(18,2),(19,1)],30),
  'future_indefinite':([(19,2)],24),
 }
 counts={}
 for name,(blocks,n) in paradigms.items():
  ids={f'grammar:pdf{p}:block{b}' for p,b in blocks}
  actual=sum(r['block_id'] in ids and r['role']=='predicate' for r in records)
  counts[name]={'expected_printed_predicates':n,'actual':actual,'status':'pass' if actual==n else 'incomplete'}
  if ids<=seen:assert actual==n,(name,actual,n)
 summary={'stage':'Whole grammar first reading in progress; not final transcription or installation','records':len(records),'source_pdf_sha256':census['source_sha256'],'covered_blocks':sorted(seen),'remaining_blocks':sorted(expected-seen),'page_counts':{str(n):sum(r['pdf_page']==n for r in records) for n in range(7,21)},'uncertain_records':[r['source_unit_key'] for r in records if r.get('uncertainties')],'key_policy':'Physical page/block/role/item identity; repeated source attestations retain distinct keys.','canonical_changed':False}
 summary['stage']='All 57 blocks have first-reading records; transcription and metadata remain subject to full second reading and independent review.' if not expected-seen else summary['stage']
 summary['paradigm_counts']=counts
 summary['dispositions']=dict(collections.Counter(r['disposition'] for r in records))
 summary['literal_count_corrections']=['PDF17 past-indefinite third-person plural ending is blank/ink-blotted rather than a sixth legible ending.','Present-perfect two repeated complete forms in the ending column are additional peripheral records, not main-column predicates.','Past Substantial retains repeated second-person groups and the two-form plural group exactly.']
 summary['draft_sha256']=hashlib.sha256((P/'grammar-literal-draft.json').read_bytes()).hexdigest()
 summary['jsonl_sha256']=hashlib.sha256((P/'grammar-first-reading.jsonl').read_bytes()).hexdigest()
 (P/'grammar-first-reading-reconciliation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
