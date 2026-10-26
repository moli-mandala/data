"""Owner reconciliation after viewing every original grammar example/table page."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda name:[json.loads(x) for x in (HERE/name).read_text().splitlines()]
rows=read('grammar-examples-first-reading.jsonl')
changes=[]
fullstops={'2','3','5','6','9','10','12','18','20','23','24','28','29','30','31','32','40','41','42','43'}
questions={'26','27','33','34'}
exclamations={'36a','36b','36c','37','39b','39c'}
for r in rows:
 n=r['example']; r['review_status']='owner_original_page_reviewed'
 if n in ('1a','1b','1c'):
  old=r['aligned_glosses']
  r['aligned_glosses']=[{'1a':'monkey','1b':'two monkeys','1c':'(three or more) monkeys'}[n]]
  r['note']='Number analysis comes from explicit adjacent prose (singular unmarked, dual =kin, plural =kun). Aligned gloss retains actual printed English, not an invented morpheme-by-morpheme string.'
  changes.append(dict(key=r['source_unit_key'],field='aligned_glosses',old=old,new=r['aligned_glosses'],reason='Retain printed gloss; move prose-supported analysis to metadata.'))
 if n in ('12','20'):
  old=r['aligned_glosses'][3];r['aligned_glosses'][3]='neg=3sg.anim.subjⱼ'
  changes.append(dict(key=r['source_unit_key'],field='aligned_glosses[3]',old=old,new=r['aligned_glosses'][3],reason='Original has subject coreference subscript j omitted by plain-text extraction.'))
 if n in fullstops and not r['raw_surface'].endswith('.'):
  r['raw_surface']+='.'
 elif n in questions:r['raw_surface']+='?'
 elif n in exclamations:r['raw_surface']+='!'
 r['raw_surface_policy']='Sentence punctuation preserved as raw evidence. Lexical token forms omit editorial sentence punctuation only; brackets in [h]oɽ remain source correction evidence.'
(HERE/'grammar-examples-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
tables=read('tables-first-reading.jsonl')
for r in tables:r['review_status']='owner_original_page_reviewed'
for page,table in [(268,3),(271,5)]:
 tables.append(dict(source_unit_key=f'peterson2024turi:p{page}:table{table}:1sg-incl-blank',printed_page=page,pdf_page=page-260,table=table,form='',gloss='first-person inclusive singular table cell',status='structural_blank',note='Inclusive row has no singular cell; no unattested lexical form inferred.',review_status='owner_original_page_reviewed'))
(HERE/'tables-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tables))
report=dict(status='owner_original_page_review_passed',pages=list(range(267,280)),numbered_subexamples=len(rows),aligned_words_or_phrases=sum(len(r['forms']) for r in rows),table_units=len(tables),original_review='Each original page rendered separately at 1.8x and viewed; literal symbols checked against native-span evidence. Tables3/4/5/6 headers, all alternatives, zero markers and blanks accounted.',input_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['grammar-examples-first-reading.jsonl','tables-first-reading.jsonl']},corrections=changes,source_inconsistencies=['15b literal see / 3pl versus he went.','19b negative repeats positive English translation.','12/20 coreference indices remain documentary.','29 editorial [h] and historical reprint uncertainty retained.'],independent_review='pending',whole_source_completion=False)
(HERE/'owner-grammar-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(f'{len(rows)} subexamples and {len(tables)} table units owner-reviewed; inline inventory remains')
