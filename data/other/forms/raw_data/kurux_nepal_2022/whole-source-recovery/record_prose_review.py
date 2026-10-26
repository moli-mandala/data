"""Record bounded original-page review, preserving the earlier inventory."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=HERE/'prose-inventory-first-reading.jsonl'
rows=[json.loads(s) for s in source.read_text().splitlines()]
changes=[]
for row in rows:
 if row['source_unit_key']=='kurux-nepal:p81:prose:quoted-occupation':
  changes.append({'key':row['source_unit_key'],'old_forms':row['forms'],'forms':['mod͡ʒduri'],'basis':'Original 9x crop and CharisSIL embedded span both show plain d and upper affricate tie.'})
  row['forms']=['mod͡ʒduri']
  row['note']='Exact bracketed GJ03 occupation13 quotation [mod͡ʒduri]. Embedded CharisSIL text and enlarged original agree on plain d with affricate tie. Language attribution is not explicit; documentary evidence only.'
 row['status']='source inventory reconciled; independent whole-output review pending'
(HERE/'prose-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
scratch=ROOT/'tmp/kurux-nepal-whole'
shutil.copyfile(scratch/'active.png',HERE/'appendix-occupation-original.png')
shutil.copyfile(scratch/'occupation-detail.json',HERE/'appendix-occupation-font-trace.json')
report={'status':'owner prose reconciliation complete','source_pdf_sha256':'aedf47151ebea80852cb7ed66d085fc18310aa3f4d35e189c698b624f3cbef5a','inventory_sha256':sha(source),'reviewed_sha256':sha(HERE/'prose-reviewed.jsonl'),'groups':len(rows),'corrections':changes,'original_visual_pages':[4,5,15,16,81], 'independent_appendix_review':'root-appendix81-83-scope-review.json','documentary_context_review':'Script-name and comparison-segment controls checked in extracted native text; not claimed as additional independent image transcription. All other appendix labels follow the root bounded original review.','scope_limit':'This is lexical/documentary inventory reconciliation, not a transcription of every administrative table cell. Whole-page census separately accounts for all137 pages.','keys':[r['source_unit_key'] for r in rows],'evidence_hashes':{n:sha(HERE/n) for n in ['appendix-occupation-original.png','appendix-occupation-font-trace.json','root-appendix81-83-scope-review.json']}}
(HERE/'prose-owner-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'groups':len(rows),'corrections':len(changes)}))
