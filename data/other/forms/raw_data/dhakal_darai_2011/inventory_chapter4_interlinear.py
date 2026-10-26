"""Inventory every printed interlinear form run without guessing word/gloss alignment."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
rows=[]
for raw in (P/'chapter4-expressions-first-reading.jsonl').read_text().splitlines():
 r=json.loads(raw)
 evidence=r['interlinear_evidence']
 for li in evidence:
  spans=[(i,s) for i,s in enumerate(li['spans']) if 'IPA93' in s['font'] and s['decoded'].strip()]
  for i,s in spans:
   # Preserve the exact physical run. Multiple words stay together until manual alignment.
   below=[x for x in evidence if x['bbox'][1]>li['bbox'][1]+3 and x['bbox'][1]<li['bbox'][1]+35 and any(t['decoded'].strip() and 'IPA93' not in t['font'] for t in x['spans'])]
   rows.append(dict(source_unit_key=r['source_unit_key']+f":interlinear:l{li['line']}:s{i+1}",parent_expression_key=r['source_unit_key'],printed_page=r['printed_page'],pdf_page=r['pdf_page'],native_line=li['line'],native_span=s,form_run=s['decoded'].strip(),following_gloss_evidence=below,status='printed interlinear run retained; word/gloss alignment pending',note='No lemma, morphological expansion, or lexical gloss inferred from the surface translation. This is a coverage inventory, not an approved output row.'))
(P/'chapter4-interlinear-runs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
(P/'chapter4-interlinear-census.json').write_text(json.dumps(dict(status='complete physical interlinear-run census; semantic alignment pending',parent_expressions=181,printed_runs=len(rows),source_sha256=hashlib.sha256((P/'chapter4-expressions-first-reading.jsonl').read_bytes()).hexdigest(),output_sha256=hashlib.sha256((P/'chapter4-interlinear-runs.jsonl').read_bytes()).hexdigest()),indent=2)+'\n')
print(len(rows),'physical interlinear form runs, preserved without guessed alignment')
