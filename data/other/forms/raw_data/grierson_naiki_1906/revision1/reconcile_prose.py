"""Apply the pinned independent Naiki prose review without rewriting first readings."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reconcile(directory=HERE):
 review=json.loads((directory/'independent-prose-review.json').read_text())
 sources={}
 for name,digest in review['input_sha256'].items():
  assert sha(directory/name)==digest, f'Stale independent review: {name}'
  sources[name]=[json.loads(s) for s in (directory/name).read_text().splitlines()]
 indexed={r['source_unit_key']:r for rows in sources.values() for r in rows}
 assert len(indexed)==94
 assert {r['source_unit_key'] for r in review['coverage']}==set(indexed)
 for correction in review['corrections']:
  row=indexed[correction['source_unit_key']];field=correction['field']
  assert row[field]==correction['before'], correction
  row[field]=correction['after']
  row['independent_correction']=correction
  if row['source_unit_key'].endswith(':done-phrase'):
   row['source_caveat']='Independent original glyph review confirms plain t in this occurrence. The earlier dotted reading is retained in the immutable first reading.'
 for row in indexed.values():
  row['review_status']='Independent original literal/gloss/scope review reconciled; final output review pending'
  row['independent_review']='independent-prose-review.json'
 for added in review['omitted_units']:
  key=added['proposed_source_unit_key'];assert key not in indexed
  row=dict(source_unit_key=key,printed_page=added['original_page'],physical_djvu_page=added['original_page']+20,
   raw_form_group=added['raw_form_group'],raw_gloss_or_analysis=added['raw_gloss_or_analysis'],
   disposition=added['disposition'],source_caveat=added['source_caveat'],source_quote=added['source_quote'],
   review_status='Independent original omission recovery; final output review pending',independent_review='independent-prose-review.json',canonical_changed=False)
  sources['prose-first-reading.jsonl'].append(row);indexed[key]=row
 # Preserve the comparative attribution and tentative case claim explicitly.
 for row in sources['prose-first-reading.jsonl']:
  if row['disposition']=='control-Kui':
   row['source_caveat']+=' Source additionally says the same forms are usual in Kōlāmī; no extra target-language form is inferred.'
  if row['source_unit_key'].endswith(':case-suffixes'):
   row['source_caveat']+=' Source qualifies the syncretism: dative and accusative do not appear to be distinguished.'
 outputs={}
 for source,target in [('prose-first-reading.jsonl','prose-reviewed.jsonl'),('pronoun-table-first-reading.jsonl','pronoun-table-reviewed.jsonl')]:
  (directory/target).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sources[source]))
  outputs[target]=sha(directory/target)
 result=dict(status='Independent prose corrections reconciled; not whole-source installation approval',
  input_sha256=review['input_sha256'],review_sha256=sha(directory/'independent-prose-review.json'),
  corrected_fields=review['corrections'],added_units=review['omitted_units'],source_context_requirements=review['context_requirements'],
  output_sha256=outputs,counts=dict(prose_groups=84,pronoun_cells=11,literal_corrections=2,added_morphology=1),canonical_changed=False)
 (directory/'prose-reconciliation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(reconcile(),ensure_ascii=False))
