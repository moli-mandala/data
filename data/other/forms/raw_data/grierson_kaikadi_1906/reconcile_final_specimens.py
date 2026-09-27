"""Apply independent original-image corrections without changing pinned owner input."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
names=['specimens-reviewed.jsonl','independent-full-specimens-review-20260926.json','independent-specimens-p339-342-review-20260926.json']
hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in names}
units=[json.loads(x)for x in(P/names[0]).read_text().splitlines()]
reports=[json.loads((P/n).read_text())for n in names[1:]]
assert all(r['input_sha256']==hashes[names[0]]for r in reports)
covered=set(reports[0]['reviewed_keys'])|{a['source_unit_key']for a in reports[1]['coverage']}
assert covered=={a['source_unit_key']for a in units}
lookup={a['source_unit_key']:a for a in units}
for correction in reports[0]['corrections']:
 a=lookup[correction['key']];assert a['forms']==correction['before']
 a['forms']=correction['after'];a['independent_correction']=correction
 if a['source_unit_key']=='grierson1906kaikadi:p337:specimen:line1:word1':
  a['uncertainty']='The original clearly has ā; provisional ḷ is retained, but a separate l underdot is not secure against the noisy background. Literal l/ḷ remains unresolved.'
  a['note']=' '.join(filter(None,[a.get('note'),a['uncertainty']]))
for a in units:a['review']='complete owner and independent original-grayscale reading reconciled'
out=P/'specimens-final-reviewed.jsonl';out.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n'for a in units))
report={'status':'all706sourceunits independently reviewed and three literal corrections reconciled','input_hashes':hashes,'corrections':reports[0]['corrections'],'output':'specimens-final-reviewed.jsonl','output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'uncertain_units':['grierson1906kaikadi:p337:specimen:line1:word1'],'first_and_owner_readings_unchanged':True}
(P/'independent-specimen-reconciliation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
