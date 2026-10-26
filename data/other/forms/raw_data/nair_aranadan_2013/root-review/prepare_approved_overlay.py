"""Freeze root-reviewed whole-chapter reconciliation; no canonical installation."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
evidence={
 'first-reading/complete-first-reading.jsonl':'22fe3f77dd063609585a70cbe032392e143c955dc237df471a4c16dedf316136',
 'review/metadata-graph-proposal.json':'3b378f91e92896a580efff61a4ee6f6cb397d8be518bc8b0d3397ad977a8d51a',
 'independent-reading/complete-independent-original-review.json':'19e107c63d77670dd250587e7f77a08de3b124b1b9f3e95f12afc0aa48007e28',
}
for name,digest in evidence.items():assert sha(P/name)==digest,name
for name in ['root-review/ethnonym-disposition.json','root-review/transcription-and-provenance.json']:
 evidence[name]=sha(P/name)
first=[json.loads(x) for x in (P/'first-reading/complete-first-reading.jsonl').read_text().splitlines()]
units={r['source_unit_key']:r for r in first}
witnesses={w['witness_key']:w for r in first for w in r['witnesses']}
proposal=json.loads((P/'review/metadata-graph-proposal.json').read_text())
decisions=[]
for d in proposal['keyed_field_proposals']:
 assert witnesses[d['witness_key']][d['field']]==d['old']
 decisions.append(dict(d,decision='accepted'))
assert len(decisions)==90
excluded=[]
for key in json.loads((P/'root-review/ethnonym-disposition.json').read_text())['source_unit_keys']:
 excluded.append({'source_unit_key':key,'expected_witness_keys':[w['witness_key'] for w in units[key]['witnesses']],
                  'reason':'Community naming discussion; lexical language attribution not established. Preserve exact source witnesses in documentary audit.'})
assert len(proposal['additional_source_witnesses'])==1
item=proposal['additional_source_witnesses'][0]
anchor=item['physical_locator']+':w1'
assert witnesses[anchor]['form']==item['form']=='-um'
added=[{'source_unit_key':item['physical_locator'],'shared_physical_witness_key':anchor,
 'expected_physical_form':'-um','witness':{'witness_key':item['proposed_key'],'form':'-um','gloss':'','language':'Malayalam',
 'tags':['suffix','fut'],'note':'The source explicitly identifies the same future suffix in Aranadan and Malayalam.', 'typed_uncertainty':None},
 'reason':item['basis'],'evidence':'Original p10 future-tense paragraph; review/metadata-graph-proposal.json'}]
links=[]
for edge in proposal['proposed_edges']:
 assert edge['type']=='variant'
 links.append({'child_key':edge['child'],'parent_key':edge['parent'],
 'expected_child_form':witnesses[edge['child']]['form'],'expected_parent_form':witnesses[edge['parent']]['form'],
 'reason':edge['basis']+' Second listed form points to first for display; no chronological priority inferred.',
 'evidence':'Original p9 medial-consonant comparison; review/metadata-graph-proposal.json'})
assert len(links)==2
out={'approved':True,'scope':'Full reviewed chapter reconciliation only; output audit/parser/installation gates remain',
 'first_reading_sha256':evidence['first-reading/complete-first-reading.jsonl'],'evidence_sha256':evidence,
 'decisions':decisions,'unit_exclusions':excluded,'added_witnesses':added,'variant_links':links}
target=P/'root-review/approved-reconciliation.json'
text=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
assert not target.exists() or target.read_text()==text
target.write_text(text)
print(json.dumps({'approved_fields':len(decisions),'excluded_name_units':len(excluded),'shared_attribution_rows':len(added),'variant_edges':len(links),'sha256':sha(target)}))
