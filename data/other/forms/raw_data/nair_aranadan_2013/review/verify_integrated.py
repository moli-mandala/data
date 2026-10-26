"""Independent exhaustive Aranadan proposal validator; no importer calls or canonical writes."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,ast
H=Path(__file__).resolve().parent;P=H.parent
FIRST_SHA='22fe3f77dd063609585a70cbe032392e143c955dc237df471a4c16dedf316136'
METADATA_SHA='3b378f91e92896a580efff61a4ee6f6cb397d8be518bc8b0d3397ad977a8d51a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def verify(directory):
 first=P/'first-reading/complete-first-reading.jsonl';meta=H/'metadata-graph-proposal.json'
 assert sha(first)==FIRST_SHA and sha(meta)==METADATA_SHA,'Reviewed authority changed'
 originals=jl(first);proposal=json.loads(meta.read_text());units={r['source_unit_key']:r for r in originals}
 ow={w['witness_key']:w for u in originals for w in u['witnesses']}
 metadata={(d['witness_key'],d['field']):d for d in proposal['keyed_field_proposals']}
 assert len(units)==97 and len(ow)==105 and len(metadata)==90
 status=json.loads((directory/'review-status.json').read_text())
 for name,pin in status['inputs'].items():assert sha(P/name)==pin,('input hash',name)
 for name,pin in status['outputs'].items():assert sha(directory/name)==pin,('output hash',name)
 approvals=[]
 for name in status['inputs']:
  if name.startswith('root-review/'):
   a=json.loads((P/name).read_text());assert a['approved'] is True
   assert a['first_reading_sha256']==FIRST_SHA
   for e,pin in a.get('evidence_sha256',{}).items():assert sha(P/e)==pin,('approval evidence',e)
   approvals.append(a)
 assert approvals,'No root-approved overlays pinned'
 approved={(d['witness_key'],d['field']):d for a in approvals for d in a.get('decisions',[]) if d['decision']=='accepted'}
 for pair,d in metadata.items():
  assert pair in approved and approved[pair]['old']==d['old'] and approved[pair]['new']==d['new'],('metadata approval',pair)
 for pair,d in approved.items():
  assert pair[1] in ('tags','note','typed_uncertainty'),('unexpected lexical/routing change',pair)
 rows=list(csv.reader((directory/'review-candidate.csv').open(newline='')));assert len(rows)==102
 assert all(len(r)==15 for r in rows)
 out={r[10]:r for r in rows};assert len(out)==102
 assert Counter(r[0] for r in rows)==Counter(Aranadan=81,Malayalam=20,Kannada=1)
 excluded={w['witness_key'] for u in originals if u['section']=='ethnonyms' for w in u['witnesses']}
 assert len(excluded)==4 and not(excluded & out.keys())
 additional=proposal['additional_source_witnesses'][0];extra=additional['proposed_key']
 assert set(out)==set(ow)-excluded|{extra},'Output occurrence key loss/addition'
 audit=jl(directory/'review-audit.jsonl');assert len(audit)==97
 aud={a['source_unit_key']:a for a in audit};assert set(aud)==set(units)
 assert sum(a['physical_witness_slots'] for a in audit)==105
 assert Counter(k for a in audit for k in a['review_candidate_keys'])==Counter(out.keys())
 aw={w['witness_key']:w for a in audit for w in a['source_record']['witnesses']}
 assert set(aw)==set(ow)|{extra}
 for k,w in ow.items():
  assert aw[k]['form']==w['form'] and aw[k]['gloss']==w['gloss'],('source literal drift',k)
  assert aw[k]['language']==w['language'],('raw historical language label changed',k)
  if k in excluded:continue
  r=out[k];assert r[0]==w['language'] and r[2]==w['form'] and r[3]==w['gloss'],('export lexical drift',k)
  tags=metadata.get((k,'tags'),{}).get('new',w['tags']);note=metadata.get((k,'note'),{}).get('new',w['note'])
  uncertainty=approved.get((k,'typed_uncertainty'),{}).get('new',w['typed_uncertainty'])
  expected_tags=set(tags)|({'uncertain'} if uncertainty else set())
  assert set(r[14].split())==expected_tags,('tags',k,r[14],expected_tags)
  assert r[6]==note and aw[k]['note']==note,('scholarly Notes',k)
  assert aw[k]['typed_uncertainty']==uncertainty,('typed uncertainty',k)
  unit=next(u for u in originals if any(v['witness_key']==k for v in u['witnesses']))
  expected_ref=f"nair2013aranadan[p. {unit['printed_page']}, {unit['section']}, item {unit['position']}, witness {k.rsplit(':w',1)[1]}]"
  assert r[7]==expected_ref,('citation',k)
 r=out[extra];anchor=out['nair2013aranadan:p10:tense-suffixes:1:w1']
 assert r[0]=='Malayalam' and r[2]=='-um' and r[3]=='' and set(r[14].split())=={'suffix','fut'}
 assert r[7]==anchor[7],'Shared Malayalam attribution invented a physical locator'
 added=[d for a in audit for d in a.get('added_shared_attributions',[])];assert len(added)==1
 assert added[0]['witness']['witness_key']==extra and added[0]['shared_physical_witness_key']==anchor[10]
 assert added[0]['reason'] and added[0]['evidence']
 for k in excluded:
  a=aud[k.rsplit(':w',1)[0]];assert k in a['exclusions'] and not a['review_candidate_keys']
 expected_edges={e['child']:e['parent'] for e in proposal['proposed_edges']}
 assert {r[10]:r[11] for r in rows if r[11]}==expected_edges
 assert len(expected_edges)==2 and all(k in out and v in out for k,v in expected_edges.items())
 assert all(not r[i] for r in rows for i in [1,4,5,8,9,12,13]),'Invented native, phonemic, ID or graph field'
 expressions=[u for u in originals if u['kind']=='whole-translated-expression']
 assert len(expressions)==10
 for u in expressions:
  for w in u['witnesses']:assert 'multiword-expression' in out[w['witness_key']][14].split()
 paradigm=[u for u in originals if u['section']=='verb-paradigm'];assert len(paradigm)==32
 for u in paradigm:
  r=out[u['witnesses'][0]['witness_key']];column=str(u['position']).split(':')[-1]
  assert set(r[14].split())=={'verb',{'stem':'stem','past':'pret','present':'pres','future':'fut'}[column]}
 for pos in [1,2]:
  r=out[f'nair2013aranadan:p10:pronoun-prose:{pos}:w1'];assert set(r[14].split())=={'pron','1pl'}
  assert ('large' if pos==1 else 'small') in r[6]
 assert all(not({'inclusive','exclusive','loanword','derived'} & set(r[14].split())) for r in rows)
 W=next(p for p in H.parents if (p/'data/tags.py').exists())
 registry=next(ast.literal_eval(n.value) for n in ast.parse((W/'data/tags.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRAMMATICAL_TAGS' for t in n.targets))
 assert all(t in registry for r in rows for t in r[14].split())
 return dict(status='passed',reviewer='asur_finish',rows=102,languages=dict(Counter(r[0] for r in rows)),units=97,physical_witnesses=105,documentary_ethnonyms=4,metadata_changes_checked=90,shared_physical_attributions=1,variant_edges=2,whole_expressions=10,paradigm_cells=32,hashes={p.name:sha(p) for p in [first,meta,directory/'review-candidate.csv',directory/'review-audit.jsonl',directory/'review-status.json']},scope='Exhaustive source-stage corpus comparison only; not a database build or browser test.')
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,default=P/'integration');ap.add_argument('--report',type=Path);args=ap.parse_args()
 result=verify(args.directory);text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
 if args.report:
  assert args.report.resolve().is_relative_to(H.resolve()),'Report must stay inside review/'
  args.report.write_text(text)
 print(text)
