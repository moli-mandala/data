"""Independent structural conservation verifier; never runs owner builders or modifies inputs."""
from pathlib import Path
from collections import Counter, defaultdict
import copy, hashlib, json
HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'specimens'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return [json.loads(s) for s in p.read_text().splitlines()]
def check():
 snapshot=json.loads((HERE/'input-snapshot.json').read_text())
 for name,expected in snapshot['sha256'].items(): assert sha(INPUT/name)==expected,('input drift',name)
 status=json.loads((INPUT/'emission-review-status.json').read_text())
 for name,pin in status['inputs'].items(): assert sha(INPUT/name)==pin
 for name,pin in status['outputs'].items(): assert sha(INPUT/name)==pin
 lines=[r for page in [564,565,568,569] for r in rows(INPUT/f'p{page}-first-reading.jsonl')]
 assert len(lines)==64
 physical=[a for l in lines for a in l['target_atoms']]
 keys=[a['source_unit_key'] for a in physical]
 assert len(keys)==561 and len(set(keys))==561
 bykey={a['source_unit_key']:a for a in physical}
 for line in lines:
  atoms=line['target_atoms']; assert atoms
  assert [a['physical_word'] for a in atoms]==list(range(1,len(atoms)+1))
  assert all(a['printed_page']==line['printed_page'] and a['physical_line']==atoms[0]['physical_line'] for a in atoms)
  assert ' '.join(a['raw_source_form'] for a in atoms)==line['raw_target_line']
  assert ' '.join(a['raw_interlinear_gloss'] for a in atoms)==line['raw_aligned_english_line']
 for page in [564,565,568,569]:
  chosen=[l for l in lines if l['printed_page']==page]
  assert [l['target_atoms'][0]['physical_line'] for l in chosen]==list(range(1,len(chosen)+1))
 corrected=copy.deepcopy(bykey)
 correction=json.loads((INPUT/'owner-literal-corrections-p564.json').read_text())
 for change in correction['corrections']:
  a=corrected[change['source_unit_key']]
  assert a[change['field']]==change['old']
  a[change['field']]=change['new']; a['literal_correction_proposal']=change
 assert sha(INPUT/'owner-literal-corrections-p564.json')==status['correction_proposal_sha256']
 emissions=rows(INPUT/'atomic-emission-proposal.jsonl'); audit=rows(INPUT/'physical-atom-emission-audit.jsonl'); expressions=rows(INPUT/'expression-emission-proposal.jsonl')
 assert len(emissions)==560 and len(audit)==561 and len(expressions)==51
 assert [r['source_unit_key'] for r in audit]==keys
 ef={e['source_unit_key']:e for e in emissions}; assert len(ef)==560
 first='grierson1906kolamibasim:p568:specimen76-line:11:word:8'; tail='grierson1906kolamibasim:p568:specimen76-line:12:word:1'
 assert bykey[first]['raw_source_form']=='Bāp-' and bykey[tail]['raw_source_form']=='phōkān'
 assert bykey[first]['raw_interlinear_gloss']=='Father-' and bykey[tail]['raw_interlinear_gloss']=='near'
 assert tail not in ef and ef[first]['physical_source_keys']==[first,tail]
 assert ef[first]['raw_source_form']=='Bāp-phōkān' and ef[first]['raw_interlinear_gloss']=='Father-near'
 assert ef[first]['source_parts']==[bykey[first],bykey[tail]]
 emitted_physical=[]
 for e in emissions:
  k=e['source_unit_key']; emitted_physical.extend(e['physical_source_keys'])
  assert e['raw_source_record']==bykey[k]
  if k!=first:
   assert e['physical_source_keys']==[k]
   for f in ['raw_source_form','raw_interlinear_gloss','source_scope','printed_page','physical_line','physical_word']:
    assert e[f]==corrected[k][f],(k,f)
 assert emitted_physical==keys
 aliases=[]
 for a in audit:
  k=a['source_unit_key']; assert a['raw_source_record']==bykey[k]
  expected=[first] if k==tail else [k]
  assert a['proposed_entry_keys']==expected
  assert all(target in ef for target in expected)
  if k==tail: aliases.append({'physical_key':k,'emitting_key':first,'kind':'explicit line-break continuation, not inferred variation'})
 assert len(aliases)==1
 flat=[]; expr_details=[]
 for specimen in [75,76]:
  selected=[e for e in expressions if e['specimen']==specimen]
  assert len(selected)==(27 if specimen==75 else 24)
  expected_physical=[k for k in keys if f':specimen{specimen}-line:' in k]
  assert [k for e in selected for k in e['physical_source_keys']]==expected_physical
  depth=0
  for index,e in enumerate(selected,1):
   ks=e['physical_source_keys']; flat.extend(ks)
   assert e['source_unit_key']==ks[0]+':expression'
   assert e['display_sequence']==index
   assert e['first_source_key']==ks[0] and e['last_source_key']==ks[-1]
   atoms=[corrected[k] for k in ks]
   assert e['raw_physical_atoms']==atoms
   assert e['printed_pages']==sorted({a['printed_page'] for a in atoms})
   forms=[a['raw_source_form'] for a in atoms]; gloss=[a['raw_interlinear_gloss'] for a in atoms]
   if first in ks:
    j=ks.index(first); assert ks[j+1]==tail
    forms[j:j+2]=['Bāp-phōkān']; gloss[j:j+2]=['Father-near']
   assert e['raw_source_form']==' '.join(forms)
   assert e['raw_aligned_interlinear_gloss']==' '.join(gloss)
   for k in ks:
    form=corrected[k]['raw_source_form']; before=depth
    depth+=form.count('‘')+form.count('“')-form.count('’')-form.count('”')
    assert depth>=0,(k,'quote balance')
    if k!=ks[-1]:
     assert not (depth==0 and (form.endswith(('.', '?','!')) or (form.count('’')+form.count('”')>0))),('missed explicit boundary',k)
   last=corrected[ks[-1]]['raw_source_form']
   assert depth==0,('expression split inside quotation',e['source_unit_key'])
   assert last.endswith(('.', '?','!')) or last.count('’')+last.count('”')>0 or e==selected[-1]
   expr_details.append({'key':e['source_unit_key'],'physical_keys':ks,'pages':e['printed_pages'],'first_key':ks[0],'last_key':ks[-1],'boundary_evidence':e['boundary_evidence'],'original_span_review':'Source punctuation/quote span reviewed; no physical-line-to-sentence assertion.'})
 assert flat==keys and Counter(flat)==Counter(keys)
 cross=[e for e in expressions if len(e['printed_pages'])>1]
 assert [(e['specimen'],e['printed_pages']) for e in cross]==[(75,[564,565]),(76,[568,569])]
 duplicate=defaultdict(list)
 for e in emissions: duplicate[(e['source_scope'],e['raw_source_form'],e['raw_interlinear_gloss'])].append(e['source_unit_key'])
 duplicates=[{'source_scope':k[0],'form':k[1],'gloss':k[2],'separate_occurrence_keys':v} for k,v in duplicate.items() if len(v)>1]
 assert all(a.get('accepted_for_installation') is False for a in audit)
 return {'status':'passed_structural_conservation_on_pinned_proposals','reviewer':'/root/asur_finish','input_sha256':snapshot['sha256'],'counts':{'physical_lines':64,'physical_atoms':561,'atomic_emissions':560,'explicit_continuation_aliases':1,'expressions':51,'specimen75_expressions':27,'specimen76_expressions':24,'cross_page_expressions':2,'repeated_exact_form_gloss_scope_groups':len(duplicates)},'line_accounting':[{'key':l['source_unit_key'],'page':l['printed_page'],'atom_keys':[a['source_unit_key'] for a in l['target_atoms']],'source_scope':l['source_scope']} for l in lines],'continuation_aliases':aliases,'expression_partitions':expr_details,'repeated_occurrences':duplicates,'checks':['Every physical atom unique and order-preserved exactly once in each independent atomic and expression partition','Physical audit equals immutable first readings, including old correction history','Only four pinned owner literal corrections applied, without silent gloss or key edits','Target and aligned English reconstructed from contributing paired source atoms; only explicit Bāp-/phōkān join','Every output endpoint and continuation alias resolves; no untracked atom disappearance','Quoted expressions cannot split at physical lines/pages or internal terminal punctuation','No cross-lect duplicate collapse; repeated attestations retain distinct physical keys'],'limits':['This verifier checks conservation and source-bounded expression policy, not final acceptance of every fine glyph, grammar tag, language/reference/profile field.','Current source-supported atomic alignment and whole spans visually reviewed separately; original interlinear anomalies are not repaired semantically.','Final owner glyph overlays require new hash snapshot and rerun; no canonical input modified.']}
if __name__=='__main__':
 result=check(); (HERE/'structural-conservation-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'counts':result['counts']}))
