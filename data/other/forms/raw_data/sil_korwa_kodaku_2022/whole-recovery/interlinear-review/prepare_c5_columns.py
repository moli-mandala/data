"""Source-column proposal, manually scoped against original printed pp95–98.

This is not equal-token-count alignment. Explicit group lengths preserve two
unresolved spans; every target token and every gloss token remains accounted.
"""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
events=json.loads((H/'independent-physical-events.json').read_text())
grammar_overlay=json.loads((H/'c5-source-grammar-overlay.json').read_text())
grammar_by_key={r['source_unit_key']:r for r in grammar_overlay['records']}
assert len(grammar_by_key)==len(grammar_overlay['records'])==26
grammar_used=set()
# Each pipe marks a reviewed source column; commas in lengths join target tokens.
spec={
'1':('when|I|was|small|then',None),
'2':('one|day|boarding|school-from|house|went-I',None),
'3':('home-at|that|day|it-so-happened|heavy|rain|came',None),
'4':('so|one|day|morning-in|all|people|shout-began|wolf|wolf',None),
'5.1':('then|village-from|people|bow-arrow|long-axe|axe|sticks',None),
'5.2':('these|all|holding|they-came-out',None),
'6.1':('and|those hunters-who-were|they|much|bow-arrow|they-came|ahead|ahead',[1,3,1,1,1,1,1,1]),
'6.2':('they-moved-around',None),
'7':('village-from|women|unmarried-girls|and|children|even|many|gathered-we',None),
'8.1':('and|then|some|hunters|forward|they-went|this|jute|shrub-in',None),
'8.2':('what-is-it',None),
'9':('whenever|jute|shrub|shakes|then|fear-in|it-moves',None),
'10':('again|and|jute|shrub|shakes|then|and|in-fear|more|it-runs',None),
'11':('after-that|some|some|of-them|pressed-ahead-they|and|forward|went',None),
'12.1':('after-that|when|while searching for-it then jute shrub|shaking|shaking',[1,1,5,1,1]),
'12.2':('forward|went-it',None),
'13':('that-time-at|someone|exclaimed|oh|my|tiger|it-is|tiger',None),
'14.1':('then|what|happened|all|living|that|place-in|all|living|very',None),
'14.2':('house-to|running|quick|quick|quick|quick',None),
'15':('so|that|place-from|one|animal|(PAUSE)|came-out-it',None),
'16.1':('so|where-from|tiger|or|where-from|wolf|one|jackal',None),
'16.2':('only-it-was',None),
'17.1':('that|moment|more|some|hunters|listen|listen|big',None),
'17.2':('this|tiger|-not-is|jackal|it-is|to-this|fear-we|that-said|to|it|chasing',None),
'17.3':('started-they',None),
'18.1':('and|chasing|chasing|started|then|that|jackal|(emph)|another',None),
'18.2':('shrub|went-to',None),
'19.1':('and|when|they|it|overcame|then|that|jackal|(emph)',None),
'19.2':('one|(pause)|hole-in|entered (drawn out for emphasis)',None),
'20.1':('so|look|you|please|people|fear|how-was-it|they|of-me|leaving',None),
'20.2':('fled-they',None),
}
out=[];coverage=[]
for number,(label,lengths) in spec.items():
 es=[e for e in events if e['section']=='C.5' and e['printed_number']==number and e['question_number'] is None]
 ps=[e for e in es if e['marker']=='p'];gs=[e for e in es if e['marker']=='g'];assert len(ps)==len(gs)==1
 p,g=ps[0],gs[0];tokens=p['content'].split();glosses=label.split('|')
 assert ' '.join(glosses)==g['content'],(number,g['content'])
 lengths=lengths or [1]*len(glosses)
 assert sum(lengths)==len(tokens) and len(lengths)==len(glosses),(number,tokens,lengths)
 offset=0;keys=[]
 for index,(length,gloss) in enumerate(zip(lengths,glosses),1):
  form=' '.join(tokens[offset:offset+length]);key=f'behera2022korwakodaku:C.5:column:{number}:{index}'
  ambiguity=(number=='6.1' and index==2) or (number=='12.1' and index==3)
  role='unaligned-source-span' if ambiguity else 'source-aligned-column'
  analysis_only=gloss in ('(pause)','(PAUSE)','(emph)')
  out.append(dict(source_unit_key=key,section='C.5',printed_number=number,source_pages=[p['pdf_page']],form=form,raw_gloss=gloss,gloss='' if analysis_only else gloss,source_analysis=gloss if analysis_only else '',language='Sadri',role=role,physical_p_key=p['physical_key'],physical_g_key=g['physical_key'],target_token_range=[offset+1,offset+length],whole_expression_key=f'behera2022korwakodaku:C.5:expression:{number.split(".")[0]}',alignment_status='Internal lexical alignment unresolved; literal contiguous span retained without distributing gloss.' if ambiguity else 'Printed column alignment visually reviewed.',typed_uncertainty=['source-word-alignment'] if ambiguity else [],accepted_for_installation=False))
  row=out[-1];row['tags']=[];row['notes']=[]
  if number=='1' and index in (3,4):
   row['typed_uncertainty']=['source-word-alignment']
   row['notes'].append("The printed columns pair tʃotʰeɾɐ with 'was' and holo with 'small'; the anomalous pairing is preserved without reassignment or inferred part of speech.")
  if number=='19.2' and index==4:
   assert row['gloss']=='entered (drawn out for emphasis)'
   row['gloss']='entered';row['source_analysis']='drawn out for emphasis'
  if gloss=='(emph)':row['tags'].append('emph')
  if row['typed_uncertainty']:row['tags'].append('uncertain')
  if key in grammar_by_key:
   addition=grammar_by_key[key]
   assert form==addition['expected_form'] and gloss==addition['expected_raw_gloss'],key
   row['tags']=list(dict.fromkeys(row['tags']+addition['add_tags']))
   row['typed_uncertainty']=list(dict.fromkeys(row['typed_uncertainty']+addition.get('add_typed_uncertainty',[])))
   if addition.get('add_note'):row['notes'].append(addition['add_note'])
   grammar_used.add(key)
  keys.append(key);offset+=length
 assert offset==len(tokens)
 coverage.append(dict(printed_number=number,physical_p_key=p['physical_key'],physical_g_key=g['physical_key'],target_tokens=len(tokens),children=keys,all_target_tokens_accounted=True,all_gloss_tokens_accounted=True))
assert len(coverage)==31
assert grammar_used==set(grammar_by_key)
(H/'c5-column-proposal.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
(H/'c5-column-coverage.json').write_text(json.dumps(dict(status='Original-reviewed source-column proposal; independent acceptance pending',physical_pairs=31,child_rows=len(out),unresolved_internal_spans=[r['source_unit_key'] for r in out if r['role']=='unaligned-source-span'],printed_alignment_conflicts=[r['source_unit_key'] for r in out if r['printed_number']=='1' and r['typed_uncertainty']],events_sha256=hashlib.sha256((H/'independent-physical-events.json').read_bytes()).hexdigest(),coverage=coverage),ensure_ascii=False,indent=2)+'\n')
print(len(out),'column/group rows, 31 pairs, 2 unresolved internal spans')
