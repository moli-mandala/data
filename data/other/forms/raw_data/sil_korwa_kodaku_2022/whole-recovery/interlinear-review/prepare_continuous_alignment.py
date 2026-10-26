"""Validate manual source-semantic partitions; preserve every raw token exactly."""
from pathlib import Path
import json,re,hashlib
from continuous_alignment_specs import SPECS
H=Path(__file__).resolve().parent
inputs=json.loads((H/'continuous-review-input.json').read_text())
assert len(inputs)==85 and set(SPECS)==set(range(1,86))
out=[];coverage=[];errors=[]
for x in inputs:
 label,lengths=SPECS[x['index']];glosses=label.split('|');lengths=lengths or [1]*len(glosses)
 if ' '.join(glosses)!=x['gloss'] or sum(lengths)!=len(x['form_tokens']) or len(lengths)!=len(glosses):
  errors.append({'index':x['index'],'expected_p':len(x['form_tokens']),'proposed_p':sum(lengths),'gloss_equal':' '.join(glosses)==x['gloss'],'groups':len(lengths),'gloss_groups':len(glosses)})
  continue
 offset=0;keys=[]
 for index,(length,raw_gloss) in enumerate(zip(lengths,glosses),1):
  form=' '.join(x['form_tokens'][offset:offset+length]);key=x['key']+f':semantic-span:{index}'
  unresolved=length>1
  tags=[];notes=[];gloss=raw_gloss
  # Only explicit parenthetical source labels are structured here. No tense/POS
  # is inferred merely from an English inflected verb or modal paraphrase.
  if not unresolved:
   for label,tag in [('past','pret'),('present','pres'),('future','fut')]:
    if f'({label})' in gloss:
     tags.append(tag);gloss=gloss.replace(f'({label})','').strip()
     notes.append(f'Source interlinear analysis explicitly labels this form {label}.')
  uncertainty=['source-word-alignment'] if unresolved else []
  if raw_gloss in ('?','cm','(cm)'):
   gloss='';uncertainty.append('source-analysis-unresolved')
   notes.append('Source analysis is retained literally without expanding an undefined label or supplying a lexical gloss.')
  if unresolved:
   gloss='';notes.append('The source gives a continuous gloss span; internal word-to-gloss alignment is unresolved. Raw target and English spans are preserved together.')
  if 'sa.b' in form:
   uncertainty.append('source-notation')
   notes.append('The source prints sa.b with a baseline dot; its phonetic interpretation is unspecified. The dot is retained literally.')
  if uncertainty:tags.append('uncertain')
  out.append(dict(source_unit_key=key,physical_p_key=x['key'],section=x['section'],printed_number=x['number'],source_pages=[int(x['key'].split(':pdf')[1].split(':')[0])],language={'C.2':'kw','C.3':'Kodaku','C.4':'Sadri'}[x['section']],form=form,gloss=gloss,raw_interlinear_gloss=raw_gloss,target_token_range=[offset+1,offset+length],tags=tags,notes=notes,typed_uncertainty=uncertainty,role='unresolved-internal-alignment-span' if unresolved else 'proposed-source-semantic-alignment',whole_expression_key=f"behera2022korwakodaku:{x['section']}:expression:{x['number']}",review_basis='Manual source-scoped semantic partition of printed ordered p/g lines; not automatically zipped by token count.',accepted_for_installation=False))
  keys.append(key);offset+=length
 coverage.append(dict(index=x['index'],physical_p_key=x['key'],target_tokens=len(x['form_tokens']),gloss_tokens=len(x['gloss_tokens']),all_tokens_preserved=True,child_keys=keys))
assert not errors,errors
assert len(coverage)==85
from reviewed_grammar_overlay import apply
apply(out)
(H/'continuous-semantic-proposal.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
report=dict(status='Owner-reviewed semantic proposal; independent alignment review required before acceptance',blocks=85,rows=len(out),single_token_proposals=sum(r['role']=='proposed-source-semantic-alignment' for r in out),unresolved_spans=sum(r['role']=='unresolved-internal-alignment-span' for r in out),all_target_and_gloss_tokens_preserved=True,inputs_sha256=hashlib.sha256((H/'continuous-review-input.json').read_bytes()).hexdigest(),spec_sha256=hashlib.sha256((H/'continuous_alignment_specs.py').read_bytes()).hexdigest(),coverage=coverage)
(H/'continuous-semantic-coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='coverage'})
