from pathlib import Path
import json
H=Path(__file__).resolve().parent
es=json.loads((H/'independent-physical-events.json').read_text())
out=[]
for i,e in enumerate(es):
 if e['marker']!='p' or e['section']=='C.5':continue
 following=es[i+1] if i+1<len(es) else None
 g=following['content'] if following and following['marker']=='g' else ''
 form=e['content']
 if e['section']=='C.4' and e['printed_number']=='20':
  form,g=form.split(' g ',1)
 out.append({'index':len(out)+1,'key':e['physical_key'],'section':e['section'],'number':e['printed_number'],'form':form,'gloss':g,'form_tokens':form.split(),'gloss_tokens':g.split()})
(H/'continuous-review-input.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for x in out:print(f"{x['index']} {x['section']}:{x['number']} {len(x['form_tokens'])}/{len(x['gloss_tokens'])}\nP {x['form']}\nG {x['gloss']}")
