"""Expand only source-parenthesized optional material or explicit token-local slash lists."""
import json,re,copy
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[json.loads(l) for l in (P/'witness-relations-proposed.jsonl').open()];decisions=[]
for r in rows:
 if not r['source_record'].get('alternate_segmentation_pending'):continue
 assert len(r['candidates'])==1
 c=r['candidates'][0];form=c['form'];ipa=c['phonetic'];forms=[];ipas=[]
 m=re.search(r'\(([^()]*)\)',form)
 if m:
  assert form.count('(')==1
  forms=[(form[:m.start()]+form[m.end():]).strip(),(form[:m.start()]+m.group(1)+form[m.end():]).strip()]
  im=re.search(r'\(([^()]*)\)',ipa)
  if im:ipas=[ipa[:im.start()]+ipa[im.end():],ipa[:im.start()]+im.group(1)+ipa[im.end():]]
  elif ipa:
   assert form=='p(ʰ)en' and ipa=='pʰen';ipas=['',ipa]
  else:ipas=['','']
  decision='Source explicitly parenthesizes optional material; expand absence/presence only. Printed raw response remains in the audit.'
 else:
  tokens=form.split(' ');amb=[(i,t) for i,t in enumerate(tokens) if '/' in t];assert len(amb)==1
  i,t=amb[0]
  for alt in t.split('/'):
   assert alt;parts=tokens.copy();parts[i]=alt;forms.append(' '.join(parts));ipas.append('')
  assert not ipa
  decision='Source prints a token-local slash list; substitute only each printed alternative in the unchanged full response frame.'
 out=[]
 for n,(f,ip) in enumerate(zip(forms,ipas),1):
  a=copy.deepcopy(c);a.update(entry_key=c['entry_key']+f':answer{n}',form=f,phonetic=ip,status='source alternate proposed')
  a['notes']=(a['notes']+' '+('Subject pronoun is optional in the source.' if form.startswith('(ho)') else 'The source marks the initial interrogative particle as optional.' if form.startswith('(ôyô)') else '')).strip()
  if n>1:a['variant_of_key']=out[0]['entry_key'];a['tags']=list(dict.fromkeys(a['tags']+['alternate']))
  out.append(a)
 r['candidates']=out;decisions.append(dict(source_unit_key=r['source_unit_key'],raw_form=form,raw_phonetic=ipa,forms=forms,phonetic=ipas,reason=decision))
(P/'witness-expanded-proposed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(P/'explicit-alternate-decisions.json').write_text(json.dumps(dict(status='independent review pending',decisions=decisions),ensure_ascii=False,indent=2)+'\n')
print('units',len(rows),'witnesses',sum(len(r['candidates']) for r in rows),'expanded',len(decisions))
