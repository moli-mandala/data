"""Conservative typographic candidate extraction; installation requires the audited decisions."""
import collections,csv,json,re,unicodedata
from pathlib import Path
from languages import MAP,PAT
RAW=Path(__file__).parent
if (RAW/'language-map-proposed.json').exists():
 for label,v in json.loads((RAW/'language-map-proposed.json').read_text()).items():MAP[label]=(v['language'],v['dialect'])
 PAT=re.compile(r'(?<![\w.])('+ '|'.join(map(re.escape,sorted(MAP,key=len,reverse=True)))+r')(?=\s|[,;:)—]|$)')
GRAM=r'(?:n\.[mfn]\.|n\.|adj\.|adv\.|v\.[ti]\.|v\.|m\.|f\.|pl\.|sg\.|poet\.|lit\.|pron\.|pp\.|p\.p\.|tr\.|intr\.|v\.t\.|v\.i\.|caus\.|inf\.|ptc\.|past|pres\.|imper\.|aor\.|ind\.pres\.|n\.m\.|n\.f\.|n\.pl\.|aux\.)'

class Quote:
 def __init__(self,text,a,b):self.text=text;self.a=a;self.b=b
 def start(self):return self.a
 def end(self):return self.b
 def __getitem__(self,key):return self.text[self.a+1:self.b-1] if key==1 else self.text[self.a:self.b]

def quoted_definitions(text):
 out=[];stack=[]
 for i,c in enumerate(text):
  if c=='‘':stack.append(i)
  elif c=='’' and stack:
   if re.match(r's\b',text[i+1:]):continue
   start=stack.pop()
   if not stack:out.append(Quote(text,start,i+1))
 return out

def flatten(rec):
 ts=[]
 for t in rec['tokens']:
  if t['style']=='sup':
   if ts and ts[-1]['style']=='f' and not ts[-1]['text'].endswith(' ') and t['text'].strip().isdigit() and int(t['text'])<=55:
    ts[-1]['text']+=t['text'].translate(str.maketrans('0123456789','⁰¹²³⁴⁵⁶⁷⁸⁹'));continue
   ts.append(dict(style='r',text=' '));continue
  t=dict(t)
  if ts and ts[-1]['style']==t['style']:ts[-1]['text']+=t['text']
  else:ts.append(t)
 # Small roman hyphens and superscripts embedded in an italic word belong to it.
 changed=True
 while changed:
  changed=False
  for i in range(1,len(ts)-1):
   if ts[i-1]['style']==ts[i+1]['style']=='f' and re.fullmatch(r'[-*\d ()\[\]]*',ts[i]['text']):
    ts[i-1]['text']+=ts[i]['text']+ts[i+1]['text'];ts[i:i+2]=[];changed=True;break
 text='';spans=[]
 for t in ts:
  start=len(text);text+=t['text']
  if t['style']=='f':
   a=len(t['text'])-len(t['text'].lstrip());b=len(t['text'].rstrip());spans.append((start+a,start+b,t['text'].strip()))
 return text,spans

def candidates(rec):
 text,spans=flatten(rec)
 quotes=quoted_definitions(text)
 # Parenthesized comparisons have their own language scope; returning to the
 # main clause must not carry the comparison language into subsequent forms.
 stack=[];containers=[]
 quoted={i for q in quotes for i in range(q.start(),q.end())}
 for pos,ch in enumerate(text):
  if pos in quoted:continue
  if ch in '([':stack.append((ch,pos))
  elif ch in ')]' and stack and stack[-1][0]=={')':'(',']':'['}[ch]:
   _,start=stack.pop();containers.append((start,pos))
 def in_scope(pos,target):return all(not (start<pos<end) or start<target<end for start,end in containers)
 citation_spans=[m for m in re.finditer(r'\([^()]*\)|\[[^][]*\]',text) if re.search(r'\d{3,4}|\b(?:parallel|Turner|Sharma|Bhayani|Zoller)\b',m[0])]
 labels=[m for m in PAT.finditer(text) if not any(q.start()<m.start()<q.end() for q in quotes+citation_spans)]
 out=[]
 for i,(a,b,form) in enumerate(spans):
  for lm in labels:
   if lm.start()<=a<lm.end()<b:
    a=lm.end();form=text[a:b].strip()
  if any(q.start()<a<q.end() for q in quotes):continue
  # All candidates remain accounted for, including unglossed explanatory forms.
  after=[q for q in quotes if q.start()>=b];q=after[0] if after else None
  between=text[b:q.start()] if q else ''
  residual=between
  for c,d,_ in spans:
   if c>=b and q and d<=q.start():residual=residual.replace(text[c:d],' ')
  residual=PAT.sub(' ',residual)
  residual=re.sub(GRAM,' ',residual)
  residual=re.sub(r'\([^()]*\)|\[[^][]*\]',' ',residual)
  residual=re.sub(r'\b(?:and|or|all|both|also|with|meaning|meanings|ditto|singular|plural)\b',' ',residual)
  if q and re.search(r'\ball\b',between):residual=re.sub(r'\bbut\b',' ',residual)
  clean_gap=not re.search(r'[A-Za-z‘’<>]',residual)
  if q and not in_scope(a,q.start()):clean_gap=False
  prior=[m for m in labels if m.end()<=a and in_scope(m.start(),a)]
  label=prior[-1][1] if prior else ''
  # Trailing language attribution: form (A., B.) 'gloss'.
  post=[m for m in labels if b<=m.start() and q and m.end()<q.start()]
  if post and not re.search(r'\b(?:all|both)\b',between) and not re.search(r'\band\b',text[b:post[0].start()]):clean_gap=False
  if not label and post:label=post[0][1]
  label_list=[label] if label else []
  if prior and '(' not in text[prior[-1].end():a]:
   for pm in reversed(prior[:-1]):
    nxt=prior[prior.index(pm)+1]
    if re.fullmatch(r'[ ,]*(?:and|or)?[ ,]*',text[pm.end():nxt.start()]) and re.search(r',|and|or',text[pm.end():nxt.start()]):label_list.insert(0,pm[1])
    else:break
  if q:
   for lm in labels:
    if q.end()<=lm.start()<q.end()+40 and re.fullmatch(r'[;, ]*these (?:latter )?',text[q.end():lm.start()]) and re.match(r'\s+forms\b',text[lm.end():]):
     label=lm[1];label_list=[label];break
   for lm in labels:
    if q.end()<=lm.start()<q.end()+12 and re.fullmatch(r'\s+(?:in|of)\s+',text[q.end():lm.start()]):
     label=lm[1];label_list=[label];break
  bur=re.search(r'Bur\.\s*\(([HNY])\.\)\s*$',text[:a])
  if bur:label='Bur.';label_list=['Bur.']
  lang,dialect=MAP.get(label,('unresolved',''))
  if bur:dialect={'H':'Hunza','N':'Nagar','Y':'Yasin'}[bur[1]]
  if label=='Maleng' and re.search(r'Maleng\s*\(Bro\)\s*$',text[:a]):dialect='Bro'
  status='candidate'
  shared=[(c,d) for c,d,_ in spans if c>=b and q and d<=q.start()]
  if shared and not re.search(r',|\bor\b|\band\b|/',text[b:shared[0][0]]) and not re.search(r'\ball\b',between):clean_gap=False
  if not q or not clean_gap or not q[1].strip() or q.start()-b>450:status='context-or-unglossed'
  if lang in ('excluded','reconstruction','comparison-control'):status='comparison-control'
  if not label:status='unresolved-language'
  # Flag a nearer unregistered abbreviation so it cannot inherit a preceding lect.
  recent=text[prior[-1].end():a] if prior else text[:a]
  unknown=re.findall(r'(?<!\w)([A-Z][A-Za-zāīūṭḍṇśṣṅã.-]{1,15}\.)\s',recent)
  unknown=[u for u in unknown if u not in MAP and u not in ('Cf.','Note.','MIA.','OIA.')]
  full_label=re.search(r'(?<![\w.])([A-ZÀ-ÖØ-ÞĀ-Ž][\w-]*(?: [A-ZÀ-ÖØ-ÞĀ-Ž][\w-]*)*)[ =]*$',re.sub(r'\([^()]*\)',' ',recent).rstrip())
  if full_label and full_label[1][0].isupper() and full_label[1] not in MAP and full_label[1] not in ('Note','Cf','VS','PAN','Z'):
   unknown.append(full_label[1])
  if unknown:status='unresolved-language';label=unknown[-1];lang='unresolved';dialect=''
  # An unlabelled etymon after a derivation arrow cannot inherit the reflex lect.
  last_arrow=max([m.start() for m in re.finditer('[<←]',text[:a]) if in_scope(m.start(),a)] or [-1])
  if last_arrow>=0 and (not prior or last_arrow>prior[-1].end()):
   status='unresolved-etymon-language';lang='unresolved';label_list=[]
  if lang=='unresolved':status='unresolved-language'
  if re.search(r'\bmy (?:earlier )?(?:published|recorded|noted)|\bour (?:earlier |own )?(?:form|word|material)',recent,re.I):
   status='unresolved-language';lang='unresolved';label_list=[]
  if prior:
   modifier=re.search(r'\b(Old|Middle|West|East|North|South|Western|Eastern|Northern|Southern|Central)\s+$',text[:prior[-1].start()],re.I)
   if modifier:
    qualifier=modifier[1].title()
    qualifier={'North':'Northern','South':'Southern','West':'Western','East':'Eastern'}.get(qualifier,qualifier)
    if qualifier in ('Old','Middle'):
     status='unresolved-language';lang='unresolved';label=qualifier+' '+label;label_list=[]
    else:
     dialect=qualifier+(' '+dialect if dialect else '')
  if form.startswith('*') or re.search(r'\*\s*[-(\[]?\s*$',text[:a]):status='reconstruction'
  if prior and re.search(r'\b(?:pre-|Proto-)$',text[:prior[-1].start()]):status='reconstruction'
  if lang=='Sk' and status!='reconstruction':status='etymon-evidence'
  gloss=q[1] if q and clean_gap else ''
  gloss=re.sub(r'\s+',' ',gloss).strip()
  if gloss in ('ditto','id.','idem'):
   prevq=[x for x in quotes if x.end()<q.start()]
   if prevq:gloss=prevq[-1][1]
  if text[b:b+1]=='-':form+='-'
  if a and text[a-1]=='-' and (a<2 or not text[a-2].isalnum()):form='-'+form
  if a and text[a-1] in '[(' and {'[':']','(':')'}[text[a-1]] in form:form=text[a-1]+form
  tags=[]
  review=[]
  if gloss.count('(')!=gloss.count(')') or gloss.count('[')!=gloss.count(']'):
   tags.append('uncertain');review.append('gloss: unbalanced punctuation in quoted source definition; no silent repair')
  if '-̇' in form:tags.append('uncertain');review.append('transcription: source places dot above boundary hyphen; preserved diplomatically')
  if q and re.match(r'\s*\(\?\)',text[q.end():]):tags.append('uncertain');review.append('gloss: source question mark')
  # grammatical labels immediately after this form only
  local=text[b:q.start()] if q else ''
  local=local[:max(0,min([c-b for c,d,_ in spans if c>=b] or [len(local)]))]
  trailing_dialect=re.match(r'\s*\[([^][]]+) dialect\]',local)
  leading_dialect=re.search(r'\b([A-Z][\w-]*(?: [A-Z][\w-]*)*) dialect\s*$',text[:a])
  if trailing_dialect:dialect=trailing_dialect[1]
  elif leading_dialect:dialect=leading_dialect[1]
  if shared:local=re.sub(r'\bpl\.','',local)
  if re.search(r'\bpl\.\s*$',text[:a]):tags.append('pl')
  for tok,t in [('n.m.','noun m'),('n.f.','noun f'),('n.n.','noun n'),('v.t.','verb tr'),('v.i.','verb intr'),('adj.','adj'),('adv.','adv'),('pron.','pron'),('pl.','pl')]:
   if tok in local:tags+=t.split()
  for m in re.finditer(r'(?<!\w)(n|m|f|v)\.',local):tags.append({'n':'noun','m':'m','f':'f','v':'verb'}[m[1]])
  context=text[max(0,prior[-1].end() if prior else 0):a]
  if 'poet.' in context and len(context)<60:tags.append('poetic')
  if q and re.match(r'\s*\(vulgar\)',text[q.end():]):tags.append('vulgar')
  out.append(dict(key=rec['key']+f':span{i+1}',record=rec['key'],page=rec['page'],section=rec['section'],span=[a,b],form=form,label=label,labels=label_list,language=lang,dialect=dialect,gloss=gloss,tags=tags,review=review,status=status,context=text[max(0,a-90):min(len(text),b+180)],raw=text))
 return out

if __name__=='__main__':
 from cache import load_records
 rr=load_records();out=[c for r in rr for c in candidates(r)]
 (RAW/'candidates.json').write_text(json.dumps(out,ensure_ascii=False))
 print(len(out),collections.Counter(c['status'] for c in out))
 print('Labels:',collections.Counter(c['label'] for c in out if c['status']=='candidate').most_common(30))
 print('UNRESOLVED',collections.Counter(c['label'] for c in out if c['status']=='unresolved-language').most_common(50))
 for c in out[:22]:print(c['status'],c['label'],c['form'],':',c['gloss'])
