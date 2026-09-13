import json,re,collections,csv,unicodedata
from pathlib import Path
R=Path(__file__).parent
ROOT=R.parents[4]
GRAM={'mfn.':'noun m f n','pl.':'pl','sg.':'sg','caus.':'caus','refl.':'refl','dem.':'pron dist','quan.':'quantifier','suff.':'suffix','ord.':'ord','pers.':'pron','poet.':'poetic','card.':'num','morph.':'stem','cond.':'conditional','n.':'noun','m.':'noun m','f.':'noun f','mf.':'noun mf','mn.':'noun mn','adj.':'adj','adv.':'adv','vt.':'verb tr','vi.':'verb intr','v.':'verb','v.aux.':'verb','v.adj.':'adj','v.adv.':'adv','pron.':'pron','numr.':'num','postp.':'postp','pref.':'prefix','suf.':'suffix','intrj.':'interj','snt.':'interj','conj.':'conj','part.':'part','p.part.':'pp','imp.':'impv','gen.':'gen','dat.':'dat','acc.':'acc','ins.':'instr','obl.':'obl','subj.':'subj','denm.':'denom','col.':'colloquial','onom.':'onomatopoeia','mim.':'onomatopoeia'}
REFERENCE_ALIASES=json.loads((R/'reference-map.json').read_text())
REG={'Ĺ':'archaic','đ':'formal','Ď':'dialectal','Ě':'poetic','*':'colloquial','?':'uncertain','♠':'uncertain','♢':'uncertain'}
# Subheads use a dedicated Roman type size; explicit extraction marks prevent examples from becoming entries.
HEAD=re.compile(r'(?m)^(?P<reg>[ĹđĎĚ♠♢?* ]*)(?P<native>[\u0c80-\u0cff, /\-\n]+?)\s*(?P<hom>[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*\{R\}(?P<form>[^{}]*?)\{/R\}\s*\[(?P<ipa>[^]]*)\]',re.S)
GR=re.compile(r'(?<![\w])\(?(?:'+ '|'.join(re.escape(x) for x in sorted(GRAM,key=len,reverse=True))+r')\)?')
def tidy(s):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',re.sub(r'(?<=[\w])-\n+(?=[\w])','',s)).strip())
def split_records():
 out=[]
 import gzip
 for l in gzip.open(R/'trace-records.jsonl.gz','rt'):
  r=json.loads(l);t=r['layout'].replace('್್','್')
  if r['key']=='keed2018:p698:c1:e22':t=out[-1]['raw']+'\n'+t
  for old,new in {'\ue895':'ē̃','\ue8ac':'ō̃','\ue84f':'ī̃','\ue859':'ū̃'}.items():t=t.replace(old,new)
  pieces=re.split(r'(\{SM\}.*?\{/SM\})',t,flags=re.S);depth=0;fixed=[]
  for piece in pieces:
   small=piece.startswith('{SM}');value=piece[4:-5] if small else piece
   keep=not small or depth>0 or any(c in '[ú(' for c in value)
   for c in value:
    if c in '[ú(':depth+=1
    elif c in ']û)':depth=max(0,depth-1)
   if keep:fixed.append(value)
  t=''.join(fixed)
  t=re.sub(r'-\{/R\}[^\S\n]*\n\{R\}', '',t)

  # Rejoin adjacent roman-font runs; source word spaces are retained.
  t=re.sub(r'\{/R\}(\s*)\{R\}',lambda m:' ' if '\n' not in m[1] and m[1] else '',t)
  hs=[h for h in HEAD.finditer(t) if not h['form'].startswith('(')]
  if not hs:
   out.append(dict(r,status='alphabet-heading' if not r['roman'] and len(r['raw'])<5 else 'parse-review',units=[]));continue
  units=[]
  for i,h in enumerate(hs):
   locs=list(re.finditer(r'⟦LOC (\d+) (\d+)⟧',t[:h.start()]))
   physical=[int(x) for x in locs[-1].groups()] if locs else [r['page'],r['column']]
   u=h.groupdict();u.update(physical_page=physical[0],physical_column=physical[1],key=r['key']+('' if i==0 else f':sub:{i}'),parent=r['key'] if i else '',body=t[h.end():hs[i+1].start() if i+1<len(hs) else len(t)],raw=t[h.start():hs[i+1].start() if i+1<len(hs) else len(t)])
   for k in ('body','raw'):u[k]=re.sub(r'⟦LOC \d+ \d+⟧','',u[k])
   for k in ('form','native','ipa'):u[k]=tidy(u[k])
   if ' ' not in u['form']:u['ipa']=u['ipa'].replace(' ','')
   units.append(u)
  out.append(dict(key=r['key'],page=r['page'],column=r['column'],ordinal=r['ordinal'],raw=r['raw'],status='parsed',units=units))
 return out

def parse(u):
 b=u['body'].replace('\ue834','R̥').replace('{R}','').replace('{/R}','');tags=[]
 for c in u['reg']:
  if c in REG:tags.append(REG[c])
 preamble='';extra_ipa=[];prefix_citations=[];prefix_alts=[]
 first_grammar=GR.search(b)
 if first_grammar:
  preamble=b[:first_grammar.start()]
  if not re.search(r'[A-Za-z]{6,}', re.sub(r'\([^)]*\)|\[[^]]*\]','',preamble)):
   extra_ipa=re.findall(r'\[([^]]+)\]',preamble)
   prefix_citations=re.findall(r'\(([^)]+)\)',preamble)
   cleaned=re.sub(r'\([^)]*\)|\[[^]]*\]','',preamble)
   cleaned=re.sub(r'[^\u0c80-\u0cff0-9⁰¹²³⁴⁵⁶⁷⁸⁹ ,/\n-]','',cleaned)
   prefix_alts=[tidy(x) for x in cleaned.split(',') if re.search('[\u0c80-\u0cff]',x)]
   b=b[first_grammar.start():]
 refs=[];depth=0;start=None;spans=[]
 for pos,char in enumerate(b):
  if char=='[':
   if depth==0:start=pos
   depth+=1
  elif char==']' and depth:
   depth-=1
   if depth==0:refs.append(b[start+1:pos]);spans.append((start,pos+1))
 for start,end in reversed(spans):b=b[:start]+b[end:]
 refs+=re.findall(r'\{([^{}]*\+[^{}]*)\}',b)
 b=re.sub(r'\{[^{}]*\+[^{}]*\}','',b)
 # Paradigms and usage/semantic labels remain in the per-record evidence, not definitions.
 paradigms=re.findall(r'｟([^｠]*)｠',b);b=re.sub(r'｟[^｠]*｠','',b)
 for paradigm in paradigms:
  if paradigm.strip() in {'pl.','sg.','caus.','refl.'}:tags.extend(GRAM[paradigm.strip()].split())
  if paradigm.strip() in {'ibc.','ifc.'}:tags.extend(['stem','compound'])
  if paradigm.strip() in {'redp.','redup.'}:tags.append('reduplicated')
 b=re.sub(r'ú[^û]*û','',b)
 if re.search(r'\?(?=[①②③④⑤⑥⑦⑧⑨⑩]|\{S:)',b):tags.append('uncertain');b=re.sub(r'\?(?=[①②③④⑤⑥⑦⑧⑨⑩]|\{S:)','',b)
 usages=re.findall(r'þ([^ß]*)ß',b)
 # Native forms immediately after the head pronunciation are printed alternates.
 alt=re.match(r'^\s*([\u0c80-\u0cff0-9⁰¹²³⁴⁵⁶⁷⁸⁹, \n-]+)',b)
 alts=[]
 if alt:
  alts=[tidy(x) for x in alt[1].split(',') if re.search('[\u0c80-\u0cff]',x)]
  b=b[alt.end():]
 alts=prefix_alts+alts
 pieces=re.split(r'(\{S:\d+\}|[①②③④⑤⑥⑦⑧⑨⑩])',b)
 definitions=[];crossrefs=[];citations=prefix_citations
 for part in pieces:
  if re.fullmatch(r'\{S:\d+\}|[①②③④⑤⑥⑦⑧⑨⑩]',part):
   if part.startswith('{S:'):definitions.append(part[3:-1]+'.')
   else:definitions.append(part)
   continue
  # An example's translation never scopes into the lexical definition.
  part=part.split('ű')[0]
  crossrefs+=re.findall(r'[ù=]\s*([\u0c80-\u0cff\s,-]+)\s*\(([^)]+)\)([0-9]?)',part)
  part=re.split(r'[ù=↔⇀♢]|\bcf\.',part)[0]
  for mark in ('Ĺ','đ','Ď','Ě'):
   if mark in part:tags.append(REG[mark]);part=part.replace(mark,'')
  for usage in re.findall(r'þ([^ß]*)ß',part):
   tag={'fig.':'figurative','pej.':'pejorative','dig.':'honorific','col.':'colloquial','poet.':'poetic','rur.':'dialectal','vulg.':'vulgar'}.get(usage.strip())
   if tag:tags.append(tag)
  part=re.sub(r'þ[^ß]*ß','',part)

  # Source bibliographical parentheses, distinguished from ordinary English parentheses.
  def cit(m):
   v=m[1]
   if any(re.search(r'(?<!\w)'+re.escape(label)+r'(?![^\W\d_])',v) for label in REFERENCE_ALIASES):citations.append(v);return ''
   if re.search(r'ĩX-ī',v):citations.append(v);return ''
   if re.search(r'Kitt|KPN|^(?:[A-ZŠČ][A-Za-zŠČâ.]*\.?\s*\d)|\b(?:My\.|NK|SK\.|Hav\.|Nanj\.)',v):citations.append(v);return ''
   return m[0]
  part=re.sub(r'\(([^()]*(?:\([^()]*\)[^()]*)?)\)',cit,part)
  def gram(m):tags.extend(GRAM[m[0].strip('()')].split());return ''
  # Only grammar labels at the beginning of this sense; avoid scientific-name abbreviations.
  while (g:=re.match(r'^[, ]*\s*([ĹđĎĚ]*\s*)('+GR.pattern+r')',part)):
   for ch in g[1]:
    if ch in REG:tags.append(REG[ch])
   token=g[2];tags.extend(GRAM[token.strip('()')].split());part=part[g.end():]
  if (gender:=re.match(r'^\s*,\s*([mfn])(?=\s|$)',part)):
   tags.extend(['noun',gender[1]]);part=part[gender.end():]
  definitions.append(tidy(part))
 gloss=tidy(' '.join(definitions)).strip(' ;')
 gloss=re.sub(r'(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d+\.)\s*$', '',gloss).strip()
 # Bracketed English insertions in example translations are not etymologies.
 ety=[tidy(x) for x in refs if not x.strip().startswith('see Fig') and not re.match(r'^(?:the |his$|her$|of the |in his |to me$|about this)',x.strip())]
 square_citations=[e for e in ety if any(re.match(r'^'+re.escape(label)+r'\s*\d',e) for label in REFERENCE_ALIASES if label not in {'DEDR','M','M.','T','T.'})]
 citations.extend(square_citations);ety=[e for e in ety if e not in square_citations]
 if any(re.match(r'(?:Sk|H|M|Eg|Ar|Pe|Pt|Tk|Pk)\.?(?:\s|$)',e) and '+' not in e for e in ety):tags.append('loanword')
 if any('+' in e for e in ety):tags.append('compound')
 return dict(**u,gloss=gloss,tags=list(dict.fromkeys(tags)),ety=ety,alternates=alts,paradigms=paradigms,usages=usages,aux_citations=citations,crossrefs=crossrefs,extra_ipa=extra_ipa,preamble=preamble)

