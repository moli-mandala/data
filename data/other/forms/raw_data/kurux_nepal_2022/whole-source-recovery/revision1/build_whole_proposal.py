"""Whole-source review proposal. Never writes canonical data or identity state."""
from __future__ import annotations
import csv,hashlib,json,re,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[5]
SOURCE='shackelford-swenson-chaudhary-maggard2022kurux'
OUT=HERE/'proposal'
SITE_NAMES={'kurux_lochani':'Lochani','kurux_bhokraha':'Bhokraha','kurux_siddhapur':'Siddhapur','kurux_tokla':'Tokla','kurux_pathibari':'Pathibari','kurux_ranchi':'Ranchi','kurux_lohardaga':'Lohardaga','kurux_jharkhand':'Jharkhand'}
SOURCE_NOTE='Phonetic field transcription as printed; source warns of speaker/transcriber variation. Source conventions omit some onset/dental distinctions and use h broadly. Raised vowels remain raised; no narrower phonology inferred.'
def jl(name):return [json.loads(s) for s in (HERE/name).read_text().splitlines()]
def nfc(s):return unicodedata.normalize('NFC',s)
def uniq(xs):return list(dict.fromkeys(xs))
def aligned_metadata(gloss,withheld=False):
 if withheld:return gloss,[],''
 explicit={'EMPH':('emphatic marker',['emph']),'POSS':('possessive marker',['poss']),'POSTP':('postposition',['postp']),'PREP':('preposition',['prep']),'beat-PT':('beat',['verb','pret']),'killed-2P':('killed',['verb','second-person','uncertain'])}
 if gloss in explicit:
  clean,tags=explicit[gloss]
  return clean,tags,'Printed aligned analysis: '+gloss+'.'+(' Source 2P label retained without guessing singular/plural.' if gloss=='killed-2P' else '')
 pronouns={'I':['first-person','sg','1sg'],'me':['first-person','sg','1sg','obj'],'my':['first-person','sg','1sg','poss'],'mine':['first-person','sg','1sg','poss'],'we':['first-person','pl','1pl'],'our':['first-person','pl','1pl','poss'],'he':['third-person','sg','3sg'],'she':['third-person','sg','3sg'],'him':['third-person','sg','3sg','obj'],'his':['third-person','sg','3sg','poss'],'her':['third-person','sg','3sg'],'it':['third-person','sg','3sg'],'they':['third-person','pl','3pl'],'their':['third-person','pl','3pl','poss']}
 if gloss in pronouns:return gloss,['pron','personal']+pronouns[gloss],'Explicit aligned personal-pronoun gloss; no gender morphology or ambiguous her case inferred.'
 if gloss in {'one','two'}:return gloss,['num'],'Explicit aligned numeral gloss.'
 if gloss=='who':return gloss,['pron','interr'],'Explicit aligned interrogative pronoun.'
 if gloss in {'not','do not'}:return gloss,['neg'],'Explicit aligned negation gloss; no unprinted imperative/person analysis inferred.'
 return gloss,[],''
def prompt_metadata(n,gloss):
 tags=[];notes=[];clean=gloss
 if 204<=n<=217:tags+=['num']
 if '(3S-PT)' in gloss.replace('3S- PT','3S-PT'):
  tags+=['verb','pret','third-person','sg','3sg'];notes+=['Source prompt: third-person singular simple past.'];clean=re.sub(r'\s*\(3S-\s*PT\)','',clean)
 if gloss.startswith('don’t '):tags+=['verb','neg','impv','second-person','sg','2sg'];notes+=['Source p41 explicitly defines negative elicitation as second-person singular imperative.']
 if n in {268,269}:tags+=['verb']
 if 234<=n<=243:tags+=['interr']
 if n in {234,235,236}:tags+=['pron']
 if n==235:tags+=['poss']
 if 244<=n<=247:tags+=['demonstrative']
 if n in {246,247}:tags+=['pl']
 personal={325:['first-person','sg','1sg','poss'],327:['second-person','sg','2sg','poss'],328:['second-person','pl','2pl','poss'],329:['third-person','sg','3sg','poss'],330:['third-person','pl','3pl','poss'],331:['first-person','sg','1sg'],332:['second-person'],333:['third-person','sg','3sg'],334:['third-person','sg','3sg'],337:['second-person','pl','2pl'],338:['third-person','pl','3pl']}
 if n in personal:tags+=['pron','personal']+personal[n]
 if n==326:tags+=['pron','personal','first-person','pl','1pl','poss']
 if n in {335,336}:tags+=['pron','personal','first-person','pl','1pl','inclusive' if n==335 else 'exclusive']
 if '(my)' in gloss:tags+=['poss','first-person','sg','1sg']
 if n in {327,328,329,330,332,333,334,337,338}:tags+=['informal'];notes+=['Source labels this prompt informal; no honorific system inferred.']
 clean=re.sub(r'\s*\(informal(?:/\s*(?:sing|plural)| plural)?\)','',clean)
 clean=clean.replace(' (verb)','')
 return clean,uniq(tags),' '.join(notes)
def main():
 OUT.mkdir(exist_ok=True)
 old=list(csv.reader((DATA/'data/other/forms/20260813-kurux-nepal.csv').open()))
 old_by_key={r[10]:r for r in old};assert len(old_by_key)==1336
 outputs={k:[] for k in ['wordlist','grammar','native','roman']};audit=[];changes=[];allkeys=set();reuses={}
 def emit(bucket,key,form,gloss,page,locator,site='',native='',phonemic='',notes='',tags=(),variant=''):
  form=nfc(form);native=nfc(native);phonemic=nfc(phonemic)
  ts=([f'dialect:Kurux:{site}:{SITE_NAMES[site]}'] if site else [])+list(tags)
  if ' ' in form:ts+=['multiword-expression']
  row=['Kurux','',form,gloss,native,phonemic,notes,f'{SOURCE}[p. {page}, {locator}]','','',key,variant,'','', ' '.join(uniq(ts))]
  assert key not in allkeys;allkeys.add(key)
  # Preserve every installed key. New exact attestations may reuse a prior new
  # row only when all lexical, grammatical and uncertainty fields agree.
  identity=(bucket,tuple(row[:7]),row[9],tuple(row[11:]))
  if key not in old_by_key and identity in reuses:
   prior=reuses[identity];prior[7]+=';'+row[7];return prior[10]
  outputs[bucket].append(row)
  if key not in old_by_key:reuses[identity]=row
  return key
 for r in jl('wordlist-reviewed.jsonl'):
  keys=[];n=r['prompt'];gloss,tags,note=prompt_metadata(n,r['gloss'])
  forms=[] if not r['form'].strip() or re.fullmatch(r'[-–—\s]+',r['form']) else [x.strip() for x in r['form'].split(',') if x.strip()]
  for i,form in enumerate(forms,1):
   key=f"{r['source_unit_key']}:{i}"
   k=emit('wordlist',key,form,gloss,r['printed_page'],f'wordlist {n}',r['lect'],phonemic=form,notes=' '.join([SOURCE_NOTE,note]).strip(),tags=tags,variant=f"{r['source_unit_key']}:1" if i>1 else '')
   keys.append(k)
  audit.append(dict(r,entry_keys=keys,disposition='emitted' if keys else 'source-blank',metadata_basis=note))
 for r in jl('morphology-reviewed.jsonl'):
  keys=[];site='kurux_'+(r.get('site') or '').lower();site=site if site in SITE_NAMES else ''
  note='Source Table B.2 is a dataset-specific analysis; the author explicitly disclaims detailed morphology. '+r.get('note','')
  if r['kind']=='zero-prefix':pass
  elif r['kind']=='suffix-group':
   for i,form in enumerate(r['alternatives'],1):keys.append(emit('grammar',r['key']+f':v{i}',form,'third-person singular simple-past suffix',r['printed_page'],'Table B.2',site,phonemic=form,notes=note,tags=['suffix','verb','pret','third-person','sg','3sg']))
  elif r['kind']=='prefix-suffix-cooccurrence-group':
   prefixes=r['raw_prefix'].replace('\n','').split('/');suffixes=[x.strip() for x in re.split(r'[/;\n]',r['raw_suffix']) if x.strip()]
   context=' Printed cooccurrence group: '+r['raw_prefix'].replace('\n',' ')+' with '+r['raw_suffix'].replace('\n',' ') +'. Alternative-specific pairing is not asserted.'
   for category,forms in [('prefix',prefixes),('suffix',suffixes)]:
    for i,form in enumerate(forms,1):keys.append(emit('grammar',r['key']+f':{category}:{i}',form,f'second-person singular negative-imperative {category}',r['printed_page'],'Table B.2',site,phonemic=form,notes=note+context,tags=[category,'verb','neg','impv','second-person','sg','2sg']+(['uncertain'] if r.get('uncertainty') else [])))
  elif 'inline-suffix' in r['key']:
   form=r['raw'];gloss='at' if form=='-n̤u' else ''
   keys.append(emit('grammar',r['key'],form,gloss,r['printed_page'],'inline removed suffix',phonemic=form,notes='Source explicitly says this suffix was removed for lexical comparison. '+('No semantic gloss supplied.' if not gloss else ''),tags=['suffix']+(['uncertain'] if not gloss else [])))
  audit.append(dict(r,entry_keys=keys,disposition='emitted' if keys else 'documentary-morphology'))
 narrative=jl('narrative-first-reading.jsonl');review=json.loads((HERE/'narrative-owner-original-review-20260927.json').read_text())
 notes={x['key']:x['note'] for x in review['source_qualifications'] if 'key' in x}
 withheld_analysis={'kurux-nepal:p62:narrative:1:word1','kurux-nepal:p62:narrative:4:word4','kurux-nepal:p63:narrative:12:word1','kurux-nepal:p65:narrative:25:word4'}
 mergekeys=[f'kurux-nepal:p67:narrative:35:word{i}' for i in (2,3,4)]
 mergedkey=mergekeys[0]+':combined';mergeoutput=None
 for r in narrative:
  key=r['source_unit_key'];tags=[];note='Lohardaga, Jharkhand narrative, recorded 5 January 2018. No unprinted phonemic form or lemma inferred.'
  gloss=r['gloss']
  if not key.endswith(':whole'):
   gloss,more,basis=aligned_metadata(gloss,key in withheld_analysis);tags+=more
   if basis:note+=' '+basis
  if key in notes:note+=' '+notes[key];tags+=['uncertain']
  if key.endswith(':whole'):
   tags+=['sentential'];note+=' Complete source segment and its free English translation.'
   if r['segment'] in {1,4,10,12,25,28,35}:note+=' The aligned source evidence contains a separately documented literal or gloss uncertainty.';tags+=['uncertain']
  if key in mergekeys:
   if mergeoutput is None:mergeoutput=emit('native',mergedkey,'आ उल्ला तिम','since that day',67,'narrative 35 aligned groups 2–4','kurux_lohardaga',native='आ उल्ला तिम',notes=note+' Three adjacent physical glosses are preserved jointly because the printed individual alignment is inconsistent; no reordering or POS inference.',tags=['uncertain'])
   keys=[mergeoutput]
  else:keys=[emit('native',key,r['form'],gloss,r['printed_page'],f"narrative {r['segment']} "+('whole' if key.endswith(':whole') else f"aligned group {r['word']}"),'kurux_lohardaga',native=r['native'],notes=note,tags=tags)]
  audit.append(dict(r,entry_keys=keys,status='owner original-reviewed; final output audit pending'))
 orth=json.loads((HERE/'orthography-p75-reviewed.json').read_text());form=' '.join(orth['native_lines'])
 key=emit('native',orth['source_unit_key'],form,orth['english_translation'],75,'quoted orthography passage','kurux_jharkhand',native=form,notes=orth['source_attribution']+' First-line ढ/ड glyph uncertainty retained; preferred हयकढुरा, alternative हयकडुरा. No word-level translation alignment is supplied.',tags=['sentential','uncertain'])
 audit.append(dict(orth,entry_keys=[key]))
 for i,line in enumerate(orth['native_lines'],1):audit.append(dict(source_unit_key=f'kurux-nepal:p75:orthography:line{i}',form=line,gloss='',entry_keys=[key],disposition='physical-line-evidence',note='Part of the complete paragraph; no separate English word/line gloss invented.'))
 for r in jl('prose-reviewed.jsonl'):
  keys=[]
  if r['disposition']=='emit':
   for i,form in enumerate(r['forms'],1):
    route=r['route'];bucket={'phonetic':'grammar','native':'native','literal':'roman'}[route]
    keys.append(emit(bucket,r['source_unit_key']+f':v{i}',form,r['gloss'],r['printed_page'],'prose attestation',native=form if route=='native' else r.get('native',''),phonemic=form if route=='phonetic' else '',notes=r['note'],tags=['proper-noun'] if 'name' in r['gloss'] else ['noun','uncertain']))
  audit.append(dict(r,entry_keys=keys))
 current={r[10]:r for rs in outputs.values() for r in rs}
 assert set(old_by_key)<=set(current),'A legacy key was lost'
 for k,before in old_by_key.items():
  after=current[k];diff={str(i):{'before':(before+['']*15)[i],'after':after[i]} for i in range(15) if (before+['']*15)[i]!=after[i]}
  if diff:changes.append(dict(entry_key=k,fields=diff))
 for bucket,rs in outputs.items():
  with (OUT/f'{bucket}.csv').open('w',newline='') as f:csv.writer(f).writerows(rs)
 (OUT/'audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
 (OUT/'legacy-before-after.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
 summary=dict(status='unfrozen integration draft; prose/page census, metadata/profile and independent output gates pending',rows={k:len(v) for k,v in outputs.items()},audit_units=len(audit),legacy_keys_preserved=len(old_by_key),legacy_rows_changed=len(changes),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='summary.json'})
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
