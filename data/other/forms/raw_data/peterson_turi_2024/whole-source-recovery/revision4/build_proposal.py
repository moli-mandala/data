"""Whole-source Turi proposal only. Never writes canonical forms or shared metadata."""
import csv,hashlib,json,re,unicodedata as ud
from pathlib import Path
import yaml
from review_legacy_metadata import apply as apply_legacy_metadata
HERE=Path(__file__).resolve().parent
ROOT_REVIEW=HERE.parent
DATA=HERE.parents[6]
SOURCE='peterson2024turi'
BASE=ROOT_REVIEW.parent.parent.parent/'20260921-peterson-turi.csv'
DIALECTS={'Odisha':'dialect:Turi:turi_odisha:Odisha','Jharkhandi':'dialect:Turi:turi_jharkhandi_peterson2024:Jharkhandi','historical Ranchi':'dialect:Turi:turi_ranchi_lsi1906:Ranchi','historical Jashpur':'dialect:Turi:turi_jashpur_lsi1906:Jashpur','historical Sarangarh':'dialect:Turi:turi_sarangarh_lsi1906:Sarangarh'}
MAP={'1':'first-person','2':'second-person','3':'third-person','incl':'inclusive','excl':'exclusive','inst':'instr','prs':'pres','pst':'pret','perf':'perfect','mid':'middle','imp':'impv','proh':'prohibitive','rdp':'reduplicated','cvb':'conjunctive-participle','seq':'conjunctive-participle','sim':'conjunctive-participle','cop':'copula','anim':'animate','inan':'inanimate','prog':'progressive','itr':'intr','refl':'refl'}
KEEP={'sg','pl','du','gen','loc','voc','obj','subj','poss','ipfv','neg','pres','fut','pret','impv','num','pron','noun','verb','loanword','uncertain','part','interr'}
MAP.update({'past':'pret','present':'pres','future':'fut','singular':'sg','plural':'pl','dual':'du','middle':'middle','perfect':'perfect','imperfective':'ipfv','first':'first-person','second':'second-person','third':'third-person','inclusive':'inclusive','exclusive':'exclusive','genitive':'gen','locative':'loc','instrumental':'instr','objective':'obj','possessive':'poss','prohibitive':'prohibitive','negative':'neg','negation':'neg','imperative':'impv'})
EXPAND={'1sg':'I','2sg':'you','3sg':'third-person singular','1pl':'we','2pl':'you','3pl':'third-person plural','1du':'we (dual)','2du':'you (dual)','3du':'third-person dual','neg':'negation','voc':'vocative','refl':'reflexive','anaph':'anaphoric pronoun','proh':'prohibition','loc.cop':'locative copula','neg.prs.cop':'negative present copula'}
EXPAND['cop']='copula'
read=lambda n:[json.loads(x) for x in (HERE/n).read_text().splitlines()]
def nfc(s):return ud.normalize('NFC',s)
def tags(analysis):
 out=[]
 for token in re.split(r'[^a-zA-Z0-9]+',analysis):
  token=token.lower()
  if token in KEEP:out.append(token)
  elif token in MAP:out.append(MAP[token])
  elif re.fullmatch(r'[123](?:sg|pl|du)',token):
   out.extend([MAP[token[0]],token[1:]])
   if token[1:]!='du':out.append(token)
  elif token=='3anim':out.extend(['third-person','animate'])
  elif token=='ia':out.append('loanword')
 if '(n.)' in analysis:out.append('noun')
 if re.search(r'(?:^|[.=-])fin(?:$|[.=-])',analysis):out.append('verb')
 if any(x in out for x in ['pret','pres','fut','impv','conjunctive-participle','copula','progressive']):out.append('verb')
 return list(dict.fromkeys(out))
def lexical_gloss(analysis):
 # Preserve exact source analysis separately. Only overt printed morphology is removed here.
 bare=re.sub(r'[ⱼₖᵢ]','',analysis)
 bare=re.sub(r'^rdp-','',bare)
 if bare in EXPAND:return EXPAND[bare]
 base=re.split(r'[=-]',bare,1)[0]
 if base in EXPAND:return EXPAND[base]
 base=re.sub(r'\((?:n\.|itr\.|IA)\)','',base)
 person=re.match(r'^([123](?:sg|pl|du))(?:\.|$)',base)
 if person:return EXPAND[person[1]]
 base=re.sub(r'\.(?:imp|intr|itr)$','',base)
 return base.replace('.',' ').strip(' !')
def lect_tags(lect):
 if lect=='Odisha and Jharkhandi':return [DIALECTS['Odisha'],DIALECTS['Jharkhandi']]
 return [DIALECTS[lect]] if lect in DIALECTS else []
def main():
 baseline=list(csv.reader(BASE.open()));assert len(baseline)==246
 legacy_changes=apply_legacy_metadata(baseline)
 audit=[];groups={'grammar':[],'historical':[]}
 def emit(u,key,form,gloss,analysis='',extra=(),note=''):
  lect=u.get('lect','Odisha'); group='historical' if lect.startswith('historical') else 'grammar'
  explicit_inline='structured_tags' in u
  ts=lect_tags(lect)+list(extra)+(u['structured_tags'] if explicit_inline else tags(analysis)+u.get('metadata_tags',[]))
  if not explicit_inline and (u.get('kind')=='morpheme' or u.get('table') in (5,6)):
   ts=[t for t in ts if t!='verb'];ts.append('affix')
  notes=[]
  if analysis:notes.append('Printed analysis: '+analysis+'.')
  if note:notes.append(note)
  if u.get('note'):notes.append(u['note'])
  if u.get('kind')=='epenthetic':notes.append('General source notation retained literally; no locality or phonemic analysis inferred.')
  elif group=='grammar':notes.append('Source p.266 uses plain h for aspiration/breathy voice in examples. Preserve authorial notation; Phonemic is not independently asserted.')
  else:notes.append('Historical Turi witness retranscribed/quoted by Peterson et al.; not new Odisha elicitation. '+lect+'.')
  if lect=='historical unspecified':notes.append('Finer historical dialect not specified in this passage.')
  if any(s in ' '.join(notes).lower() for s in ['uncertain','conflict','inconsistent','tentative','proposed correction','reprint artefact','exclusive independent']):ts.append('uncertain')
  if u.get('scoped_uncertainty'):ts.append('uncertain')
  claim=''
  if u['source_unit_key'].endswith('inline:copula-derivation'):
   claim='The authors tentatively say hɛn- appears to derive from tahɛn stay/remain; no independently established etymological edge.'
  if 'Indo-Aryan' in u.get('note','') and (u.get('example')=='36a' or u['source_unit_key'].endswith(('inline:four','inline:five'))):
   ts.append('loanword');claim='Source explicitly attributes this form to Indo-Aryan, without identifying an individual donor endpoint.'
  if 'loanword' in ts and not claim:claim='Printed IA borrowing annotation retained; no donor form or graph endpoint inferred.'
  loc=f"p. {u['printed_page']}"+(', example '+u['example'] if 'example' in u else ', '+u['source_unit_key'].split(f":p{u['printed_page']}:")[1].replace(':',' '))
  row=['Turi','',form,gloss,'','',' '.join(dict.fromkeys(notes)),f'{SOURCE}[{loc}]','',claim,key,'','','',' '.join(dict.fromkeys(ts))]
  row=[nfc(x) for x in row];assert row[2] and row[3] and len(row)==15
  groups[group].append(row);return key
 # Retain the existing complete Appendix1 source accounting and all 246 keys.
 old=json.loads((ROOT_REVIEW.parent/'20260921-peterson-turi-audit.json').read_text())
 for u in old:audit.append(dict(section='appendix1',source_unit_key=u['entry_key'],status=u['status'],entry_keys=u['emitted_keys'],evidence=u,metadata_revision=[c for c in legacy_changes if c['entry_key'] in u['emitted_keys']]))
 for file in ['grammar-examples-reviewed.jsonl','texts-reviewed.jsonl']:
  for u in read(file):
   key=u['source_unit_key'];keys=[]
   # Full expression always retained. A one-word expression is one attestation, not duplicated.
   multi=len(u['forms'])>1 or ' ' in u['forms'][0]
   whole_tags=['multiword-expression'] if multi else []
   if u.get('kind')=='number_paradigm':whole_tags+=['noun',{'1a':'sg','1b':'du','1c':'pl'}[u['example']]]
   if u.get('example','').startswith('35') or u.get('example','').startswith('36'):whole_tags+=['verb','impv']
   if multi and u.get('kind') not in ('text_title','number_paradigm') and '…' not in u['raw_surface']:whole_tags.append('sentential')
   ana=u['aligned_glosses'][0] if not multi else ''
   omission=' Source prints an ellipsis indicating omitted surrounding material; this is the attested excerpt, not a reconstructed complete utterance.' if '…' in u['raw_surface'] else ''
   whole_u=dict(u,scoped_uncertainty=u.get('example') in ['15b','19b-negative','45','46','54','29'])
   whole=emit(whole_u,key+':whole',' '.join(u['forms']),u['translation'],ana,whole_tags,'Numbered expression.'+omission+' '+('Complete printed alignment: '+ ' | '.join(u['aligned_glosses'])+'.' if multi else ''))
   keys.append(whole); wordmap=[]
   for i,(f,g) in enumerate(zip(u['forms'],u['aligned_glosses']),1):
    if multi:
     word_u=dict(u)
     # Unit commentary belongs to the whole expression unless it describes this word.
     scoped={'29':{2},'19b-negative':set(),'45':{4},'46':{1,2,3},'54':{2}}
     if u['example'] in scoped and i not in scoped[u['example']]:
      word_u['note']='Full expression commentary is retained on '+key+':whole; this alignment is outside the affected span.'
     word_u['scoped_uncertainty']=i in scoped.get(u['example'],set())
     extra=['multiword-expression'] if ' ' in f else []
     bare=re.sub(r'[ⱼₖᵢ]','',g)
     if bare.split('=',1)[0] in ['one','two','three','four','five']:extra.append('num')
     if re.match(r'^[123](?:sg|pl|du)(?:[.=]|$)',bare) or bare in ('refl','anaph'):extra.append('pron')
     if bare in ('what','which'):extra+=['interr','pron']
     if bare.startswith('Prabhat'):extra.append('proper-noun')
     if bare in ('go','sit','sit.down','laugh','rdp-go') or bare.startswith('go='):extra.append('verb')
     if u['example']=='45' and i==4:extra+=['first-person','pl','1pl','inclusive']
     ek=emit(word_u,key+f':word{i}',f,lexical_gloss(g),g,extra)
     keys.append(ek)
    else:ek=whole
    wordmap.append(dict(word=i,form=f,printed_gloss=g,entry_key=ek))
   audit.append(dict(section='numbered',source_unit_key=key,status='emitted',entry_keys=keys,word_alignment=wordmap,evidence=u))
 for u in read('tables-reviewed.jsonl'):
  key=u['source_unit_key']
  if u.get('status') in ('structural_blank','zero_morphology_evidence_only'):
   audit.append(dict(section='tables',source_unit_key=key,status=u['status'],entry_keys=[],evidence=u));continue
  ts=[]
  for t in u.get('tags',[]):ts.extend(tags(t))
  if u['table']==5:
   ts.append('affix')
   tail=key.rsplit(':',1)[1]
   ts+=['obj'] if tail=='3sg-obj' else ['subj'] if tail=='3sg-subj' else ['subj','obj']
  if u['table']==6:ts+=tags(u['gloss']);ts.append('affix')
  ek=emit(u,key,u['form'],u['gloss'],extra=ts)
  audit.append(dict(section='tables',source_unit_key=key,status='emitted',entry_keys=[ek],evidence=u))
 for u in read('inline-reviewed.jsonl'):
  key=u['source_unit_key']
  if u['kind'] not in ('target','morpheme','epenthetic'):
   audit.append(dict(section='inline',source_unit_key=key,status=u['kind'],entry_keys=[],evidence=u));continue
  assert 'structured_tags' in u,u['source_unit_key']
  ts=[]
  if ' ' in u['form']:ts.append('multiword-expression')
  ek=emit(u,key,u['form'],u.get('display_gloss',u['gloss']),u.get('printed_analysis',''),extra=ts)
  audit.append(dict(section='inline',source_unit_key=key,status='emitted',entry_keys=[ek],evidence=u))
 # Only exact complete analyses may share a display row; audit keeps every source unit.
 reuse={}; representatives={}
 for group,rs in groups.items():
  final=[]
  for r in rs:
   sig=tuple(r[:7]+r[8:10]+r[11:])
   if sig in representatives:
    prior=representatives[sig];reuse[r[10]]=prior[10]
    if r[7] not in prior[7].split('; '):prior[7]+='; '+r[7]
   else:representatives[sig]=r;final.append(r)
  groups[group]=final
 for a in audit:
  a['pre_reuse_entry_keys']=list(a['entry_keys']);a['entry_keys']=list(dict.fromkeys(reuse.get(k,k) for k in a['entry_keys']))
  for w in a.get('word_alignment',[]):w['entry_key']=reuse.get(w['entry_key'],w['entry_key'])
 # Only explicitly printed alternatives, with identical scoped analyses, get variant links.
 index={r[10]:r for rs in groups.values() for r in rs}
 pairs=[
  ('p277:example:35a2:whole','p277:example:35a1:whole'),
  ('p277:example:35b2:whole','p277:example:35b1:whole'),
  ('p271:table5:1sg-b','p271:table5:1sg-a'),
  ('p267:inline:sit-2','p267:inline:sit-1'),
  ('p267:inline:sit-3','p267:inline:sit-1'),
  ('p269:inline:genitive-pronoun-2','p269:inline:genitive-pronoun-1'),
 ]
 for category,count in [('past-mid',4),('future-mid',3),('past-perfect-act',2)]:
  pairs.extend((f'p272:table6:{category}-{i}',f'p272:table6:{category}-1') for i in range(2,count+1))
 variant_edges=[]
 for child,parent in pairs:
  child=SOURCE+':'+child;parent=SOURCE+':'+parent
  child=reuse.get(child,child);parent=reuse.get(parent,parent)
  assert child in index and parent in index and child!=parent
  index[child][11]=parent
  variant_edges.append(dict(child=child,parent=parent,basis='Explicit source alternatives in the same numbered example, table cell, or prose list.'))
 output=HERE/'proposal';output.mkdir(exist_ok=True)
 with (output/'20260921-peterson-turi.csv').open('w',newline='') as f:csv.writer(f).writerows(baseline)
 for group,rs in groups.items():
  stem=f'20260927-peterson-turi-{group}'
  with (output/(stem+'.csv')).open('w',newline='') as f:csv.writer(f).writerows(rs)
  profile=f'peterson-turi-{group}'
  settings=dict(file=stem+'.csv',defaults=dict(transcription=[dict(profile=profile,preserve_hyphens=True)],identity=dict(dedupe_by_entry_key=True,legacy_ids='stem'),forms=dict(split_alternates=False),importer=dict(commands=[['data/other/forms/raw_data/peterson_turi_2024/whole-source-recovery/revision4/install_proposal.py','--install']],note='Installs only independently approved pinned whole-source artifacts. Does not rebuild CLDF or databases.')),sources={})
  (output/(stem+'.yaml')).write_text(yaml.safe_dump(settings,sort_keys=False,allow_unicode=True))
  literals=''.join(r[2] for r in rs)
  chars=sorted(set(literals+ud.normalize('NFD',literals)))
  mapping={' ':'#'}
  if group=='grammar':mapping.update({'ɑ':'a','ʋ':'v','ʈ':'ṭ','ɖ':'ḍ','ɳ':'ṇ','ɽ':'ṛ','ɲ':'ñ','ɟ':'j','ʧ':'c','ʤ':'j','ʃ':'ś','j':'y'})
  lines=['Grapheme\tIPA']+[c+'\t'+mapping.get(c,c) for c in chars]
  if group=='grammar':lines+=['tʃ\tc','dʒ\tj']
  (output/(profile+'.txt')).write_text('\n'.join(lines)+'\n')
 (output/'whole-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
 (output/'variant-decisions.json').write_text(json.dumps(variant_edges,ensure_ascii=False,indent=2)+'\n')
 keys=[r[10] for rs in [baseline,*groups.values()] for r in rs];assert len(keys)==len(set(keys))
 summary=dict(status='owner_proposal_not_independently_approved',canonical_unchanged_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),baseline_rows=len(baseline),new_rows={k:len(v) for k,v in groups.items()},total_rows=len(keys),audit_units=len(audit),exact_analysis_reuses=len(reuse),new_dialect_proposal=DIALECTS['Jharkhandi'],raw_unit_counts={f:len(read(f)) for f in ['grammar-examples-reviewed.jsonl','texts-reviewed.jsonl','tables-reviewed.jsonl','inline-reviewed.jsonl']},pending=['Independent original/output audit','Canonical installation and shared dialect/Bib registration after approval','Full build and browser explicitly deferred'])
 (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
