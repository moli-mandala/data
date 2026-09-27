"""Reconcile the whole Hahn primer into a proposal without canonical writes."""
import csv,importlib.util,json,re,unicodedata
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent
SOURCE='hahn1900asur'
DIALECT='dialect:Asuri:hahn-1900-asur-dukma:Asur%20Dukma%20%28Hahn%201900%29'
def read(name):return [json.loads(x) for x in (P/name).read_text().splitlines() if x]
def nfc(s):return unicodedata.normalize('NFC',s)
def legacy():
 path=P/('historical-lexical-importer.py' if (P/'historical-lexical-importer.py').exists() else 'import_source.py')
 sp=importlib.util.spec_from_file_location('hahn_legacy_builder',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m.build()
def grammar_tags(gloss,section):
 tags=[];g=gloss.lower()
 if '?' in g or g.startswith(('where','what','who','how','can you','do you','have you','to whom')):tags+=['interr']
 if any(x in g for x in [' not ',' no ','n\'t','nobody','unable']):tags+=['neg']
 if section in ['33-negatives','34-object-agreement','35-compounds','36-causatives','36-completive','37-auxiliary','38-negative-copula']:tags+=['verb']
 if g.startswith(('i ','my ')):tags+=['1sg']
 elif g.startswith(('thou ','thy ','your ')):tags+=['2sg']
 elif g.startswith(('he ','his ')):tags+=['3sg']
 elif g.startswith(('they ','their ')):tags+=['3pl']
 if any(x in g for x in [' was ',' were ',' sent ',' returned',' replied',' came',' died',' seized',' destroyed',' have come',' have beaten']):tags+=['pret']
 if any(x in g for x in [' shall ',' will ',' wilt ']):tags+=['fut']
 if any(x in g for x in [' am ',' is ',' are ',' do not ',' dost not ']):tags+=['pres']
 if g.startswith(('do not ','don\'t ','bring ','give ','go ','come ','take ','look ')):tags+=['impv']
 return tags
# Per-function rows preserve source distinctions rather than flattening morphology lists.
MORPH={
154:[('tē','suffix dat abl','Directional dative or ablative sign'),('ā','suffix gen','Genitive sign'),('rā','suffix gen','Genitive sign'),('ren','suffix poss','Corresponding possessive sign'),('renī','suffix poss','Corresponding possessive sign'),('rē','suffix loc','Locative sign')],
160:[('tanā','suffix pres','Present active and neuter ending'),('ā','suffix pres','Present active and neuter ending'),('yanā','suffix','Indefinite ending'),('tadā','suffix','Indefinite ending'),('ldiā','suffix pret tr','Imperfect of transitive verbs'),('lidiā','suffix pret tr','Imperfect of transitive verbs'),('lā','suffix pret tr','Imperfect of transitive verbs'),('lenā','suffix pret intr','Imperfect of intransitive verbs'),('yanā','suffix pret intr','Imperfect of intransitive verbs')],
161:[*[(x,'suffix perfect','Perfect tense characteristic') for x in ['ā','kedā','ked','ledā','ya','yanā','kan','kanā']],*[(x,'suffix fut','Future ending') for x in ['eā','eyā','yā','nā']],('rē','suffix participle adv','Adverbial participle ending'),('tē','suffix participle pres','Present participle ending after repeated stem'),('kan','suffix participle pret perfect','Past perfect participle ending'),('tē','suffix participle pret perfect','Past perfect participle ending'),('len','suffix participle pret perfect','Past perfect participle ending'),('ked tē','suffix conjunctive-participle','Conjunctive participle ending'),('tē','suffix conjunctive-participle','Conjunctive participle ending'),('ta’ā','suffix inf','Infinitive ending'),('rē','suffix conditional','Conditional ending'),('dō','part conditional','Conditional particle compared by source with Hindi tō'),('tē','suffix conditional','Also used for the conditional'),('oā','suffix pass pres','Present passive ending'),('vā','suffix pass pres','Present passive ending'),('vā','suffix pass fut','Future passive ending'),('goā','suffix pass fut','Future passive ending'),('ae','suffix noun','Noun-of-agency ending added to repeated root')]
}
def build():
 rows,audit=legacy();baseline={r[10] for r in rows};edges=[]
 # Every former edge came from a pipe-separated cell; classify its actual source construction.
 for r in rows:
  if r[11]:
   k=r[10];unit=next(a for a in audit if k.startswith(a['entry_key']+':variant:'))
   section=str(unit['section']);num=section.split('-')[0]
   reason=('Alternative subject expression or tense/negation constructions' if num in {'24','25','26','27','28','29','30','31','32'} else 'Distinct case/possessive constructions within a shared paradigm' if num in {'6','10','11','13'} else 'Parenthesized morphological segmentation, not a second spelling' if num=='19' else 'Co-listed lexical alternatives without an asserted variant relation')
   edges.append(dict(entry_key=k,old_target=r[11],source_unit_key=unit['entry_key'],classification=reason,decision='remove_edge_retain_attestation'));r[11]=''
 # Explicit isolated original corrections confirmed independently; no cross-page harmonization.
 corrections={'hahn1900asur:p171:s49:04': 'pōtā', 'hahn1900asur:p165:s32-cont:04': 'alom rūēmē', 'hahn1900asur:p165:s32-cont:01:variant:1': 'Alōkuiŋ rūēāiŋ', 'hahn1900asur:p149:sintro-totems:01': 'Bes’erā', 'hahn1900asur:p149:sintro-totems:02': 'Īnd', 'hahn1900asur:p149:sintro-totems:03': 'Baṛeā', 'hahn1900asur:p149:sintro-totems:04': 'Hōrō', 'hahn1900asur:p149:sintro-totems:05': 'Būā', 'hahn1900asur:p149:sintro-totems:06': 'Rōtē','hahn1900asur:p165:s35-head:01':'dohóteā','hahn1900asur:p167:s38:02':'Kuniā','hahn1900asur:p167:s38:03':'Kuneā','hahn1900asur:p150:sintro:01':'jhaṛī','hahn1900asur:p159:s16:06:variant:2':'kuniā'}
 adjudication=json.loads((P/'root-kinship-literal-review-20260926.json').read_text())
 assert adjudication['status']=='reviewed_no_change'
 corrections.pop(adjudication['entry_key'],None)
 for r in rows:
  if r[10]==adjudication['entry_key']:assert r[2]==adjudication['current_form']
  if r[10] in corrections:
   r[2]=corrections[r[10]]
  r[14]=' '.join('conditional' if t=='cond' else t for t in r[14].split())
  if ' ' in r[2] and 'multiword-expression' not in r[14]:r[14]+=' multiword-expression'
  if ':comparison:' in r[10] and r[6]==r[2]:r[6]=''
  if ':s39:' in r[10]:r[14]+=' temporal'
  if ':s40:' in r[10] and any(x in r[3] for x in ['here','there','near','far','under','beyond']):r[14]+=' spatial'
  if ':s40-cont:' in r[10]:r[14]+=' manner' if r[3] in ['thus','in this way','somehow, anyhow','well, exactly','quickly'] else ''
  if ':s41:' in r[10] and r[3] in ['no, not','do not']:r[14]+=' neg'
 for u in audit:
  key=u['entry_key'];u['source_unit_key']=key
  u['entry_keys']=[r[10] for r in rows if r[10]==key or r[10].startswith(key+':variant:')]
  if key=='hahn1900asur:p159:s16:06':u['prior_source_form']=u['source_form'];u['source_form']='ci konā|kuniā';u['reading_correction']='Independent original review confirms short i.'
  if key in corrections:u['prior_source_form']=u.get('source_form');u['source_form']=corrections[key];u['reading_correction']='Original-image independent targeted correction; prior inventory retained unchanged.'
  if u['status']=='excluded_nonlexical_context':u['historical_status']=u['status'];u['status']='source_context_census';u['scope_reconciliation']='Former blanket expression exclusions superseded by whole-source supplementary records.'
 def emit(u,form,gloss,tags,key=None,note='',locator=None):
  key=key or u.get('entry_key',u.get('source_unit_key'));page=u['printed_page'];loc=locator or f"p. {page}, section {u.get('section','source context')}, {key.rsplit(':',1)[-1]}"
  tags=list(dict.fromkeys([DIALECT]+['conditional' if t=='cond' else t for t in tags if t!='multiword']))
  if ' ' not in form:tags=[t for t in tags if t!='multiword-expression']
  if ' ' in form and 'multiword-expression' not in tags:tags.append('multiword-expression')
  unc=u.get('uncertainty_types',u.get('typed_uncertainty',[]))
  if unc and 'uncertain' not in tags:tags.append('uncertain')
  rows.append(['Asuri','',nfc(form),nfc(gloss),'','',nfc(note),f'{SOURCE}[{loc}]','','',key,'','','',' '.join(tags)])
  u.setdefault('entry_keys',[]).append(key);return key
 def update_old(key,forms,glosses,tags,note):
  u=next(x for x in audit if x['entry_key']==key);u['historical_status']=u['status'];u['status']='recovered';u['recovered_forms']=forms;u['entry_keys']=[]
  for i,(form,gloss) in enumerate(zip(forms,glosses),1):emit(u,form,gloss,tags,key if len(forms)==1 else key+f':recovered:{i}',note)
 # Recover six previously held units.
 for u in read('held-recovery-first-reading-20260926.jsonl'):
  tags=u['tags'][:]
  if 'completive' in u['entry_key']:tags=['affix','completive']
  if 'deictic' in u['decision']:tags+=['spatial']
  notes={
   'hahn1900asur:p170:comparison:15':'Source labels this a bound plural ending in the comparison table.',
   'hahn1900asur:p170:comparison:24':'The source orders the forms and glosses as here and there; section 40 also glosses hondē as there, thither.',
   'hahn1900asur:p149:sintro-deity:01':'Source names the Creator, apparently identified with the sun.',
   'hahn1900asur:p159:s16-declination:01':'Source gives okoe rā, renī; okoe tī; okoe rē under the declensional heading. The repeated base in the second form is expanded; individual case glosses are not supplied.',
   'hahn1900asur:p163:s27-auxiliary:01':'Source calls this an auxiliary for the past future and supplies a complete example separately.',
   'hahn1900asur:p166:s36-completive:01':'Source calls cabā the completive and illustrates it in jomcabāyanā and rūcabākedākū.'}
  update_old(u['entry_key'],u['forms'],u['glosses'],tags,notes[u['entry_key']])
  next(x for x in audit if x['entry_key']==u['entry_key'])['recovery_review_note']=u['note']
 # Distinct p171 quoted claims, no unsupported exact reuse.
 update_old('hahn1900asur:p171:s49-unglossed:01',['pa’eṉ','paheṉ'],['',''],[], 'Hahn says these forms are shared by Asur and Kurukh; no borrowing or cognacy direction is asserted here.')
 update_old('hahn1900asur:p171:s49-affix:01',['hōṉ'],[''],['affix','emph'],'Hahn calls this an emphatic affix shared by both languages; separate from the lexical even attestation.')
 for raw in read('expression-recovery-p154-161-reviewed.jsonl'):
  u={**raw,'entry_key':raw['source_unit_key'],'entry_keys':[]};audit.append(u)
  if u['status']=='bound_context':
   for i,(form,tags,function) in enumerate(MORPH[u['printed_page']],1):
    section=u.get('section','19–23')
    if u['printed_page']==161:section='19' if i<=12 else '20' if i<=19 else '21' if i<=23 else '22' if i<=27 else '23'
    emit(u,form,'',tags.split(),u['entry_key']+f':function:{i}', 'Source morphology: '+function+'.',locator=f"p. {u['printed_page']}, section {section}, quoted morphology {i}")
   u['status']='recovered_morphology';continue
  if not u['status'].startswith('recovered_'):continue
  for i,form in enumerate(u['forms'],1):
   tags=[t for t in u.get('tags',[]) if t!='multiword']+grammar_tags(u['gloss'],u.get('section',''))
   if u['status']=='recovered_complete_response' and not any(x in u['gloss'] for x in ['my house','our house','tail of','rope of','sword of','king of','men of','in the house','top of','underneath']):tags+=['sentential']
   if u.get('section')=='8':tags+=['degree']
   if u['printed_page']==161 and u.get('section')=='19':tags+=['fut','verb']
   if u['printed_page']==161 and u.get('section')=='21':tags+=['conditional','verb']
   emit(u,form,u['gloss'],tags,u['entry_key'] if len(u['forms'])==1 else u['entry_key']+f':alternative:{i}',u.get('literal_translation','') if isinstance(u.get('literal_translation',''),str) else '')
 for raw in read('expression-recovery-p162-169-first-reading.jsonl'):
  u={**raw,'source_unit_key':raw['entry_key'],'entry_keys':[]};u['status']='independently_reviewed_response';audit.append(u)
  tags=u['tags']+grammar_tags(u['gloss'],u['section'])
  if u['section']=='30-potential-permissive':tags+=['modal']
  if u['section']=='36-completive':tags+=['completive']
  note=u['notes'].split(' Final review crop')[0].strip()
  emit(u,u['source_form'],u['gloss'],tags,note=note)
  for i,form in enumerate(u.get('source_alternates',[]),2):emit(u,form,u['gloss'],tags,u['entry_key']+f':predicate-alternate:{i}',note='Source gives this literal alternative predicate in the shared hāsu (pain) context. Its translation is scoped to that complete construction; unprinted material is not inserted into the form.')
 for raw in read('expression-recovery-song-first-reading.jsonl'):
  u={**raw,'source_unit_key':raw['entry_key'],'entry_keys':[]};u['status']='independently_reviewed_song';audit.append(u);emit(u,u['source_form'],u['gloss'],u['tags']+['poetic'],note='Source labels this a song. Whole stanza and translation retained without inferred word alignment.')
 # Additional source quotations independently inventoried and reviewed.
 extras=[(162,'25','quoted-auxiliary','dohótauā','',['auxiliary','uncertain'],'Source quotes this auxiliary for forming the imperfect. The printed u-shaped glyph differs from the later dohótanā; possible source anomaly, preserved without normalization.'),(167,'37','repeated-auxiliary','dohóta’ā','to remain',['verb'],'Continuing discussion of the auxiliary introduced on p166; identical source form and analysis.'),(153,'3','dual','kiŋ','',['suffix','du'],'Source adds kiŋ to form the dual.'),(153,'3','plural','kū','',['suffix','pl'],'Source adds kū to form the plural.'),(152,'index-24','verb','rūtu’ā','to beat',['verb','inf'],'Quoted conjugation head in the source index.'),(152,'index-38','verb','Konoā','not to be',['verb','neg'],'Quoted verbal head in the source index.'),(151,'introduction','speech','dukmā','',[],'Quoted as the original language/speech label; no independent lexical gloss supplied.'),(150,'introduction','deity','Siŋboŋā','Creator',['proper-noun'],'Source repeats the same named deity in its narrative; identifying context on p149 applies.'),(164,'30','potential','kā','',['suffix','modal'],'Source adds kā to the modified verbal stem for the potential mood.'),(164,'31','imperative-1','mē','',['suffix','impv'],'Source imperative ending; often preceded by e for euphony.'),(164,'31','imperative-2','pē','',['suffix','impv'],'Source imperative ending; often preceded by e for euphony.'),(164,'31','imperative-3','kā','',['suffix','impv'],'Source imperative ending; often preceded by e for euphony.'),(166,'36','causal','gē','',['infix','caus'],'Source inserts gē between the root and its termination to form causal verbs.'),(166,'37','present','tanā','',['suffix','pres'],'Source restricts tanā to an inflectional ending of the present tense, meaning to be.')]
 for page,section,item,form,gloss,tags,note in extras:
  key=f'{SOURCE}:p{page}:whole-recovery:{section}:{item}';u=dict(entry_key=key,source_unit_key=key,printed_page=page,pdf_page=page+12,section=section,forms=[form],gloss=gloss,source_claim=note,status='recovered',entry_keys=[]);audit.append(u)
  if page==162 and item=='quoted-auxiliary':u['typed_uncertainty']=['possible_printed_anomaly'];u['review_evidence']='doh-form-class-review-20260926.json'
  if page==167 and item=='repeated-auxiliary':
   first=next(r for r in rows if r[10]=='hahn1900asur:p166:s37:02');note=first[6]
   u['reuse_basis']='Exact auxiliary repeated in the continuing discussion; translation inherited from explicit introduction on p166.'
  if page==150 and item=='deity':
   first=next(r for r in rows if r[10]=='hahn1900asur:p149:sintro-deity:01');note=first[6]
   u['reuse_basis']='Same named Creator in continuing introductory narrative; no new lexical meaning asserted.'
  emit(u,form,gloss,tags,note=note)
 assert baseline<={r[10] for r in rows};assert len(rows)==len({r[10] for r in rows})
 # Exact same-source reuse requires matching form, sense, analysis and residual information.
 groups={}
 for r in rows:
  sig=tuple(' '.join(sorted(v.split())) if i==14 else v for i,v in enumerate(r) if i not in {7,10})
  groups.setdefault(sig,[]).append(r)
 final=[];aliases={}
 for group in groups.values():
  durable=[r for r in group if r[10] in baseline]
  if len(durable)>1:final+=group;aliases.update({r[10]:r[10] for r in group});continue
  representative=(durable or group)[0].copy();cites=[]
  for r in group:
   aliases[r[10]]=representative[10]
   for cite in r[7].split('; '):
    if cite not in cites:cites.append(cite)
  representative[7]='; '.join(cites);final.append(representative)
 for u in audit:
  pre=u.get('entry_keys',[]);u['pre_reuse_entry_keys']=pre;u['entry_keys']=list(dict.fromkeys(aliases[k] for k in pre));u['exact_reuse']={k:aliases[k] for k in pre if k!=aliases[k]}
 for u in audit:
  if u['status']=='source_context_census':
   u['historical_scope_note']=u.get('note','')
   u['note']='Whole source-page coverage reconciled with linked recovery records; historical exclusions are superseded.'
   u['recovery_unit_keys']=[x['source_unit_key'] for x in audit if int(x['printed_page'])==int(u['printed_page']) and x['entry_keys'] and x['status']!='selected']
 assert baseline<={r[10] for r in final}
 return final,audit,edges

def main():
 rows,audit,edges=build()
 with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 for name,records in [('proposal-audit.jsonl',audit)]: (P/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 (P/'variant-edge-review-20260926.json').write_text(json.dumps({'reviewed':len(edges),'decisions':edges},ensure_ascii=False,indent=2)+'\n')
 profile={c:('#' if c==' ' else '' if c in '.,!?;:' else c.lower().replace('w','v')) for c in set(''.join(r[2] for r in rows))}
 profile['ch']='cʰ';profile['Ch']='cʰ'
 (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(k+'\t'+v+'\n' for k,v in sorted(profile.items())))
 (P/'whole-proposal-counts.json').write_text(json.dumps({'rows':len(rows),'audit_units':len(audit),'statuses':dict(Counter(u['status'] for u in audit)),'exact_reused_keys':sum(len(u['exact_reuse']) for u in audit),'legacy_keys':621},indent=2)+'\n')
 print(len(rows),len(audit))
if __name__=='__main__':main()
