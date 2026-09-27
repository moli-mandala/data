"""Stage complete Kaikadi source input, preserving reviewed evidence and legacy keys."""
from pathlib import Path
from urllib.parse import quote
import csv,hashlib,json,unicodedata
from specimen_grammar import classify as specimen_tags
from table_grammar import classify as table_tags
P=Path(__file__).resolve().parent
SOURCE='grierson1906kaikadi'
def read(n):return [json.loads(x) for x in (P/n).read_text().splitlines() if x]
def nfc(s):return unicodedata.normalize('NFC',s)
def normalize_tags(tags):
 tags=list(tags)
 if tags and tags[0]=='n':tags[0]='noun'
 tags=['verb' if x=='v' else x for x in tags]
 for p in ['1','2','3']:
  if p in tags:
   number=next((x for x in ['sg','pl'] if x in tags),None)
   if number:tags=[x for x in tags if x not in [p,number]]+[p+number]
   else:tags=[{'1':'first-person','2':'second-person','3':'third-person'}.get(x,x) for x in tags]
 return tags

def lect(d):
 if not d:return ''
 if d=='Sholapur':return 'dialect:Kaikadi:kaikadi_sholapur_lsi1906:Sholapur'
 if d=='Buldana (Melkapur Taluka)':d='Buldana'
 return f'dialect:Kaikadi:lsi1906-kaikadi:{quote(d,safe="")}'

def build():
 grammar=read('grammar-reviewed.jsonl');spec=read('specimens-final-reviewed.jsonl');table=read('table-reviewed.jsonl');outside=read('outsidechapter-reviewed.jsonl')
 assert [len(grammar),len(spec),len(table),len(outside)]==[124,706,241,8]
 addenda=read('addenda-reviewed.jsonl');by_prompt={a['target_prompt']:a for a in addenda}
 legacy=set(json.load((P/'legacy164-preservation-manifest.json').open())['entry_keys'])
 rows=[];audit=[]
 for u in table+grammar+outside+spec+addenda:
  a={**u,'entry_keys':[]};tags=[];notes=[];gloss=u['gloss'];forms=u['forms']
  if u['scope']!='target':
   a['disposition']='audit-only-'+u['scope'];audit.append(a);continue
  citation_source=u.get('source_citation_key',SOURCE)
  if u['section']=='later-addendum':
   tags=table_tags(u['target_prompt']);base=u['source_unit_key'];keys=[base]
   loc=f'p. 18, Kaikadi correction to original p. {u["target_original_page"]}, item {u["target_prompt"]}'
  elif u['section'].startswith('standard'):
   tags=table_tags(u['prompt']);gloss=gloss.replace(' (past tense)','');base=f'grierson1906lsi4:kaikadi_sholapur:{u["prompt"]}'
   loc=f'p. {u["printed_page"]}, standard list item {u["prompt"]}'
   keys=[base if i==1 and base in legacy else f'{base}:{i}' if len(forms)>1 else base for i in range(1,len(forms)+1)]
   if u['prompt'] in by_prompt:
    correction=by_prompt[u['prompt']]
    notes.append('Original1906 edition reading. Later bound-in Addenda Minora IV p18: '+correction['printed_correction_description']+' The later witness is accounted separately; its publication date is unverified.')
    a['later_correction_evidence_key']=correction['source_unit_key']
   if u['prompt'] in [133,136]:notes.append('Source prompt explicitly comparative degree.')
   if u['prompt'] in [134,137]:notes.append('Source prompt explicitly superlative degree.')
  elif u['section'].startswith('specimen'):
   tags,obs=specimen_tags(u);a['grammatical_observations']=obs
   base=u['source_unit_key'];keys=[base];loc=f'p. {u["printed_page"]}, specimen {u["section"][8:]}, line {u["line"]}, word {u["word"]}'
   notes.append('Aligned interlinear occurrence; gloss is local to this attestation.')
  else:
   tags=normalize_tags(u.get('tags',[]));base=u['source_unit_key'];keys=[base if i==1 else f'{base}:alternate:{i}' for i in range(1,len(forms)+1)]
   loc=f'p. {u["printed_page"]}, {u["section"]}, {base.rsplit(":",1)[-1]}'
   if base.endswith('relative'):tags=['pron','relative']
   label=base.rsplit(':',1)[-1]
   if label in ['who','what','who-neuter','whose']:tags+=['interr']
   if label=='who-neuter':tags+=['n']
   if label=='whose':tags+=['gen']
   if label=='good-woman':
    tags=['noun','sg']
    notes.append('The source tentatively says the neuter singular seems to be used as feminine here; the literal English gloss is retained without asserting feminine morphology.')
   if ':copula-' in base or ':present-' in base and base.rsplit(':',1)[-1] in ['present-am','present-is','present-are','present-berar-am']:tags+=['copula']
   if base.rsplit(':',1)[-1] in ['women','woman']:tags+=['loanword']
   if 'inclusive-we' in base:tags+=['inclusive']
  if u.get('note'):notes.append(u['note'])
  if u.get('uncertainty'):tags.append('uncertain')
  dt=lect(u.get('source_district',''))
  if dt:tags.append(dt)
  if u.get('source_district')=='Buldana (Melkapur Taluka)':notes.append('Source specimen explicitly from Melkapur Taluka, Buldana district.')
  a['exported_grammatical_tags']=list(dict.fromkeys(tags))
  a['exported_notes']=nfc(' '.join(notes))
  for i,(form,key) in enumerate(zip(forms,keys)):
   rt=tags+(['multiword-expression'] if ' ' in form else [])
   alt=''
   if i:
    rt+=['alternate'];alt=keys[0]
   rows.append(['Kaikadi','',nfc(form),nfc(gloss),'','',nfc(' '.join(notes)),f'{citation_source}[{loc}]','','',key,alt,'','',' '.join(dict.fromkeys(rt))]);a['entry_keys'].append(key)
  a['disposition']='emitted-target';audit.append(a)
 # Preserve every legacy identity, even if it has the same complete analysis as
 # another historical table key. New repeated attestations may reuse a key.
 reps={};aliases={};out=[]
 for r in rows:
  fp=tuple(x for i,x in enumerate(r) if i not in [7,10])
  if fp not in reps or r[10] in legacy:
   out.append(r);reps.setdefault(fp,r);aliases[r[10]]=r[10]
  else:
   rep=reps[fp];rep[7]+='; '+r[7];aliases[r[10]]=rep[10]
 for r in out:
  if r[11]:r[11]=aliases.get(r[11],r[11])
 for a in audit:
  a['pre_reuse_entry_keys']=a['entry_keys'][:];a['entry_keys']=[aliases.get(k,k) for k in a['entry_keys']]
  if a['entry_keys']!=a['pre_reuse_entry_keys']:a['exact_attestation_reuse']=True
 assert legacy<=set(r[10] for r in out)
 assert len({r[10] for r in out})==len(out)
 with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(out)
 (P/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit))
 clusters=set()
 for r in out:
  cur=''
  for ch in r[2]:
   if unicodedata.combining(ch) and cur:cur+=ch
   else:
    if cur:clusters.add(cur)
    cur=ch
  if cur:clusters.add(cur)
 # Literal preservation: source underbars/diaereses are retained. No guessed
 # affricate realization or vowel harmony is introduced.
 (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(c+'\t'+('NULL' if c in '?.,!' else '#' if c==' ' else c.lower().replace('ṅ','ŋ').replace('w','v'))+'\n' for c in sorted(clusters)))
 report={'status':'complete whole-source proposal; independent output audit pending','source_units':len(audit),'table_cells':241,'specimen_atoms':706,'grammar_groups':124,'outsidechapter_units':8,'source_blanks':17,'audit_only_controls_or_morphology':30,'later_addendum_units':len(addenda),'later_addendum_audit_only':4,'candidate_forms':len(rows),'proposal_rows':len(out),'exact_reuse':len(rows)-len(out),'legacy_keys_preserved':len(legacy),'hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['proposal.csv','proposal-audit.jsonl','proposal-profile.txt']}}
 (P/'proposal-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':build()
