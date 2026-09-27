"""Prepare the entire reviewed Korava/Yerukala source; never install canonical files."""
import csv,hashlib,json,re,unicodedata
from pathlib import Path
from collections import Counter
from table_grammar import table_tags
from specimen_grammar import classify
P=Path(__file__).resolve().parent
SOURCE='grierson1906korava'
ERRATA='grierson_addenda_minora_iv_boundin'
LECTS={
 'specimen1':'dialect:Yerukula:korchari_belgaum_lsi1906:Belgaum',
 'specimen2':'dialect:Yerukula:korchari_belgaum_lsi1906:Belgaum',
 'specimen3':'dialect:Yerukula:korvi_belgaum_lsi1906:Belgaum',
 'specimen4':'dialect:Yerukula:korvi_jamkhandi_lsi1906:Jamkhandi',
 'specimen5':'dialect:Yerukula:korvaru_bijapur_lsi1906:Bijapur'}
def read(name):return [json.loads(x) for x in (P/name).read_text().splitlines() if x]
def nfc(x):return unicodedata.normalize('NFC',x)
def generate():
 rows=[];audit=[]
 def emit(unit,form,gloss,tags,note,key,locator,variant='',citations=None):
  tags=list(tags)
  if gloss.endswith(' (past tense)'):
   assert 'pret' in tags
   gloss=gloss.removesuffix(' (past tense)')
  if ' ' in form:tags+=['multiword-expression']
  rows.append(['Yerukula','',nfc(form),nfc(gloss),'','',nfc(note),'; '.join([f'{SOURCE}[{locator}]']+(citations or [])),'','',key,variant,'','',' '.join(dict.fromkeys(tags))]);return key
 for unit in read('grammar-reviewed.jsonl')+read('source-wide-attestations.jsonl'):
  key=unit['source_unit_key'];keys=[];target='control' not in unit['source_scope'] and unit['kind']=='lexical'
  locator=f"p. {unit['printed_page']}, {unit['section']}, {key.rsplit(':',1)[1]}"
  if target:
   note='Source gives this shared Korava/Yerukala example without assigning a single locality. '+unit.get('note','')
   for i,form in enumerate(unit['forms']):
    child=key if i==0 else key+f':alternate:{i+1}'
    tags=unit.get('alternate_tags',[unit['tags']]*len(unit['forms']))[i]
    variant=key if i and not unit.get('distinct_analyses') else ''
    keys.append(emit(unit,form,unit['gloss'],tags,note,child,locator,variant))
  audit.append({**unit,'entry_keys':keys,'status':'ingested' if target else 'excluded_control' if 'control' in unit['source_scope'] else 'bound_morphology_evidence','citation_locator':locator})
 for unit in read('specimens-reviewed.jsonl'):
  key=unit['source_unit_key'];locator=f"p. {unit['printed_page']}, {unit['section']}, editorial line {unit['line']}, aligned unit {unit['word']}"
  tags,observations=classify(unit)
  note=' '.join(x for x in [f"Source specimen: {unit['source_lect']}, {unit['source_locality']}.",unit.get('note',''),unit.get('source_commentary',''),*observations] if x)
  keys=[emit(unit,unit['forms'][0],unit['gloss'],tags+[LECTS[unit['section']]],note,key,locator)]
  audit.append({**unit,'entry_keys':keys,'status':'ingested','citation_locator':locator,'grammatical_observations':observations})
 table=read('table-reviewed.jsonl');assert [r['prompt'] for r in table]==list(range(1,242))
 for unit in table:
  n=unit['prompt'];key=unit['source_unit_key'];locator=f"p. {unit['printed_page']}, item {n}";keys=[];tags=table_tags(n)+[LECTS['specimen3']];notes=unit.get('note','');citations=[]
  gloss=unit['gloss']
  if n in {164,181,186,187}:notes+=' Later bound-in Addenda Minora explicitly corrects this printed reading; the original and correction are retained as separate attestations.'
  if n in {197,207,213}:notes+=' Later bound-in Addenda Minora confirms nasal ãva; this original scan already shows that mark.';citations.append(f'{ERRATA}[p. 18, Korvi item {n}]')
  if n==207:notes+=' Original English prompt prints He goest; later addendum corrects it to He goes.';gloss='He goes'
  if n==211:notes+=' Original prompt number is misprinted 11; later addendum corrects its number to 211.';citations.append(f'{ERRATA}[p. 18, English item 211]')
  for i,form in enumerate(unit['forms']):
   child=key if len(unit['forms'])==1 else key+f':{i+1}'
   gloss_i=unit.get('alternate_senses',[gloss]*len(unit['forms']))[i]
   # Multiple table answers alone do not establish a variant relationship.
   variant=keys[0] if i and n==29 else ''
   keys.append(emit(unit,form,gloss_i,tags,notes.strip(),child,locator,variant,citations))
  audit.append({**unit,'entry_keys':keys,'status':'ingested','citation_locator':locator})
  if n in {164,181,186,187}:
   corrected=unit['forms'][0].replace('Ava','Ãva') if n!=186 else 'Nī aḍasā'
   correction_key=key+':erratum'
   note='Explicit later bound-in Addenda Minora correction. The original 1906 reading remains separately represented; no date is asserted for the bound-in witness.'
   emit(unit,corrected,gloss,tags,note,correction_key,locator,citations=[f'{ERRATA}[p. 18, Korvi item {n}]'])
   audit.append(dict(source_unit_key=correction_key,original_source_unit_key=key,forms=[corrected],gloss=gloss,printed_page=18,physical_leaf=708,section='later-erratum',entry_keys=[correction_key],status='ingested',original_forms=unit['forms'],source_key=ERRATA))
 legacy={r[10] for r in csv.reader((P/'historical-column.csv').open())}
 assert legacy<={r[10] for r in rows},legacy-{r[10] for r in rows}
 assert len({r[10] for r in rows})==len(rows)
 # Expand only explicitly parenthesized source material; raw source forms stay in audit.
 for row in list(rows):
  if '(' not in row[2]:continue
  assert re.fullmatch(r'[^()]*\([^()]+\)[^()]*',row[2]),row[2]
  literal=row[2];short=re.sub(r'\([^()]+\)','',literal);long=literal.replace('(','').replace(')','')
  row[2]=short
  explanation='Source explicitly marks optional material in '+literal+'; both printed alternatives are represented.'
  row[6]=' '.join(x for x in [row[6],explanation] if x)
  alternate=row.copy();alternate[2]=long;alternate[10]=row[10]+':optional:2';alternate[11]=row[10];rows.append(alternate)
  for unit in audit:
   if row[10] in unit['entry_keys']:
    unit['entry_keys'].append(alternate[10]);unit['optional_expansion']={'literal':literal,'forms':[short,long]}
 # Reuse only exact language/form/sense/analysis/notes equality, preserving all citations.
 groups={}
 for row in rows:
  fingerprint=tuple(' '.join(sorted(v.split())) if i==14 else v for i,v in enumerate(row) if i not in {7,10})
  groups.setdefault(fingerprint,[]).append(row)
 final=[];aliases={}
 for group in groups.values():
  durable=[r for r in group if r[10] in legacy]
  if len(durable)>1:
   final.extend(group);aliases.update({r[10]:r[10] for r in group});continue
  representative=(durable or group)[0].copy();cites=[]
  for row in group:
   aliases[row[10]]=representative[10]
   for citation in row[7].split('; '):
    if citation not in cites:cites.append(citation)
  representative[7]='; '.join(cites);final.append(representative)
 for row in final:
  if row[11]:row[11]=aliases[row[11]]
 for unit in audit:
  before=unit['entry_keys'];unit['pre_reuse_entry_keys']=before[:]
  unit['entry_keys']=list(dict.fromkeys(aliases[k] for k in before))
  changes={k:aliases[k] for k in before if aliases[k]!=k}
  if changes:unit['exact_attestation_reuse']=changes
 assert legacy<={r[10] for r in final}
 return final,audit

def main():
 rows,audit=generate()
 with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (P/'proposal-audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit))
 clusters=set()
 for row in rows:
  for c in row[2]:
   if unicodedata.combining(c):continue
   clusters.add(c)
 # Include complete base+combining clusters, including nasalized long vowels.
 clusters=set(re.findall(r'[^\u0300-\u036f][\u0300-\u036f]*',''.join(r[2] for r in rows)))
 profile={c:('#' if c==' ' else '' if c in '.?' else c.lower().replace('ṅ','ŋ').replace('w','v')) for c in clusters}
 profile['ṯs̱']='ʦ'
 for source,target in {'chh':'cʰ','ch':'c','kh':'kʰ','gh':'gʰ','jh':'jʰ','ṭh':'ṭʰ','ḍh':'ḍʰ','th':'tʰ','dh':'dʰ','ph':'pʰ','bh':'bʰ'}.items():
  profile[source]=target;profile[source.capitalize()]=target
 for unused in ['ṯ','s̱']:profile.pop(unused,None)
 (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(k+'\t'+v+'\n' for k,v in sorted(profile.items())))
 (P/'full-stage-progress-20260926.json').write_text(json.dumps(dict(status='whole_source_proposal_pending_validation',rows=len(rows),audit_units=len(audit),statuses=dict(Counter(x['status'] for x in audit)),canonical_modified=False,pending=['Full source-wide context closure','Complete grammar/tag audit','Exact-analysis citation-preserving reuse','Metadata and profile checks','Focused tests','Independent fresh original-source audit']),indent=2)+'\n')
 print(len(rows),'rows',len(audit),'source units')
if __name__=='__main__':main()
