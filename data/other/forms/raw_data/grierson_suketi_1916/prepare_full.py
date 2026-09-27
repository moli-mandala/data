"""Stage all explicit Suketi attestations, retaining shared-group attribution controls."""
from pathlib import Path
import csv,json,unicodedata,re
from collections import Counter
P=Path(__file__).resolve().parent
DATA=P.parents[4]
FILES=['full-prose-transcription.jsonl','full-specimen-transcription.jsonl','full-table-transcription.jsonl','full-shared-context-transcription.jsonl']
def nfc(s):return unicodedata.normalize('NFC',s)
def table_tags(n):
 if n<=13:return ['num']
 if n<=31:
  person=('1sg' if n<=16 else '1pl' if n<=19 else '2sg' if n<=22 else '2pl' if n<=25 else '3sg' if n<=28 else '3pl')
  return ['pron','personal',person]+(['gen','poss'] if n not in [14,17,20,23,26,29] else [])
 if n<=76:return ['noun']
 if n<=85:return ['verb','impv']
 if n<=91:return ['adv','spatial']
 if n in [92,93]:return ['pron']
 if n==94:return ['adv','interr']
 if n<=97:return ['conj']
 if n in [98,99,100]:return ['interj']+(['neg'] if n==99 else [])
 if n<=118:
  q=n-101 if n<=109 else n-110
  return ['noun','pl' if q>=4 else 'sg']+({1:['gen'],2:['dat'],3:['abl'],6:['gen'],7:['dat'],8:['abl']}.get(q,[]))
 if n<=127:
  q=n-119
  return ['noun','adj','pl' if q>=4 else 'sg']+({1:['gen'],2:['dat'],3:['abl'],6:['gen'],7:['dat'],8:['abl']}.get(q,[]))
 if n<=131:return ['noun','adj']+(['pl'] if n==130 else ['sg'])
 if n<=137:return ['adj']+(['degree'] if n in [133,134,136,137] else [])
 if n<=155:return ['noun','pl' if n in [140,141,144,145,148,149,152,155] else 'sg']
 if n<=167:return ['verb','copula','pres' if n<=161 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-156)%6]]
 if n==168:return ['verb','copula','impv']
 if n==169:return ['verb','copula','inf']
 if n==170:return ['verb','copula','participle']
 if n==171:return ['verb','copula','conjunctive-participle']
 if n in [172,173,174]:return ['verb','copula','1sg']+{172:['subjunctive'],173:['fut'],174:['conditional']}[n]
 if n<=178:return ['verb']+{175:['impv'],176:['inf'],177:['participle'],178:['conjunctive-participle']}[n]
 if n<=190:return ['verb','pres' if n<=184 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-179)%6]]
 if n<=194:return ['verb','1sg']+{191:['pres','progressive'],192:['pret','progressive'],193:['pret','perfect'],194:['subjunctive']}[n]
 if n<=200:return ['verb','fut',['1sg','2sg','3sg','1pl','2pl','3pl'][n-195]]
 if n==201:return ['verb','1sg','conditional']
 if n<=204:return ['verb','1sg','pass',{202:'pres',203:'pret',204:'fut'}[n]]
 if n<=216:return ['verb','pres' if n<=210 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-205)%6]]
 if n==217:return ['verb','impv']
 if n in [218,219]:return ['verb','participle']+(['pret'] if n==219 else [])
 return ['sentential']
def generate():
 rows=[];audit=[]
 for file in FILES:
  units=[json.loads(x) for x in (P/file).read_text().splitlines()]
  for u in units:
   assert u['review']=='visual-second-pass',u['source_unit_key']
   u=dict(u);key=u['source_unit_key'];page=u['printed_page'];notes=[];tags=[];forms=[]
   if file.startswith('full-shared') or u.get('status')=='other-lect-control':
    status=u['status'];reason=u.get('note',u.get('decision','Explicit other-lect comparison'))
   elif u['form']=='…':status='source-blank';reason='Printed ellipsis; no inferred form'
   else:
    status='ingested';reason='Complete explicit Suketi source attestation'
    if 'prompt' in u:
     tags=table_tags(u['prompt']);notes.append('Grammar follows the explicit English survey prompt; simple lexical POS is contextual, not an independently printed source label.')
     if u['prompt'] in [92,93]:notes.append('Source prompt does not distinguish interrogative from relative use; no subtype inferred from another lect.')
     if u['prompt'] in [133,136]:notes.append('Source parenthesizes the comparative phrase '+u['form'].split(')')[0]+'); parentheses retained.')
     if u['prompt'] in [134,137]:notes.append('Source gloss identifies the superlative construction; no inflectional degree is inferred.')
     if u['prompt']==50:
      forms=[('Bahṇ',u['gloss'],tags,''),('bhēṇ',u['gloss'],tags,''),('bhaiṇā','sister (oblique)',tags+['obl'],'Source explicitly labels this form oblique.')]
     else:forms=[(f.strip(),u['gloss'],tags,'') for f in u['form'].split(';')]
    elif 'line' in u:
     notes.append('Aligned specimen cell; source word boundaries and hyphens retained. Gloss is the source’s local alignment, not an inferred citation form.')
     g=u['gloss']
     if g in ['sons','son','father','share','account','goods','famine','servant','swine','the-swine','husks']:tags=['noun']
     elif g in ['all','anything']:tags=['pron','indef']
     elif g in ['not']:tags=['negator']
     elif g in ['and']:tags=['conj']
     elif g in ['O']:tags=['interj']
     elif g in ['two']:tags=['num']
     elif g in ['when','then']:tags=['adv','temporal']
     elif g=='there':tags=['adv','spatial']
     elif g in ['may-come','may-eat']:tags=['verb','subjunctive']
     elif g in ['was-asked','was-given','was-wasted','he-was-sent','were-given']:tags=['verb','pret','pass']
     elif g in ['were','went','was-completed','fell','remained','he-remained','it-was-thought']:tags=['verb','pret']
     elif g=='give':tags=['verb','impv']
     elif g=='feeding':tags=['verb','participle']
     elif g=='made-having':tags=['verb','conjunctive-participle']
     elif g in ['I','my','his-own','me-to','by-him','them-to','those','that','what','which','him-of','by-anyone']:tags=['pron']
     if g in ['my','his-own','him-of']:tags+=['poss']
     if g in ['I','my','me-to']:tags+=['1sg']
     if g in ['by-him','him-of']:tags+=['3sg']
     forms=[(u['form'],g,tags,'')]
    else:
     tags=u['tags'];notes.append(u.get('note',''));forms=[(u['form'],u['gloss'],tags,'')]
    if u.get('flags'):notes+=u['flags'];tags.append('uncertain')
   locator=f'p. {page}, '+(f"item {u['prompt']}, Suketi column" if 'prompt' in u else f"specimen line {u['line']}, cell {u['cell']}" if 'cell' in u else f"{u.get('block','shared vocabulary')}, {key.rsplit(':',1)[-1]}")
   keys=[]
   for i,(form,gloss,t,extra) in enumerate(forms,1):
    k=key+(f':answer{i}' if i>1 else '')
    if u.get('prompt')==50 and i==3:k=key+':oblique'
    local=list(t)
    if ' ' in form or '-' in form:local.append('multiword-expression')
    rows.append(['suk','',nfc(form),gloss,'','',' '.join(x for x in notes+[extra] if x),f'grierson1916suketi[{locator}]','','',k,'','','',' '.join(dict.fromkeys(local))]);keys.append(k)
   audit.append({**u,'status':status,'reason':reason,'entry_keys':keys,'citation_locator':locator,'source_transcription_layer':'literal historical Roman; Phonemic blank','language_id':'suk' if keys else u.get('language_id')})
 assert len(audit)==440 and len({r[10] for r in rows})==len(rows)
 legacy=list(csv.reader((DATA/'data/other/forms/20260925-grierson-suketi.csv').open()))
 assert {r[10] for r in legacy}<={r[10] for r in rows}
 return rows,audit
def main():
 rows,audit=generate()
 with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (P/'proposal-audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit))
 graphemes=set()
 for r in rows:
  clusters=[]
  for c in r[2]:
   if unicodedata.combining(c) and clusters:clusters[-1]+=c
   else:clusters.append(c)
  graphemes.update(clusters)
 (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(c+'\t'+('#' if c==' ' else c.lower().replace('w','v').replace('ṅ','ŋ'))+'\n' for c in sorted(graphemes)))
 print(len(rows),'rows;',len(audit),'units;',dict(Counter(x['status'] for x in audit)))
if __name__=='__main__':main()
