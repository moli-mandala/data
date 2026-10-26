"""Read-only source/language coverage census using current graph plus accepted sidecars."""
import csv,json,sys,collections,datetime
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from etymology_assignments import read_assignments
P=Path(__file__).resolve().parent
clades={'OIA','MIA','Early NIA','Pashai','Chitrali','Shinaic','Kohistani','Kunar','Kashmiric','Sindhic','Lahndic','Punjabic','W. Pahari','C. Pahari','E. Pahari','Eastern','Bihari','E. Hindi','W. Hindi','Rajasthanic','Gujaratic','Bhil','Khandeshi','Marathi-Konkani','Halbic','Insular','Migratory'}
langs={r['ID']:r for r in csv.DictReader(open('cldf/languages.csv')) if r['Clade'] in clades and r['ID'] not in {'Indo-Aryan','Indo-ir','IE','Ind','Lom'}}
overlay=read_assignments();rejected={(r['Form_ID'],r['Etymon_ID']) for r in overlay if r['Status']=='rejected'}
linked={r['Form_ID'] for r in overlay if r['Status']=='accepted' and r['Rank']=='1'}
linked|={r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1' and (r['Child_ID'],r['Parent_ID']) not in rejected}
del overlay
c=collections.defaultdict(lambda:collections.Counter());samples=collections.defaultdict(list)
for r in csv.DictReader(open('cldf/forms.csv')):
 if r['Language_ID'] not in langs or r['Redirect'] or 'loan-source' in r['Tags'].split():continue
 keys={s.strip().split('[',1)[0] for s in r['Source'].split(';') if s.strip()}
 for key in keys:
  pair=(key,r['Language_ID']);c[pair]['total']+=1
  if r['ID'] in linked:c[pair]['linked']+=1
  else:
   c[pair]['remaining']+=1
   if len(samples[pair])<12:samples[pair].append({k:r[k] for k in ['ID','Form','Gloss','Source','Status']})
source_totals=collections.defaultdict(collections.Counter)
for (source,lang),counts in c.items():source_totals[source].update(counts)
rows=[]
for pair,counts in c.items():
 if counts['remaining']:
  rows.append(dict(source=pair[0],language=pair[1],name=langs[pair[1]]['Name'],clade=langs[pair[1]]['Clade'],**counts,linkedFraction=counts['linked']/counts['total'],sample=samples[pair]))
for r in rows:
 r['sourceTotal']=source_totals[r['source']]['total'];r['sourceLinkedFraction']=source_totals[r['source']]['linked']/r['sourceTotal']
rows.sort(key=lambda r:(0 if 100<=r['sourceTotal']<=2000 and r['sourceLinkedFraction']<.5 else 1,r['sourceTotal'],r['linkedFraction']))
(P/'source-language-coverage.json').write_text(json.dumps(rows,ensure_ascii=False,indent=1)+'\n')
(P/'inventory-note.json').write_text(json.dumps({'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Indo-Aryan attested language clades including Romani/Domari; excludes Nuristani, Lomavren mixed language, proto-entry language Indo-Aryan, redirects and generic loan-source nodes. Each citation key gets an attribution, so rows across sources are not additive.','priority':'First: 100–2000-record sources below 50% linked, ordered by source size, then language coverage. Smaller citation fragments remain visible but are not mistaken for full lexical sources. Candidates require lexical and source ownership verification.'},indent=1)+'\n')
for r in rows[:35]:print(r['source'],r['language'],r['total'],r.get('linked',0),r['remaining'],[(x['Form'],x['Gloss']) for x in r['sample'][:3]])
