import csv,json,collections,sys,re
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/dravidian-surveys-20260921';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
langs={r['ID'] for r in csv.DictReader(open(root/'data/cldf/languages.csv')) if 'Dravidian' in r['Clade'] or r['Clade']=='Brahui'}
assigned={r['Form_ID'] for r in read_assignments() if str(r.get('Rank'))=='1'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open(root/'data/cldf/edges.csv')) if r['Rank']=='1')
proposed=set();held=set()
for pass_name in json.load(open(p/'goal-pass-registry.json')):
 for decisions_path in sorted((p.parent/pass_name).glob('proposed-decisions*.json')):
  dec=json.load(open(decisions_path));proposed.update(a['record']['ID'] for a in dec['accepted']);held.update(a['record']['ID'] for a in dec['held'])
groups=collections.defaultdict(collections.Counter);records=[]
for r in csv.DictReader(open(root/'data/cldf/forms.csv')):
 if r['Language_ID'] not in langs or r['Status'] in {'entry','redirect'}:continue
 # The LSI Gadabā survey is `bodo1267` (Bodo Gadaba/Gutob, South Munda),
 # although its imported Language_ID is Dravidian Gadaba `mudh1235`.
 # Keep this cross-family source out of the Dravidian research queue until
 # source attribution is corrected through the separate ingestion workflow.
 if 'grierson-lsi1928' in r['Source'] and 'LSI-GADABA' in r['Tags']:continue
 src=re.split(r'[;\[]',r['Source'])[0];c=groups[(src,r['Language_ID'])];c['total']+=1
 state='linked' if r['ID'] in assigned else 'proposed' if r['ID'] in proposed else 'held' if r['ID'] in held else 'unexamined';c[state]+=1
 if state=='unexamined':records.append(r)
res=[]
for (src,l),c in groups.items():
 if c['unexamined']:
  res.append(dict(source=src,language=l,**c,coverage=round(c['linked']/c['total'],4)))
res.sort(key=lambda x:(x['total']>500,x['coverage']>=.5,x['total'],x['source'],x['language']))
suffix=sys.argv[1] if len(sys.argv)>1 else '123'
(p/f'all-source-priorities-{suffix}.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n');(p/f'all-source-unexamined-{suffix}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
for r in res[:45]:print(r)
print('unexamined',len(records),'groups',len(res))
