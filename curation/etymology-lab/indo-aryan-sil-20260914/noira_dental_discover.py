import csv,json,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent;ledger=json.loads((P/'pass-ledger.json').read_text());accepted=[x for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']];done={x['record']['ID'] for x in accepted}
ns={'__file__':str(P/'sixth_prepare.py')};exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0],ns);norm=ns['norm']
idx=collections.defaultdict(list)
for x in accepted:
 if not x['parent'][0].isdigit():continue
 r=x['record'];idx[(norm(r['Form']),r['Gloss'].lower())].append(x)
cs=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in done or 'varghesekumar2015noira' not in r.get('survey_sources',[]) or not re.search('[ḍṭ]',r['Form']):continue
 w=norm(r['Form'].replace('ḍ','d').replace('ṭ','t'));xs=idx.get((w,r['Gloss'].lower()),[])
 if xs:cs.append(dict(record=r,comparanda=xs[:6],parents=sorted({x['parent'] for x in xs})))
(P/'noira-dental-discovery.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1)+'\n')
for (parents,w,g),n in collections.Counter(('/'.join(x['parents']),x['record']['Form'],x['record']['Gloss']) for x in cs).items():print(parents,repr(w),g,n)
print('total',len(cs))
