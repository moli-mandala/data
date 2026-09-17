import json,csv,collections
from pathlib import Path
P=Path(__file__).resolve().parent
ledger=json.loads((P/'pass-ledger.json').read_text());done=set()
for f in ledger['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for kind in ('accepted','held') for x in d[kind])
g=collections.defaultdict(list)
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if x['record']['ID'] not in done:g['/'.join(x['parents'])].append(x['record'])
for k,rs in sorted(g.items(),key=lambda kv:-len(kv[1])):
 print(k,len(rs),'; '.join(sorted(set(r['Language_ID']+' '+r['Form']+' «'+r['Gloss']+'»' for r in rs))))
