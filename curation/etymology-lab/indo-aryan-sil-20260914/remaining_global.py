import json,collections,sys
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
g=collections.defaultdict(list)
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 if x['record']['ID'] not in done:g['/'.join(x['parents'])].append(x['record'])
for k,rs in sorted(g.items(),key=lambda z:-len(z[1])):
 if '--numeric' in sys.argv and not k[0].isdigit():continue
 print(k,len(rs),'; '.join(sorted({r['Language_ID']+' '+r['Form']+' «'+r['Gloss']+'»' for r in rs})))
