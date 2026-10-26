"""Verify targeted homepage exact-form discovery alignment and pagination."""
import glob
import json
from pathlib import Path
P=Path(__file__).resolve().parent
items=sorted(json.loads((P/'selection.json').read_text())['selected']+
             json.loads((P/'selection.json').read_text())['held'],key=lambda x:x['surveyIndex'])
requests=[x for name in sorted(glob.glob(str(P/'discovery-???-???-requests.json')))
          for x in json.loads(Path(name).read_text())]
results=[x for name in sorted(glob.glob(str(P/'discovery-???-???-results.json')))
         for x in json.loads(Path(name).read_text())]
assert len(items)==len(requests)==len(results)
for item,request,result in zip(items,requests,results):
    assert result['opts']==request
    assert request['params']['form']==item['record']['Form']
    assert any(x['id']==item['record']['ID'] for x in result['rows'])
    assert len(result['rows'])<=result['count']
capped=[{'form':x['opts']['params']['form'],'count':x['count'],'returned':len(x['rows'])}
        for x in results if x['count']>len(x['rows'])]
assert [x['form'] for x in capped]==['āṟ']
strict=json.loads((P/'discovery-ahr-strict-results.json').read_text())
assert len(strict)==1 and strict[0]['count']==len(strict[0]['rows'])==97
assert strict[0]['opts']['params']=={'form':'āṟ','relaxed':False}
report={'queries':len(results),'returnedRows':sum(len(x['rows']) for x in results),
        'totalCount':sum(x['count'] for x in results),'exactSourceIdHits':len(items),
        'broadCappedQueries':capped,'strictFallbackRows':97,'alignment':True}
(P/'exact-search-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report)
