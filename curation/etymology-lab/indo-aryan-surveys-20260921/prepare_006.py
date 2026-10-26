import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
forms=['dʰanūśbāṇ','dʰanūsbān','dʰanuś bāṇā','dʰanūsabān','dʰanaśbān','dʰānakbān','danūśbāṇ','dʰankabāṇ','dʰanūśbān','ḍhanūś bāṇ','dʰanusban']
evidence='The whole rainbow response is analysed as bow + arrow, in that order. CDIAL 6726 gives dhanu(s), Punjabi dhaṇuk/dhaṇak and regional dhanu bow forms; 9203 gives regional bān/bāṇ arrow. The source variants preserve the same two lexical members, including vowel, aspiration and sibilant differences and the expanded dhaṇak stem. Two component links record this regional compound, not a claim that the full expression is inherited from a Sanskrit compound or that its members’ learned/contact history has been settled. Single bāṇ responses, rāmbāṇ, and responses with additional unexplained material are excluded.'
a=[dict(record=r,components=['6726','9203'],kind='component',citation='CDIAL[6726];CDIAL[9203,1]',evidence=evidence) for r in rs if r['Gloss']=='rainbow' and r['Form'] in forms]
(P/'pass006-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
print(len(a),'records',len(a)*2,'component rows')
