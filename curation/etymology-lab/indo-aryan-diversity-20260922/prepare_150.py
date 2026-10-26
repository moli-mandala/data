import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('diyaḍʰiya', '6698', 'reflex', 'andersen1990[p. 151 (printed p. 153)];CDIAL[6698]',
     "Andersen's diyaḍhiya- 'one and a half' compares Sanskrit dvyardha and Pali diyaḍḍha, gives numerous Ashokan diyaḍhiyaṁ tokens, and explicitly analyzes the form as dvi + ardha with hiatus-breaking y. Turner 6698 *dviyardha, branch 2, lists Pali diyaḍḍha and Ashokan diyaḍha-/diyadha- 'one and a half'; Andersen's -iya extension remains visible in the source form."),
]
out = {'accepted':[{'record':records[f], 'parent':p, 'kind':k, 'citation':c, 'evidence':e} for f,p,k,c,e in rows],
       'held':[{'record':records['i'], 'reason':"Andersen's i- 'to go' head compares RV/Pali eti and cross-references upe/upayita, but supplies no independent inscriptional i- form there. Turner 2515 lists Ashokan eti; hold this bare cross-reference stem until its editorial status is clear."},
               {'record':records['lipikara'], 'reason':"Andersen's lipikara- 'engraver' is a transparent compound semantically, but neither lipi nor kara is an installed source head, and Turner 11058 discusses lipi alongside Old Persian dipi; hold the compound until the exact donor/component path is reviewed."}]}
(base/'pass150-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
