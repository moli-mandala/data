import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('dakʰ', '6507', 'reflex', 'andersen1990[p. 151 (printed p. 153)];CDIAL[6507]',
     "Andersen's dakʰ- 'to see' head compares RV √dr̥ś and Pali dassati/dakkhiti and records Maski dakhitaviye. Turner 6507 *dr̥kṣati, branch 1, explicitly lists Pali dakkhati and Ashokan dakhati with gerundive dakhitaviye 'sees'."),
    ('vadʰ', '11376', 'reflex', 'andersen1990[p. 166 (printed p. 168)];CDIAL[11376]',
     "Andersen's vadʰ- 'to increase, grow' head compares RV vardhati and Pali vaḍḍhati and records the Ashokan future vaḍhisiti/vaḍhisati series. Turner 11376 várdhatē directly lists Pali vaḍḍhati, Ashokan vaḍhiśati future and vaḍhati."),
]
out = {'accepted':[{'record':records[f], 'parent':p, 'kind':k, 'citation':c, 'evidence':e} for f,p,k,c,e in rows],
       'held':[{'record':records['gam'], 'reason':"Andersen's gam- head supplies Sanskrit/Pali comparison but no independent inscriptional form; Turner 3955 gácchati has Ashokan examples, yet the installed source item may be only a cross-reference stem for compounds. Hold direct linking until that editorial boundary is clear."},
               {'record':records['vac'], 'reason':"Andersen's vac- head lists future passive participles vataviyaṁ/vativiy(e), but the precise Turner article for this verbal formation was not established; avoid assigning by the shared vac- root alone."}]}
(base/'pass138-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
