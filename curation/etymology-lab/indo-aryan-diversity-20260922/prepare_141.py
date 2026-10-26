import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('jān', '5193', 'reflex', 'andersen1990[p. 148 (printed p. 150)];CDIAL[5193]',
     "Andersen's jān- 'to know' head compares RV and Pali jānāti and records the Ashokan optative jāneyu/jānevu. Turner 5193 jānā́ti 'knows' lists Pali jānāti and Ashokan imperative jānaṁtu/future jānisaṁti, confirming the same present stem."),
    ('likʰ', '11048', 'reflex', 'andersen1990[p. 165 (printed p. 167)];CDIAL[11048]',
     "Andersen's likʰ- 'to write' head compares RV √likh and Pali likhati, with Ashokan causative likhāpayatha and participial likhite/likhitaṁ. Turner 11048 likháti 'scratches; writes' lists Pali likhati and Ashokan causatives likhapeśami/likhāpisaṁ plus 'will write' likhiyisāmi."),
]
out = {'accepted':[{'record':records[f], 'parent':p, 'kind':k, 'citation':c, 'evidence':e} for f,p,k,c,e in rows],
       'held':[{'record':records['vas'], 'reason':"Andersen's vas- head supplies Sanskrit/Pali comparison and compound cross-references but no independent inscriptional form on the printed page. Turner 11435 gives direct Ashokan vasati, but the installed source record may be a cross-reference stem rather than an observed lexical item."},
               {'record':records['sātireka'], 'reason':"Andersen compares Sanskrit/Pali sātireka 'having excess', whereas Turner 212 is the unsuffixed atirēka 'surplus'. The sa- formation is plausible, but its derivational edge should not be asserted without resolving the exact source analysis."}]}
(base/'pass141-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
