import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('aham', '992', 'reflex', 'andersen1990[p. 138 (printed p. 140), p. 175 (printed p. 177)];CDIAL[992]',
     "Andersen's aham line says 'see hakam'; the hakam pronoun paradigm labels the Gujarra aham as nominative singular 'I' and compares RV/Pali ahaṁ. Turner 992 ahám 'I' explicitly lists Ashokan shah./gir. ahaṁ beside hakaṁ."),
    ('yā', '10452', 'reflex', 'andersen1990[p. 163 (printed p. 165)];CDIAL[10452]',
     "Andersen's separately bounded yā- verb means 'to go, to go on' and compares RV yāti and Pali yāti, with an inscriptional participial form. Turner 10452 yā́ti 'goes, approaches, comes to' explicitly lists Pali yāti and Ashokan yāti/yaṁti."),
    ('hu', '9416', 'reflex', 'andersen1990[p. 176 (printed p. 178)];CDIAL[9416]',
     "Andersen's hu- verb means 'to be, become', compares RV bhavati and Pali hoti, and lists inscriptional hoti/hota/hotu. Turner 9416 bhávati 'becomes, is' explicitly lists Pali hōti and Ashokan shah./man./kāl. hoti."),
]
out = {'accepted':[{'record':records[f], 'parent':p, 'kind':k, 'citation':c, 'evidence':e} for f,p,k,c,e in rows],
       'held':[{'record':records['as'], 'reason':"Andersen compares asti/atthi under as-, but the attested series in this head is chiefly sumi/smi as first-person 'I'; he quotes a disputed analogical account. A single edge to Turner 977 ásti would overstate the provenance of those forms."},
               {'record':records['sumi'], 'reason':"Andersen uses sumi both as a first-person nominative and in instrumental replacement contexts under hakam; the source also cross-references as-. The merged uses should not be collapsed under one verbal or pronominal parent."}]}
(base/'pass131-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
