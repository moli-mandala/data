import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass124';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11503-5',citation='CDIAL[11503.5]',evidence='CDIAL 11503.5 vaṅga explicitly gives Bastar Oriya bāṅgā eggplant, beside Prakrit vaṁga. Bhatri baṇga fits this geographically specific shortened branch; the source nasal is written ṇ before g rather than the dictionary’s ṅ. That notation is retained, not silently corrected. The article discusses probable deeper Dravidian origin; no direct Dravidian loan is asserted for these survey responses.'),dict(parent='11503',citation='CDIAL[11503.1]',evidence='CDIAL 11503.1 vātiṅgaṇa gives Punjabi vaĩgaṇ/baĩgaṇ, Hindi baigan and Oriya bāiṅgaṇa. Kaithal beŋgan, Gojri beŋan and Oriya baiṇgoṇõ fit this longer branch, retaining vowel, nasal notation and velar weakening in beŋan. The shortened vaṅga and separate vaṅgana subsections are not selected. Local Indo-Aryan transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='eggplant':continue
 i=0 if r['Language_ID']=='Bhatri' and r['Form']=='baṇga' else 1 if (r['Language_ID'],r['Form']) in {('kaithal','beŋgan'),('Goj','beŋan'),('Or','baiṇgoṇõ')} else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass124-vatingana-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
