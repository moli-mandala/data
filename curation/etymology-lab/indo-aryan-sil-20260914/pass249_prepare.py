import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass249';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5428',citation='CDIAL[5428.1]',evidence='Full CDIAL ṭaṅka subsection 1(b) explicitly gives Lahnda/Punjabi ṭaṅg leg and, in the addendum, Sindhi ṭaṅg(h) and Kotgarhi ṭāṅg. Selected Gojri ṭʰaŋg, Kaithal taŋ and Kului toŋgə/tãːŋ(g) leg responses follow this northwestern branch. Source dental/retroflex and aspiration notation, vowels, final inflection and optional g remain intact. Regional IA transmission remains qualified; mixed responses containing yaŋg or caŋg are excluded.'),dict(parent='5428-2',citation='CDIAL[5428.2]',evidence='Full CDIAL ṭaṅka subsection 2 ṭaṅga explicitly gives Bengali ṭeṅri, Maithili ṭā̃g/ṭãgri leg or foot, Bhojpuri ṭāṅ/ṭaṅari, and Gujarati ṭā̃g/ṭā̃go leg. The selected eastern Danuwar/Magahi/Kochila Tharu plain forms, Danuwar taŋari, Bagheli ṭeŋgri and Bhili ṭāŋgo fit those regional comparisons. Source nasalization, dental/retroflex notation, palatalization and vowels remain unchanged; local transmission is unresolved and no unique sound law distinguishing the two overlapping branches is claimed.')]
sets=[{'ṭʰaŋg','taŋ','toŋgə','tãːŋ(g)'}, {'tãŋ','taŋari','tāŋ','ṭʸaŋ','ṭeŋgri','ṭāŋgo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='leg':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare249.py').read_text());print('accepted',len(acc))
