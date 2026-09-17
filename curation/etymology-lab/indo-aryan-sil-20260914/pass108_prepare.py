import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass108';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
base='CDIAL 10187.11 *mōṭṭa explicitly gives Old Marwari moṭaü big/fat and Gujarati moṭũ; the adjective is used in the source elder-sibling expression. The specific o-vowel subsection is selected, not the general *muṭṭa head or *mōṭṭha subsection. '
rules=[dict(parent='10187-11',citation='CDIAL[10187.11];CDIAL[9661]',evidence=base+'CDIAL 9661 gives Hindi/Marwari bhāī and Gujarati bhāi brother. Two ordered component edges preserve big/elder + brother, with regional final-vowel, aspiration and y/i notation retained. No inherited Sanskrit compound or settled local Indo-Aryan transmission is asserted.'),dict(parent='10187-11',citation='CDIAL[10187.11];CDIAL[9349]',evidence=base+'CDIAL 9349 gives Hindi bahan, Gujarati bahen/ben and regional baiṇ sister. Two ordered component edges preserve big/elder + sister, retaining source feminine moṭī/moṭi and vowel/nasal notation. The English elder label belongs to the full phrase, not the sister root alone; local Indo-Aryan transmission remains open.')]
bs={'moṭobʰāī','moṭɔ bʰāyī','moṭā bɦāi','moṭobhāy','moṭobhāi','moṭābhāi'};ss={'moṭībɛhān','moṭībɛhan','moṭībɛn','moṭi bāiṇ','moṭi ben'};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=0 if r['Form'] in bs and r['Gloss']=='older brother' else 1 if r['Form'] in ss and r['Gloss']=='older sister' else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],components=['10187-11','9661' if i==0 else '9349'],family=i,kind='component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());z=json.loads((P/'pass90-primary-articles.json').read_text())
for k in ['9661','9349']:a[k]=z[k]
f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass108_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print(len(acc),len(acc)*2)
