import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass83';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
 dict(parent='12278',citation='CDIAL[12278]',evidence='CDIAL 12278 śata gives Nepali/Maithili sai, Punjabi/Lahnda sau/so, Oriya sae and Western Pahari śau/śɔ. The selected simple hundred responses match those regional numeral forms, preserving final vowels, diphthongs and source articulation. Local Indo-Aryan transmission remains open.'),
 dict(parent='2462-2',citation='CDIAL[2462.2];CDIAL[12278]',evidence='The full numeral phrase is compositionally one × hundred. Its ordered etymological components are ek, the retained-k branch *ēkka in CDIAL 2462.2, and sai/say/sau/so/śau, the hundred family śata in CDIAL 12278. These are component-family links for the two words, not a claim that the complete phrase descends from a single Sanskrit compound. Source spacing and pronunciation are preserved; local Indo-Aryan transmission remains open.'),
 dict(parent='7655',citation='CDIAL[7655];CDIAL[3503]',evidence='The hundred expression means five scores: pānc/pãc/panc five (CDIAL 7655 pañca) multiplied by kori/koṛi/koḍi score, twenty (CDIAL 3503 *kōḍi, with explicit Nepali kori, Oriya koṛi, Hindi koṛī and Marathi koḍī). Two ordered component-family links preserve this vigesimal formation. They do not equate the whole expression with a bare śata descendant or settle the deeper Austro-Asiatic hypothesis and regional transmission of the score word.')]
bare={'so','soie','soe','soye','səye','səⁱ','soᵘ','saᵒ','ʃoʊ','ʃoːʊ'}
one={'ekso','ek sai','ek say','ek sau','ek-so','ek-sō','ek saũ','ˈek ʃoʊ','ek ʃo','ek-soː'}
five={'pānckorī','pā̃nckorī','pãc koṛi','panc koḍi'}
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'].lower() not in ['hundred','one hundred','one_hundred','100']:continue
 f=r['Form'];i=0 if f in bare else 1 if f in one else 2 if f in five else None
 if i is None:continue
 q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='reflex' if i==0 else 'component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.')
 if i:x['components']=['2462-2','12278'] if i==1 else ['7655','3503']
 acc.append(x)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass83_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print({'records':len(acc),'rows':sum(len(x.get('components',[])) or 1 for x in acc)})
