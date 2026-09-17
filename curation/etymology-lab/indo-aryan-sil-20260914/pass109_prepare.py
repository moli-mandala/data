import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass109';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='9187',citation='CDIAL[9187]',evidence='CDIAL 9187 bahu explicitly gives Awan bàũ and Lahnda bahũ much/many, alongside western bo(h) forms. Pothwari baõ many fits this contracted quantity family with loss of intervocalic h and source o/nasalisation retained. The local vowel outcome is a qualification, not an exact dictionary quotation; Indo-Aryan transmission remains open.'),dict(parent='28',citation='CDIAL[28]',evidence='CDIAL 28 akṣata gives Prakrit akkhaya unbroken, Gujarati ākhũ whole, Marathi ākhā whole/undivided and Konkani ākho complete. The selected western akha/ākha/akho/akhe all responses fit this whole/entire family with source vowel length and final gender/number forms retained. This is semantic whole → all, with local Indo-Aryan transmission unresolved.'),dict(parent='5599-2',citation='CDIAL[5599.2]',evidence='CDIAL 5599.2 *dhera explicitly gives Nepali dher/dherai much, many. Nepal-area dhere many matches this dental-initial branch with its final vowel retained. The promoted dental subsection is used rather than the retroflex heap family; local Indo-Aryan transmission remains open.'),dict(parent='5599',citation='CDIAL[5599, sense 1]',evidence='CDIAL 5599 *ḍhera gives Nepali ḍher heap and Lahnda/Awadhi ḍher much/many, with related quantity senses throughout the entry. Dang ḍher many uses this retroflex-initial branch, distinct from dental Nepali dher. The heap-to-quantity development is explicit in the family; local Indo-Aryan transmission remains open.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if f=='baõ' and g=='many' and r['Language_ID']=='poth':i=0
 if f in {'akha','ākhā','ākho','akho','akhe'} and g=='all':i=1
 if f=='dʰere' and g=='many':i=2
 if f=='ḍher' and g=='many':i=3
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass109-more-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass109_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))
