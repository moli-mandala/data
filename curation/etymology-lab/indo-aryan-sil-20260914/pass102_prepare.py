import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass102';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='1135',citation='CDIAL[1135, sense 1]',evidence='CDIAL 1135 ātman explicitly gives Hindi āp respectful you, Old Marwari āpa self/honorific you/inclusive we and Gujarati āpaṇ inclusive we, beside Middle Indo-Aryan appā/appaṇa. The selected āp/āpaṇa/apaṇa/āpu/apu/apũ responses fit that reflexive-pronoun family with regional final vowels and nasalisation retained. Source person, inclusivity and politeness labels remain exact; local Indo-Aryan transmission is unresolved.'),
dict(parent='1135',citation='CDIAL[1135, sense 1];CDIAL[11119]',evidence='The full āp log expression contains polite you plus people. CDIAL 1135 documents Hindi āp respectful you and 11119 Prakrit lōga people with pluralising use in Old Bengali. Two ordered component edges preserve the survey expression, including vowel length, lɔg/log spelling and joined spacing; this is not an inherited Sanskrit compound or a settled local borrowing route.'),
dict(parent='9051',citation='CDIAL[9051];CDIAL[9092]',evidence='The full phal-phul expression contains fruit plus flower: CDIAL 9051 gives Nepali/Hindi phal fruit and 9092 Nepali phul, Hindi phūl flower. Two ordered component-family edges retain both parts of the survey response meaning fruit collectively, with source spacing, vowel length and reduced first vowel preserved. This does not assign the whole response to the fruit root alone or claim an inherited Sanskrit compound; local Indo-Aryan transmission remains open.'),
dict(parent='9051',citation='CDIAL[9051]',evidence='CDIAL 9051 phala explicitly gives Oriya phaḷa, Bastar phara and Gujarati/Marathi phaḷ fruit. Oriya pʰolo, Halbi pʰor and Bhilali phoḷə fit the regional fruit family with o/a and lateral/rhotic correspondences retained; the dictionary itself notes deeper Dravidian comparisons and a possible connection with phulla without settling them. Local Indo-Aryan transmission remains open.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None;cs=None
 pron=g.lower().startswith(('you','we (','we_')) or g=='we'
 if f in {'āp','āpaṇa','apaṇa','āpu','apu','apũ'} and pron:i=0
 if f in {'āplɔg','aap log','ap log'} and g.lower().startswith('you'):i=1;cs=['1135','11119']
 if f in {'pʰalpʰul','pʰəl pʰul','phal phul','pʰalpʰūl','phəlphul','phəl phul'} and g=='fruit':i=2;cs=['9051','9092']
 if f in {'pʰolo','pʰor','phoḷə'} and g=='fruit':i=3
 if pron and f in {'āpṇo','āpṇũ','āpəṇũ','āpəṇu','apnə','apanā','apəna'}:held.append(dict(record=r,families=[],reason='CDIAL 1135 distinguishes reflexive/self and polite/inclusive-pronoun uses from 1135.2 *ātmanaka own, including Gujarati āpṇũ and Hindi apnā. This survey response has personal-pronoun use but resembles the adjectival branch. Resolve the local reanalysis and exact parent rather than choosing solely by its pronoun gloss.',passNumber=102))
 if i is not None:
  q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if cs else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.')
  if cs:x['components']=cs
  acc.append(x)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass102-fruit-primary-articles.json').read_text()));a['11119']=json.loads((P/'pass101-primary-articles.json').read_text())['11119'];f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass102_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
from collections import Counter
print(len(acc),sum(len(x.get('components',[x['parent']])) for x in acc),len(held),Counter(x['family'] for x in acc))
