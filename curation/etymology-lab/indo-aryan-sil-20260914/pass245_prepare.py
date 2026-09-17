import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass245';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='1932',citation='CDIAL[1932]',evidence='Full CDIAL udara explicitly gives Prakrit uara belly, Gawri war and Maiyan wēr. Gawri var and Mai verī/varī belly fit these direct regional comparanda with source v/w notation, vowels and final ī retained. Marathi ūdar is a conservative learned-looking member of the same family; vowel length and its survey spelling remain intact. Regional transmission and final-vowel formation are qualified, without inventing an ancient ī-bearing subtype.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='belly':continue
 if r['Form'] in {'var','verī','varī','ūdar'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'tauṇḍ','toṇḍ','tʰoṇḍ'}:held.append(dict(record=r,families=[],passNumber=245,reason='Full CDIAL tunda 5858 subsection 2 tōnda directly gives Punjabi/Hindi tõd pot-belly and Tarai taun. These Kaithal o-vowel belly forms have a supported family comparison, with source aspiration/retroflexion needing qualification, but exact subsection node 5858-2 is absent from current compiled forms. Preserve for section/node repair, rather than substituting a mouth node or the wrong tunda branch.'))
 elif r['Form'] in {'dʌɳɖə','ˈdʌɳɖə'}:held.append(dict(record=r,families=[],passNumber=245,reason='Kului dʌɳɖə belly is comparable to tunda 5858 subsection 4 dunda and subsection 6 ḍuṇḍa, both represented by Gujarati pot-belly forms in the full article. Local retroflex/dental pattern and exact branch require clarification, and both specific subsection nodes are absent from compiled forms. Do not silently choose a branch or discard stress/phonetic notation.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare245.py').read_text());print('accepted',len(acc),'held',len(held))
