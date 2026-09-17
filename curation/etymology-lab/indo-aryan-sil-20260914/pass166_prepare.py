import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass166';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='2360-4',citation='CDIAL[2360.4]',evidence='The full CDIAL 2360 article assigns Prakrit ukkhala/okkhala and modern ukhal/okhal/ukhli to subsection 4, *udukk hala (udukkʰala), distinct from the retained-initial-l subsection 2. Explicit comparanda include Gujarati ukhaḷ/ukhḷũ/ukhaḷī, Nepali okhal/okhli, Bhojpuri ōkhar and Awadhi okharī; the selected liquid-retaining forms fit this branch. Vowel and ending variation is retained; cross-IA transmission is unresolved.'),
 dict(parent='3796',citation='CDIAL[3796]',evidence='CDIAL khaṇḍana explicitly gives Gujarati khā̃ḍṇī, khā̃ḍṇiyɔ, khā̃yṇī, khā̃yṇiyɔ, khāṇṇī and khā̃ṇiyɔ meaning mortar, beside pounding in a mortar. These supply both the full cluster and contracted nasal/y stems of the selected Bhil survey forms. Vowels, nasal realization and endings remain as transcribed; inheritance versus local IA borrowing is unresolved.'),
 dict(parent='12459',citation='CDIAL[12459]',evidence='CDIAL śilā explicitly records Prakrit silā stone slab, Hindi and Bengali sil flat grinding stone, and Oriya siḷa stone/grinding stone. Bhatri sil mortar matches this grinding-stone family in form and implement sense; the source gloss is preserved. Local IA transmission is unresolved. Extended sila-ut forms are excluded pending morphological evidence.')]
sets=[set('ukhḷyu ukhoḷ ukhḷiyo okhaḷ okhḷiyā okhaḷiyā ukhḷiya ukhol okhalu ukhale okʰʌɾi'.split()),set('khanḍaṇiu khāṇḍəṇyo khānaṇyo khanaṇi khāiṇyo khānyo khanyā khanḍaṇi khaŋḍaṇi khaŋḍaṇiu khaŋḍaṇio'.split()),{'sil'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='mortar':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
