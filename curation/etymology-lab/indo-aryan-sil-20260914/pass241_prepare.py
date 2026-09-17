import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass241';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4701-2',citation='CDIAL[4701]',evidence='Full CDIAL carman separately labels a -ḍa extension, including Punjabi camṛā, Hindi camṛā, Gujarati cāmḍũ/cāmḍī and Marathi cāmḍẽ/cāmḍī skin/hide. The existing node 4701-2 represents that unnumbered extension. The selected cambaḍ-/cambṛ-/sambaḍ- skin forms are provisionally grouped here, retaining the local medial b (an extra stop alongside m), vowels, affricate/sibilant notation and retroflex stop/rhotic variation. No independent b morpheme or ancient b-bearing reconstruction is asserted. Regional transmission and exact phonetic developments remain qualified. Complete same-family slash responses retain both forms.')]
forms={'tsambaḍo','tsambaḍa','tsambaḍi','cāmbəḍu','cāmbaḍa','cambaḍo','cambaḍi','cambaḍa','sambaḍo','sambaḍa','sambṛi','ʦambrī','tʃʌmbəɖi','ˈtʃʌmbəɖi','tʃəmbəɖi','camṛī / cambarā','cambṛī / cambṛa'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='skin' and r['Form'] in forms:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'-primary-articles.json')).write_text(json.dumps({'4701':json.loads((P/'pass240-primary-articles.json').read_text())['4701']},ensure_ascii=False,indent=1)+'\n')
(P/(stem+'-discovery-reference.json')).write_text(json.dumps(dict(homepageDiscovery='pass240-homepage-discovery.json',parentCheck='4701-2 exists, status entry, no redirect',sourceEntry='4701 unnumbered -ḍa extension'),indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare241.py').read_text());print('accepted',len(acc))
