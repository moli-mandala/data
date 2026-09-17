import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass254';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6767',citation='CDIAL[6767]',evidence='Full CDIAL dhavalá white documents Punjabi dhaulā, West Pahari dhauḷā, Kumaoni dhaulo, Hindi dhaulā/dhaurā/dhorā, Gujarati dhɔḷũ and Marathi dhavaḷ, with Sindhi dhaũro. These support the survey lateral/rhotic white forms and retained medial v. Source aspiration, voicing, retroflexion, vowels, nasalization, gemination and local inflection remain intact. Secondary lateral syllables and -io/-iyo forms are provisionally assigned to the regional family, without inventing ancient suffixes; their exact local history and cross-IA transmission remain qualified.')]
forms={'ḍʰoṛā','dāulo','doulo','dɦauḷio','dɦovəḷo','dɦovḷo','dɦovliyo','dɦoveḷo','dɦāveḷalo','dɦāvḷũ','dauḷo','dɦauḷə','davāḷlo','dɦaveḷḷā','dhoḷəḷu','dɦoḷũ','dɦoḍu','dhōḷu','dʰũḷa','ḍhaulo','ḍhouḷu','ḍɦaulio','ḍɦaolio'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='white' and r['Form'] in forms:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[])),('primary-articles',{'6767':json.loads((P/'pass253-primary-articles.json').read_text())['6767']})]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare254.py').read_text());print('accepted',len(acc))
