import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass244';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='2830',citation='CDIAL[2830]',evidence='Full CDIAL karṇa explicitly gives Kalasha kuṛõ/kᵘṛũ/kṛä̃, broader reduced nasal-vowel reflexes, and regional kān/kāna ear. Kalasha kụ̃/kạ/kʰạ̃, Bhatri kan.o and Dogri ka are provisionally grouped with that family, retaining the survey loss of rhotic or nasal segments, aspiration and punctuation exactly. The short Kalasha and Dogri forms require qualified local phonetic histories; no new intermediate stem is asserted. Regional IA transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='ear':continue
 if r['Form'] in {'kụ̃','kạ','kʰạ̃','kan.o','ka'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));continue
 reason=None
 if r['Form'] in {'æn','en','an'}:reason='Full karṇa 2830 gives Bshk kan ear. These Bshk vowel-initial æn/en/an responses may belong there, but the initial loss and vowel change need local lexical or sound evidence. No exact vowel-initial Bshk comparison was located; uncertain IA transmission alone is not the issue.'
 elif r['Form'] in {'kamta','kanik','kāiḍā','kanto'}:reason='A karṇa ear base is plausible, but the exact ending or compound formation remains unclear. Full 2830 and 2831 were inspected: karṇaka 2831 documents handles/rims and related senses, not direct support for these whole local ear responses. Do not silently discard -mta/-ik/-iḍā/-to; seek local morphology and specific extension evidence.'
 elif r['Form']=='buca':reason='Full bucca 9266 explicitly gives crop-eared/earless and related defect senses, not ordinary ear. The Mewari buca ear response requires evidence for a local nominal use or clarification of the source elicitation. Similar spelling alone does not justify the semantic reversal.'
 elif r['Form']=='ḍunḍa':reason='Mewari ḍunḍa ear remains unidentified in this ear-family review. Neither full karṇa nor bucca articles provide a matching ordinary-ear form; a regional dictionary comparison or elicitation check is needed. No etymon is proposed.'
 if reason:held.append(dict(record=r,families=[0] if r['Form'] not in {'buca','ḍunḍa'} else [],passNumber=244,reason=reason))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare244.py').read_text());print('accepted',len(acc),'held',len(held))
