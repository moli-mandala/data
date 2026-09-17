import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass167';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='4976',citation='CDIAL[4976]',evidence='Full CDIAL chattvara gives Lahnda/Punjabi chappar thatched roof, West Pahari chappar straw hut/thatched roof, and specifically Jaunsari chāpar in the addendum. The survey chappar/chapor responses match this family; affricate notation, gemination and vowels are preserved. Regional IA transmission remains unresolved.'),
 dict(parent='4971',citation='CDIAL[4971]',evidence='Full CDIAL *chatti gives Awankari chat, Punjabi chatt, West Pahari chattī, Hindi chāt and Oriya chāta roof. These support the selected t-final roof forms. Survey catʰ/tʃʌtʰə aspiration placement and śat/chot vowels or initial variation remain qualified, not silently normalized. The article explicitly reports Punjabi to Hindi to Nepali loans and questions a Sindhi source for Gujarati; these links identify the supported family without settling local IA transmission.'),
 dict(parent='4981',citation='CDIAL[4981]',evidence='Full CDIAL chadman, roof sense 1, explicitly gives Bengali chād roof. Bishnupriya chad/śad retain its d-final stem, with initial affricate/fricative and aspiration notation preserved. The match follows the eastern IA comparandum; regional borrowing versus inheritance remains unresolved. Chal and mixed chad/cal responses require separate analysis and are excluded.'),
 dict(parent='5017',citation='CDIAL[5017]',evidence='Full CDIAL chādana gives Oriya chāāṇa/chāaṇi thatching, then explicitly explains replacement by a Middle IA causative-stem formation in -āpana: Prakrit chāvaṇa covering, Bengali/Maithili chāuni thatch, Oriya chāuṇi huts and Gujarati chāvaṇ thatch/chāvṇī huts. The cauni/chavani roof responses fit that documented replacement family. Canonical node 5017 includes this unnumbered formation; this is not a claim of simple phonetic descent from unsuffixed chādana. Local IA transmission remains unresolved.')]
sets=[{'tʃʌpːʌɾa','tshappar','chapor'},{'catʰ','catʰ / cat','cʰa.to','chot','śat','tʃʌtʰə'},{'chad','śad'},{'ca.ṇi','cauni','caũni','cʰauni','cauṇi','chāvəṇi'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='roof':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
