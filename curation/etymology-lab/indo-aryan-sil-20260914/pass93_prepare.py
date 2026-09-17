import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass93';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='11109',citation='CDIAL[11109]',evidence='CDIAL 11109 leṭyati gives Lahnda leṭaṇ, Punjabi/Western Pahari leṭṇā, Nepali leṭnu and Hindi leṭnā lie down. The selected bare leṭ and simple leṭa/leṭo responses use this stem; leṭṇā is an ordinary infinitive explicitly documented by the entry. Mixed imperative/past elicitation labels and source final vowels are retained. The proposed deeper non-Aryan connection and local Indo-Aryan transmission remain unresolved.'),dict(parent='5020',citation='CDIAL[5020]',evidence='CDIAL 5020 chādi explicitly gives Sindhi/Lahnda/Punjabi chāī ashes, Assamese sāi and Bengali chāi, as well as Hindi chāī. The selected cāī/caī/chai/tʃhai/tśai/ʦaī survey forms fit this ash noun, with initial aspiration and affricate notation retained as regional/transcription qualifications. The source ash/ashes distinction and local transmission remain open.'),dict(parent='10875-2',citation='CDIAL[10875.2]',evidence='CDIAL 10875.2 *lakkuṭa gives Lahnda lakkaṛ wood and Awan lukṛī wood, with the possible influence of rukkh explicitly noted, beside Gawri lēkeṛī firewood. Pothwari lukoṛ fits the retained-k wood family with the intervening vowel and final rhotic retained. The selected subsection avoids the k-losing laur branch; local contact remains open.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if 'lay down' in g or 'lie down' in g:
  if f in {'leṭ','leṭṇā','leʈo','leʈa'}:i=0
  elif f.startswith(('leṭ','leʈ')):held.append(dict(record=r,families=[],reason='The initial leṭ stem matches CDIAL 11109, but this full response contains additional tense/auxiliary material or several alternatives (e.g. go/remain forms). Resolve the whole construction and its boundaries before assigning a single root or ordered components; the source mixed imperative/past gloss is preserved.',passNumber=93))
 if g in {'ash','ashes'} and f in {'cāī','caī','chai','tʃhai','tśai','ʦaī'}:i=1
 if f=='lukoṛ' and g=='firewood':i=2
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass93_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
