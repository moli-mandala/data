import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass231'
assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9758',citation='CDIAL[9758.1]',evidence='Full CDIAL matsya subsection 1 explicitly gives Marathi māsā fish, alongside Assamese mās and the wider maccha/mācha family. The selected western survey maso/mase/masu/maṣo/maṣu forms fit this regional fish family; vowel length, s/ṣ notation and final-vowel inflection remain unchanged. Local IA transmission is unresolved. These are ordinary survey fish responses, not māṃsa flesh or a newly reconstructed subtype.'),dict(parent='9758',citation='CDIAL[9758.1]',evidence='Full CDIAL matsya subsection 1 explicitly gives Gawri maċoṭá/mačoṭá fish and eastern Bengali māch/Assamese mās. Gawri macoṭa and the selected Bengali/Hajong affricate fish responses are grouped with these documented comparanda. Source aspiration, affricate notation and vowel length remain intact; regional IA transmission is unresolved.'),dict(parent='9758-3',citation='CDIAL[9758]',evidence='Full CDIAL matsya gives a separately labelled but unnumbered extension with -l-, including Prakrit maścalī, Hindi machlī and Marathi/Konkani māsḷī fish; the existing node 9758-3 represents that extension. The selected western masli/māslā/māsəlā/māśəlā/maśəḷu forms fit the lateral fish family, with s/ś, syllabic vowels and l/ḷ preserved. Regional transmission and local inflection are qualified. The distinct -ll- Gujarati branch is not automatically equated with this branch.')]
sets=[{'maso','mase','masu','māsā','masā','masa','maṣo','maṣõ','maṣe','maṣu'}, {'macoṭa','matśʰ','maʈʃh'}, {'masli','māslā','māsəlā','māśəlā','maśəḷu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='fish':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare231.py').read_text());print('accepted',len(acc))
