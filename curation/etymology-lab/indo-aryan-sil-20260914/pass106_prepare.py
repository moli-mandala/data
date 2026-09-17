import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass106';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='6648',citation='CDIAL[6648]',evidence='CDIAL 6648 dva explicitly gives eastern dui and Western Pahari dūī two. The selected dui̯/dui ̯/du.i and Kullui dʊːi/dʊwi/ˈdʊwi retain source syllabicity, vowel length, glide and stress notation. These are bare numerals without classifier material; local Indo-Aryan transmission remains open.'),dict(parent='6984',citation='CDIAL[6984]',evidence='CDIAL 6984 nava nine gives widespread nau/nao, eastern na/naa, Old Marwari nova and Gujarati nav. The selected eastern noi/noi̯/nɔi ̯, Bhatri nʌu̯/nʌo̯, Oriya nōo and western nove are assigned to this numeral family with their specific vowel sequences and final e retained as regional qualifications. The prose does not explicitly print every surveyed diphthong; no claim of exact attestation or resolved local transmission is made.'),dict(parent='5994-2',citation='CDIAL[5994.2]',evidence='CDIAL 5994.2 *trāyaḥ explicitly gives Torwali and Maiya c̣ā three. Selected ʦ̣ā/ʦ̣ā̃ and Maiya cā fit this long-a branch, retaining source affricate/retroflex notation and isolated nasalisation. The parent is the specific *trāyaḥ subsection rather than trayaḥ or trīṇi; local Indo-Aryan transmission remains open.'),dict(parent='5994-3',citation='CDIAL[5994.3]',evidence='CDIAL 5994.3 trīṇi gives Prakrit tiṇṇi, Hindi/Marwari tīn, Gujarati traṇ and, in its addendum, Punjabi tan and Old Gujarati traṇṇi with analogical interaction. The selected western tan/təṇ/taṇ/teṇ/tāṇə/tine fit this n-final three family with r loss and source vowel/nasal/ending variation retained. This is the trīṇi branch, not the separate northwestern *trāyaḥ branch; local Indo-Aryan transmission remains open.')]
acc=[]
sets=[{'dui̯','dui ̯','du.i','dʊːi','dʊwi','ˈdʊwi'},{'noi̯','noi','nɔi ̯','nʌu̯','nʌo̯','nōo','nove'},{'ʦ̣ā','ʦ̣ā̃','cā'},{'tan','təṇ','taṇ','teṇ','tāṇə','tine'}]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=next((i for i,ss in enumerate(sets) if r['Form'] in ss and r['Gloss']==['two','nine','three','three'][i]),None)
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a['6648']=json.loads((P/'pass105-primary-articles.json').read_text())['6648'];f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass106_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))
