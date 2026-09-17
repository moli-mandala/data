import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass87';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='2563',citation='CDIAL[2563]',evidence='CDIAL 2563 oṣṭha explicitly gives Nepali oṭh, Kumauni ōṭh and Maithili/Bhojpuri oṭh lip. The selected nonnasal, non-h-initial oṭh/oṭʰ lips responses fit that plain lip stem, with the survey plural gloss and aspiration notation preserved. No lower-lip compound is inferred; local Indo-Aryan transmission remains open.'),dict(parent='12583',citation='CDIAL[12583]',evidence='CDIAL 12583 śṛṅga gives Assamese xiṅ and Bengali siṅ horn, beside regional śiṅ/siṅ forms. Hajong/Bishnupriya hiŋ horn of buffalo fits this eastern horn family with the weakened h-initial fricative retained as a local qualification. The buffalo restriction is an elicitation qualification; the link leaves local Indo-Aryan transmission open and does not replace the source h with s.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=0 if r['Gloss']=='lips' and r['Form'] in {'oṭh','oṭʰ'} else 1 if r['Gloss']=='horn (of buffalo)' and r['Form']=='hiŋ' and r['Language_ID'] in {'Hajong','Bishnupriya'} else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass87_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
(P/'pass87-next-research.md').write_text('''# Candidates still requiring primary evidence

The homepage kad/kadī searches returned mostly unlinked surveys and irrelevant substring results. CDIAL 2910 karhi gives Kashmiri kar but does not establish the dental-d forms. These records remain unresearched pending a primary kadā/kadī treatment and a valid existing parent; no karhi link was inferred.

The thūm/thom garlic queries likewise returned unlinked forms only. Seek a documented lexical family or donor, rather than choosing a generic garlic parent.
''')
print({'accepted':len(acc)})
