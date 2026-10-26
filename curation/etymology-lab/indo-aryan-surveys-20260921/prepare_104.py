import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
groups=[
('7067',['name'],['nao̯','nāmm','ṇāu','nõ','nõ / nām','nɔ̄ʔm','na.mɔ','nãᵘ','nāme','ṇāβ'],'Turner nāman explicitly records nasal-less Oriya nā, nasalized Phalura nõ, regional nāu/nāv, and nam. The selected single-word variants, including the synonymous Phalura alternatives, fit this name family. Source punctuation and glottal-stop notation are retained. No analysis is assigned to extra lexical material or unexplained terminal p/ŋ.'),
('5260',['old'],['junlo','jūnlo','junlū','juṇlo','dzū nlū','dzū nla','jūnalɔ','juṇlo','junlā','junḷā','junla','dzū nlo','dzū nḷo','juṇlu','juṇla'],'Turner jūrṇa gives Gujarati jūnũ, Hindi jūnā and Marathi junā old. The surveyed Bhili-area jun-/juṇ- adjectives are compared with that western u-vowel family, preserving their local l-adjectival endings and source-internal spacing. This identifies the root family without positing an undocumented Sanskrit *jūrṇala or merging it with i-vowel jīrṇa.'),
('10258',['rat/mouse'],['musā','muso','mus','musuri','musa'],'Turner mūṣa first branch explicitly supplies Nepali/Kumauni muso, Maithili mūs/mūsā/musᵃrī and Hindi mūs/mūsā. These Nepal survey names match that branch, including the attested diminutive musuri, rather than the distinct mūṣika branch.'),
('10286',['soil / clay','soil/clay'],['māṭi','māto'],'Turner mṛttikā explicitly gives Nepali/Kumauni māṭo and eastern māṭi earth/clay. These Nepal survey responses match the regional forms and sense; their inherited lexical-family comparison does not settle possible Nepali-mediated diffusion.'),
('12583',['horn','horn (of buffalo)'],['siŋ','sĩŋ'],'Turner śṛṅga explicitly gives Nepali/Bengali siṅ and the eastern s-initial nasal horn forms. These survey responses agree directly in shape and sense, retaining nasalization as printed.')]
for parent,glosses,forms,evidence in groups:
 for r in rs:
  if r['ID'] not in d and r['Gloss'] in glosses and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence))
for r in rs:
 if r['ID'] not in d and r['Gloss']=='name' and r['Form'] in ['naŋ','nɔmp','navɸ']:
  h.append(dict(record=r,reason='The nāman family is a strong lead, but the final ŋ/p/ɸ in this response needs a source or language-specific explanation rather than silent normalization.'))
(P/'pass104-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print(len(a),len(h))
