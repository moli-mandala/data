import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
ids={'f_nkwvytpswqyt4':('f_dmpttw4whm66w','derived'),'f_c6ywgr55dhw52':('f_kg2m4a365nzf2','derived')}
fs={r['ID']:r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in set(ids)|{x[0] for x in ids.values()}}
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
a=[]
for id,(parent,kind) in ids.items():
 assert id not in linked
 r=fs[id];assert 'rezaibaghbidi2003zargari[' in r['Source']
 e=('Printed p.129 section 2.3.27 explicitly segments the adjective with -ālo and distinguishes stem from affix. čhib-ālo cheeky is literally having a tongue; čik-ālo muddy is formed on čik mud. Link the adjective to its existing same-source nominal base; no reconstruction of the entire adjective as a direct Sanskrit reflex.' if kind=='derived' else 'Printed p.129 section 2.3.27 explicitly presents this as a vowel-harmonized variant of the corresponding adjective: čhib-ālo > čhub-ālo and čik-ālo > čuk-ālo/čuq-ālo. Preserve the source record and link it as variant to that existing adjective, whose derivation from the noun is recorded separately.')
 a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind=kind,citation=r['Source'],evidence=e))
(P/'pass033-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
(P/'pass033-review.md').write_text('''# Zargari possessive adjectives and vowel harmony

2 records / 2 new derived rows; three existing compiled variant links preserved. Standing autonomous authorization applies.

| # | Zargari | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **čhub-ālo** ‘cheeky’ | Derived from Zargari **čhib** ‘tongue’ (f_dmpttw4whm66w) | Primary grammar p.129 §2.3.27 explicitly segments stem and affix and glosses the adjective literally as ‘having a tongue’. The base’s accepted ancestry is CDIAL5228. |
| 2 | **čuk-ālo** ‘muddy’ | Derived from Zargari **čik** ‘mud’ (f_kg2m4a365nzf2) | The same passage explicitly distinguishes the nominal stem from -ālo. The independently attested base was linked to cikka4780 in pass032. |

Evidence is the existing primary grammar decoding, zargari-primary-pages.json, PDF p7 / printed p129. This batch records source-explicit synchronic structure; its nominal bases already have reviewed accepted historical ancestry. No new lexical nodes, transcription edits or rebuild.

The compiled graph already directs čhib-ālo to čhub-ālo and čik-ālo/čuq-ālo to čuk-ālo. Preserve that orientation; the two canonical harmonized adjectives receive derived links to their nouns. The primary passage explicitly explains their vowel harmony. Initial five-row proposal failed scratch validation because it reversed existing variant relations; no rows were saved from that proposal.

Validation: pending.
''')
print('Prepared',len(a))
