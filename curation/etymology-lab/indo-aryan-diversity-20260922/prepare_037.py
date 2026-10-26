import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'domari-drink-shortlist.json').read_text());linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
a=[];held=[]
for r in rs:
 if r['ID'] in linked:continue
 rec={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']}
 if r['Form'].startswith(('pī-r','pīr')):
  held.append(dict(record=rec,reason='Herin analyses perfective -r after ī synchronically, but Turner8229 specifically gives Palestinian Domari pīra wine under pīta drunk. Need further historical discrimination before assigning Aleppo pīr- perfective to either pibati8209 or pīta8229; the noun comparison alone does not settle the verb.'))
 elif r['Form'].startswith(('lə-','nə-')):
  held.append(dict(record=rec,reason='Prefixed modal/negative or progressive form excluded pending analysis of the full construction.'))
 else:
  assert r['Form']=='pī-' or r['Form'].startswith('py'),r['Form']
  a.append(dict(record=rec,parent='8209',kind='reflex',citation=r['Source']+';CDIAL[8209];herin2012domari[section 3.1, Tables 8–9]',evidence='Turner8209 explicitly gives Palestinian Domari pī- under pibati drinks. Herin Tables8–9 analyses the Aleppo imperfective and subjunctive as the same lexical pī-/py- base plus person endings and contextualizing -e. Section3.4 explicitly glosses pyin-e drink.IMPFV.1PL-CM and section3.6 pyind drink.SUBJ.3PL. These ordinary inflections retain one lexical ancestry. Perfective pīr- records are excluded pending discrimination from pīta8229; no claim about their historical stem is made here.'))
assert len(a)==15 and len(held)==11,(len(a),len(held))
(P/'pass037-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'pass037-review.md').write_text('''# Aleppo Domari drink: present and subjunctive

15 proposed records / 15 reflex rows; standing autonomous authorization.

| # | Domari | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **pī-** ‘drink’ and **py-** imperfective/subjunctive paradigm (13 records) | **pibati**, CDIAL **8209** | Turner explicitly gives Palestinian Domari pī-. Herin Tables8–9 shows six imperfective and six subjunctive person forms built on pī-/py-, including plural imperative py-əs; these are ordinary inflections, not separate derivations. |
| 2 | **pyin-e / pyind** ‘drink’ (2 records) | **pibati**, CDIAL **8209** | Herin §3.4 glosses pyin-e as drink.IMPFV.1PL-CM; §3.6 glosses pyind as drink.SUBJ.3PL. Neither is an infinitive simply because the extracted English gloss says ‘to drink’. |

Eleven records held: nine perfective pīr- forms and two prefixed constructions. Herin’s synchronic perfective -r after ī is clear, but CDIAL8229 specifically compares Palestinian Domari pīra wine with pīta drunk. That lexical noun comparison does not prove the Aleppo perfective’s origin; do not decide between8209 and8229 merely from the productive-looking surface segmentation. The two prefixed modal/negative-progressive forms also require separate construction analysis.

Evidence: homepage searches004–005, full CDIAL articles/addenda in domari-primary004–005.json, primary grammar domari-primary-prose.txt §3.1,§3.4,§3.6. No new nodes or rebuild.

Validation: pending.
''')
print(len(a),'accepted',len(held),'held')
