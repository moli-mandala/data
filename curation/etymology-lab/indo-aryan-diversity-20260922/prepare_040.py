import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1');rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'pattanaik-koul2003varli[' in r['Source'] and r['ID'] not in linked]
a=[];held=[]
for r in rs:
 rec={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']}
 if (r['Form'],r['Gloss']) in [('piva','drink'),('pi','drink'),('kha','eat')]:
  parent='8209' if r['Gloss']=='drink' else '3865'
  ev=('Turner8209 gives Pali pivati, Prakrit pi(v)ai and western continuations Gujarati pīvũ, Marathi piṇẽ. The survey piva/pi contrast belongs to this retained-v versus contracted drinking stem series; retain source vowel lengths as printed.' if parent=='8209' else 'Turner3865 section1 gives the khā- eating stem with Gujarati khāvũ, Marathi khāṇẽ and Konkani khāuṁk. The unextended source kha eat is assigned here; no causative, passive or past-participle branch is inferred.')
  a.append(dict(record=rec,parent=parent,kind='reflex',citation=r['Source']+f';CDIAL[{parent} and addenda]',evidence=ev))
 if r['Form'] in ['cal','cala','khat','rɔgɔt','haṛke','hāṛuk','ṭɔkkɔl']:
  reasons={'cal':'Turner4715 calati and4721 calyati both have walking continuations; Gujarati/Marathi walk forms are mainly4721, but source short-a is nondiagnostic. Need lect-specific phonology/morphology before choosing.','cala':'Same calati4715 versus calyati4721 issue; no branch selected merely from one short-vowel spelling.','khat':'Eating root is clear, but final-t form needs source grammar to distinguish ordinary present/participial inflection from a historical past stem before choosing3865 versus3865-2.','rɔgɔt':'Checked full10539rakta,10778rudhira,11165lohita. None supplies the medial-g/t source shape; do not assert contamination or borrowing without evidence.','haṛke':'13952haḍḍa supports broad bone family; need account of final-ke and comparison with hāṛuk, rather than silently stripping suffix.','hāṛuk':'13952haḍḍa supports broad bone family; full addendum has hāḍgu and conjectural *hāḍku, but source-uk ordering still needs morphological corroboration.','ṭɔkkɔl':'5422 gives Marathi ṭakkal bald spot and related slang head terms; source head meaning plausible but exact ṭɔkkɔl semantic/phonological route not settled.'}
  held.append(dict(record=rec,reason=reasons[r['Form']]))
assert len(a)==3
(P/'pass040-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'pass040-review.md').write_text('''# Varli eating and drinking; unresolved-family follow-up

Three proposed reflex records under standing autonomous authorization.

| # | Varli survey | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **piva / pi** ‘drink’ (2 records) | **pibati**, CDIAL **8209** | Turner includes Pali pivati, Prakrit pi(v)ai, Gujarati pīvũ and Marathi piṇẽ. Retained-v and contracted stems fit these two survey forms; preserve printed vowel lengths. |
| 2 | **kha** ‘eat’ (1 record) | **khādati**, CDIAL **3865** | Turner section1 includes the unextended khā- series with Gujarati khāvũ, Marathi khāṇẽ and Konkani khāuṁk. No past or causative morphology is asserted. |

Held research is retained in decisions: walk cal/cala requires discrimination of calati4715 and calyati4721; khat requires analysis of final-t morphology. Full rakta10539, rudhira10778 and lohita11165 articles do not explain rɔgɔt blood. Bone haṛke/hāṛuk needs its suffixes explained; ṭɔkkɔl head has an expressive bald/head family comparison at5422 but remains unresolved.

Source audit: printed p85 items282–283, manual-page-image, no reading uncertainty. Davar/Bhili and Dungar/Varli classification unchanged. Research: varli-search011–013; full varli-primary009–011. No new nodes, source edits or rebuild.

Validation: pending.
''')
print(len(a),'accepted',len(held),'held')
