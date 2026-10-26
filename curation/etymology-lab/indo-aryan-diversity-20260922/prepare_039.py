import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'pattanaik-koul2003varli[' in r['Source']];linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['masle'],'fish','9758-3','*matsyal-','Turner9758 explicitly distinguishes the l-extension, with Marathi/Konkani māsḷī fish and Prakrit maścalī, from the bare matsya and matsiya branches. Davar masle retains that l-extension; the source final e and nonretroflex l are preserved. The existing promoted extension node is9758-3.'),
(['ḍoi'],'head','5566','*ḍōka','Turner5566 section1 explicitly gives Hindi/Marathi ḍoī head. Loss of medial k identifies this branch; do not conflate it with the retained-k head series.'),
(['ḍoka'],'head','5566-2','*ḍōkka','Turner5566 section2 gives Gujarati ḍokũ head and Marathi ḍokẽ under *ḍōkka. The survey retained k selects this subsection, unlike Davar ḍoi in section1.'),
(['dhōr'],'smoke','6852','*dhūmara','Turner6852 explicitly gives Marathi dhūr smoke/haze and Konkani dhuvōru smoke under *dhūmara. The survey r-final smoke word fits this extended family, not bare dhūma6849. Its o vowel is retained.'),
(['daṛhi','ḍaṛhi'],'beard','6250','*dāṁṣṭra','Turner6250 includes the specific beard development through Pali dāṭhikā/Prakrit dāḍhiā, with Hindi dāṛhī/ḍāṛhī and Marathi/Gujarati dāḍhī. Retain both survey initials; the article discusses uncertainty in the older fang–tusk–beard history and possible non-Indo-Aryan influence, which this family assignment does not settle.')]
a=[];lines=['# Varli survey: further nouns','','Six proposed records / six reflex rows across five exact parent nodes; standing autonomous authorization.','','| # | Varli survey | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,g,pa,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['ID'] not in linked and r['Form'] in fs and r['Gloss']==g];assert len(hits)==len(fs)
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=pa,kind='reflex',citation=f'CDIAL[{pa}];'+r['Source'],evidence=ev+' No immediate recent donor is established; existing source lect mapping and readings are retained.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ | **{label.replace("*",chr(92)+"*")}**, CDIAL **{pa}** | {ev} |')
held=[]
for r in rs:
 if r['ID'] in linked:continue
 if r['Gloss'] in ['blood','bone','tear'] or r['Form'] in ['masi','ṭɔkkɔl','dhukāv']:
  reason=('Source audit flags undefined capital L; leave transcription interpretation unresolved.' if r['Gloss']=='tear' else 'Broad family search is insufficient to select the exact historical stem/extension or explain the entire source form. Retain for further research; no assignment on gloss alone.')
  held.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason=reason))
assert len(a)==6
(P/'pass039-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
selected={('Davar','fish'),('Davar','head'),('Dungar','head'),('Dungar','smoke'),('Davar','beard'),('Dungar','beard')};aud=[json.loads(l) for l in open('data/other/forms/raw_data/20260911-dadra-varli-audit.jsonl')];aud=[r for r in aud if (r['lect'],r['gloss']) in selected and r['status']=='ingested'];assert len(aud)==6 and all(r['review']=='manual-page-image' and all(not x['uncertainty'] for x in r['forms']) for r in aud)
(P/'pass039-source-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2)+'\n')
lines+=['',f'{len(held)} candidates held: blood/bone, Dungar fish, Davar first head alternative and smoke need finer analysis; both tear records flag undefined capital L. Davar head audit row also contains the held ṭɔkkɔl alternative; only ḍoi accepted. All selected audit rows have manual-image review and no uncertainty.','Research: homepage search008–010; full primary007–008; accepted branch nodes checked in existing compiled forms. No rebuild or source edits.','Validation: pending.']
(P/'pass039-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
