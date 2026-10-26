import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
holds={
'sīge':(['d3268'],'DEDR3268(a) supplies Kannada sī/sīyi/sihi and Tulu śī sweet, but Belari sīge has an unexplained -ge. Full article, including honey subsection, read; do not infer a component or donor from the short shared initial alone.'),
'haḍi':(['d3821'],'DEDR3821 gives Tulu paji and wider pacc-/pas- green forms. Initial p:h is supported in Belari, but the survey retroflex ḍ versus Tulu j still needs an attested correspondence or verified source reading. Preserve haḍi; do not silently rewrite it haji.'),
'kuṇṭra':(['d1670','d1851'],'DEDR1670 has kuṇṭe/kuṇṭu short/dwarf while DEDR1851 has Tulu kundri little/small/short. The Belari ṇṭr sequence does not choose cleanly between these families. Full articles read; retain both and resolve morphology/correspondences before assignment.'),
'gantadɨ':([],'Tulu good search offers eḍḍe, sajji, nala and others; relaxed gant search has not established a matching good adjective. Full gantadɨ, including final -adɨ, remains unexplained. No unsupported segmentation or donor claim.'),
'hōṇe':([],'Relaxed form discovery returns son-in-law, responsibility, and small-pond entries whose modifiers are not evidence for the Belari adjective small. Neither the long vowel nor the complete form has a supported small-family comparison yet.'),
'dabra':([],'Tulu rough gloss search and relaxed dabra form search have not established a supported full-form comparison. Retain as unresolved; possible loans require evidence for an immediate donor rather than a guessed distant root.'),
'moḷesɨ':([],'Installed gloss is cold. Bhat1971 p.120 web-extracted text reads chilly, whose intended sense and transcription need visual verification; scan access has failed. Tulu cold search does not establish this form. Do not reinterpret it as a chilli/pepper word or change the installed gloss without primary evidence.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form']=='seypɨ':
  z['accepted'].append(dict(record=r,parent='d2747',kind='reflex',citation='dedr[entry 2747];'+r['Source'],evidence='DEDR2747 gives Kannada sey beauty/merit/propriety and saypu rectitude/propriety/merit. Belari seypɨ beautiful is a qualified comparison with this s-y-p formation: the article supplies both the beauty sense and the -p formation, although not the exact Belari adjective. Preserve e and final ɨ; the whole article, not just its displayed head, is proposed. Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
