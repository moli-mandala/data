import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/das-yerava-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));seen={a['record']['ID'] for a in z['accepted']+z['held']}|{r['Form_ID'] for r in read_assignments()}
h={
'Kullu':('unresolved house noun','The source-specific Kullu house search returns only Das. Kodagu kūḷï is cooked rice (1911), and kuḷḷ- short-person comparisons have the wrong sense. Do not select those homophones; a same-lect house attestation is needed.'),
'Kodu':('unresolved firewood noun','Paniya Kodu firewood searches supply no independent firewood comparator; another Paniya compound merely contains koḍu. Kannada firewood results include cutting verbs and load descriptions, not a matching whole noun. Leave the wood stem unresolved.'),
'Chooriya':('sun loan route','Tamil cūriyaṇ/curiyan sun and Kannada sūrya sun are recorded in homepage discovery. The Indo-Aryan-looking solar word does not select a donor for Ravula Chooriya; Kannada, Malayalam and other local transmission cannot be distinguished from the source table.'),
'Chandra':('moon loan route','Kannada candra moon is attested in the existing surveys; the source Chandra has a familiar Indo-Aryan loan shape. Das supplies no immediate donor history, so do not assign Sanskrit as a remote substitute or infer Kannada solely from spelling.'),
'Boomi':('earth loan route','The Kannada dictionary attests bhūmi earth/ground/land (Uchida–Rajapurohit2018 p698 col2 entry1). This comparison supports investigation of the regional loan family, but popular Boomi does not distinguish local donor or prove aspiration loss; no accepted borrowing relation is specified.'),
'Pagava':('god loan route and formation','Kannada bhagavaṃta God is found by homepage search. Ravula Pagava may reflect this wider divine-title family, but its shorter ending and initial stop need source-local evidence; do not remove or supply material to force an exact donor.'),
'Thayya':('unresolved divine name','Tamil taiy and Malayalam teyyam searches provide no independent matching entry in the existing database. Possible comparison with a divine-name family must be demonstrated, not chosen from a familiar sound; preserve Thayya God without normalizing it to a kin term.'),
'Charaya':('arrack lexical family; donor unresolved','The Madras Tamil Lexicon entry cārāyam (reproduced at agarathi.com/word/சாராயம்) gives arrack and southern comparanda sā-rāya/cārāyam/sārāyi, with only cf.sāra. This identifies a lexical family but not an immediate Ravula donor. The original global Charaya search also returned unrelated Prakrit alkali/ashes; those are excluded.'),
'Kanjala':('marriage formation and loan route','Kannada kalyāṇa includes wedding/marriage; Tamil kannālam/kannāla searches give no result in the current database. The Kanjala nj formation is not explained by simply substituting kalyāṇa. The unrelated Indo-Aryan kañjala bird is excluded. Retain the full marriage word for further local phonological/contact evidence.'),
'Pechazhainju':('unresolved birth formation','DEDR4422 peṟ-/piṟ- beget/be born is relevant to birth, but Pechazhainju contains substantially more material than its printed simple stems. No full source-local segmentation has been demonstrated; do not drop the latter portion.'),
'Sinde':('unresolved cloth noun','Kannada sinde/cinde/cindi searches supply no independent cloth comparator. Popular Sinde may conceal distinctions not marked in this source; a failed exact search is not an etymology and unrelated similar strings are not assigned.'),
'Ulesu':('unresolved sitting formation','Global Ulesu search returns only Das. Malayalam sitting searches give irikkuka and other stems, none establishing the full Ulesu response. Resolve local sound and morphology before assigning a sitting family.'),
'Ilecha':('unresolved sitting formation','Global Ilecha search returns only Das. The source labels it Sit, so the superficially similar standing forms ile/ilcas cannot override the source sense. A same-lect sitting or stance attestation is needed.')}
for r in json.load(open(p/'survey-inventory.json')):
 if r['ID'] in seen:continue
 if r['Form']=='Puranda':
  z['accepted'].append(dict(record=r,parent='d4422',kind='reflex',citation='dedr[entry 4422];'+r['Source'],evidence='DEDR4422 explicitly gives Tamil piṟa (-nt-) be born and piṟantai birth, alongside Malayalam piṟakka/piṟappu. Ravula Puranda Birth is tentatively compared with the nasal-stop birth formation, preserving the source vowel and voicing spellings. The popular transcription does not demonstrate the initial-vowel correspondence or a precise grammatical form; this is a qualified family proposal, not a source normalization.',status='pending-review'))
 elif r['Form'] in h:
  c,reason=h[r['Form']];z['held'].append(dict(record=r,candidate=c,reason=reason))
 else:raise AssertionError(r)
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
used={d for a in z['accepted'] for d in a.get('components',[a['parent']])};a={r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in used and r['Status']!='duplicate-excluded'};(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
