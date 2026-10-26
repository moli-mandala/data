import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
groups=[
dict(parent='5994-3',gloss=['three'],languages=['Vasavi','Noiri','PauriBareli','RathwiBareli'],forms=['ṭiṇ'],citation='CDIAL[5994,3]',evidence='Full CDIAL distinguishes masculine trayaḥ/trāyaḥ from neuter trīṇi. The Bhil survey ṭiṇ has the i vowel and final nasal of section 3, alongside Prakrit tiṇṇi, Gujarati traṇ and Marathi/Hindi tīn; it cannot be assigned to the a-vowel sections from the English numeral alone. Source retroflex onset and final ṇ are retained. The family link does not select a particular modern donor.'),
dict(parent='2830',gloss=['ear'],languages=['Vasavi','Noiri','PauriBareli','RathwiBareli'],forms=['kāŋ'],citation='CDIAL[2830]',evidence='CDIAL karṇa gives Prakrit kaṇṇa and Gujarati/Marathi/Hindi kān “ear”, with widespread nasal place variants across its reflexes. The selected neighbouring Bhil attestations retain the same kā- plus nasal body-part stem; their word-final ŋ is explicitly present in the original survey transcription, not silently normalized to n. This is a family assignment; the phonetic value/place of the final nasal remains as transcribed.'),
dict(parent='4147-3',gloss=['cow'],languages=['poth'],forms=['gā̃'],citation='CDIAL[4147,3]',evidence='Turner explicitly places Lahnda gā̃ (plural gãī) and Awankari gā̃ (plural gāī̃) under section 3 gāvī. The Pahari-Pothwari Lahndic survey has the same contracted nasalized cow form. This uses the same-region comparator, not nasalization alone: identical-looking Punjabi gā̃ is listed in section 2, so Gujari and Haryanvi homographs are not included.'),
dict(parent='13271',gloss=['snake'],languages=['Goj','awan'],forms=['sapʰ'],citation='CDIAL[13271]',evidence='CDIAL sarpa gives Prakrit sappa, Lahnda/Punjabi sapp and explicitly Awankari sap “snake”. The survey sapʰ matches that short-vowel, retained-stop family; final release aspiration is preserved as recorded and does not motivate a distinct etymon. These are simple responses, without the material of the feminine sarpiṇī or young-snake derivatives.'),
dict(parent='3164',gloss=['what'],languages=['Mai'],forms=['gī'],citation='CDIAL[3164]',evidence='Full kim explicitly records Maiya gī “what”, beside Gawri kī and Phalura ga. The selected Indus Kohistani/Maiya responses match that same-language comparator and require no added suffix or competing ka-stem analysis.'),
]
acc=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:
   acc.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
held=[]
for r in rs:
 reason=None;c=[]
 if r['Gloss']=='cow' and r['Form']=='gā̃' and r['Language_ID'] in ['Goj','kaithal']:
  c=['4147-2','4147-3'];reason='CDIAL gives Punjabi gā̃ under *gāvā (section 2) but Lahnda/Awankari gā̃ under gāvī (section 3). Gujari/Haryanvi singular gā̃ alone does not choose the branch; a local paradigm or explicit etymological comparator is needed.'
 if r['Gloss'] in ['path'] and r['Form'] in ['baṭ','bat','vāṭh','vāṭe']:
  c=['11366','11363'];reason='CDIAL vartman explicitly warns that some NIA feminine forms can equally derive from vartis. These bare path forms provide no paradigm or morphology to exclude the alternative; do not assign from the familiar Hindi/Marathi comparator alone.'
 if r['Gloss'] in ['where','where?'] and r['Form'] in ['kā̃','kahā','kahā̃','kāhā','kāhā̃','kahã']:
  c=['3384','3271'];reason='Homepage search and full kuha/kutaḥ articles do not establish the historical derivation of this particular kā̃/kahā̃ formation. Preserve the unresolved locative formation instead of treating initial k and a matching gloss as a direct reflex.'
 if reason:held.append(dict(record=r,candidates=c,reason=reason))
assert len(acc)==len({x['record']['ID'] for x in acc})
(P/'pass004-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass004-decisions.json').write_text(json.dumps({'accepted':acc,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(acc),[(g['parent'],sum(x['family']==i for x in acc)) for i,g in enumerate(groups)],'held',len(held))
