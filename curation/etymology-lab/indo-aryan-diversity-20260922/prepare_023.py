import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[{k:r[k] for k in ('ID','Language_ID','Form','Gloss','Source','Tags')} for r in csv.DictReader(open('cldf/forms.csv')) if 'pattanaik-koul2003varli[' in r['Source']]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('one',['yēk'],'2462-2','*ēkka','Turner section2 places Marathi/Gujarati ek and y-initial yek comparanda under *ēkka, with retained k; the simple ēka branch does not fit that consonant as directly.'),
('two',['dōn','dɔ̄n'],'6648','dva','Turner gives Marathi don and Konkani dōni, with MIA doṇṇi remodelled after tiṇṇi. The survey n-final forms match this western numeral series; their o/ɔ distinction is retained.'),
('three',['tin','tīn'],'5994-3','trīṇi','Turner section3 explicitly gives Marathi tīn and Konkani tīni from MIA tiṇṇi; the t-n forms select trīṇi rather than the trayaḥ or trāyaḥ branches.'),
('four',['cār'],'4655-2','catvāri','The full article section2 gives Gujarati cār and Marathi cyār/cār via MIA *cayāri. Use the specific catvāri subsection despite the parsed Marathi search witness pointing to the broad head.'),
('five',['pãc','pā̃c'],'7655','pañca','Turner gives Gujarati pā̃c and Marathi pā̃ċ beside Prakrit paṁca. The nasal vowel and final affricate support this family; the lects’ recorded length distinction is retained.'),
('six',['sɔ'],'12803-3','*kṣaṭ','Turner section3 groups Marathi sahā/sā and Konkani sa under *kṣaṭ or *kṣvaṭ. The survey sɔ is assigned to this regional plain-s series, retaining its rounded vowel; superficial vowel similarity to the northwestern ṣo series does not establish the separate *ṣuvaṭ branch. This is a regional family inference, not an exact Turner attestation of sɔ.'),
('seven',['sat'],'13139','sapta','Turner gives Gujarati/Marathi sāt and Konkani sāta beside MIA satta. The survey sat preserves the inherited s-t sequence; no unprinted vowel length is added.'),
('eight',['āṭh','aṭh'],'941','aṣṭā','Turner gives Gujarati/Marathi āṭh beside MIA aṭṭha. Both survey forms have the diagnostic retroflex aspirate; recorded length differences are preserved.'),
('nine',['nɔu','nɔv'],'6984','nava','Turner explicitly gives Marathi nav/naū and Gujarati/Konkani nav. The survey u/v variants align with this nine family; both separately installed Davar responses remain separate records.'),
('ten',['dɔs','dɔha'],'6227','daśa','Turner gives Marathi das/dahā, Gujarati das, and MIA dasa/daha. The survey s/h lect contrast falls within this family; the source rounded vowels and final a remain unchanged.'),
('twenty',['vis','vīs'],'11616','viṁśati','Turner gives Gujarati/Marathi vīs and Konkani vīsa from the MIA vīsa series. The dictionary gloss is numeric 20, which explains its absence from a narrow twenty-gloss search.')]
a=[]
for g,fs,pa,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Gloss']==g and r['Form'] in fs:
   a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa} and addenda];'+r['Source'],evidence=ev+' Existing manual-page-image source audit has no reading uncertainty. Davar/Bhili and Dungar/Varli labels are preserved; no specific recent donor is established.'))
assert len(a)==23,len(a)
(P/'pass023-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
lines=['# Varli survey: numerals','',f'{len(a)} records / {len(a)} reflex rows across eleven etyma. Autonomous research and save authorized by the active objective. Existing accepted rows and compiled rank-1 links excluded.','','| # | Varli survey | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,pa,label,ev) in enumerate(groups,1):
 label=label.replace('*',r'\*');n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({n}) | **{label}**, CDIAL **{pa}** | {ev} |')
lines+=['','Source audit: items202–210 on printed p.83; items211 and213 on p.84. All selected responses have existing manual-page-image review and empty uncertainty. Comparison columns marked scope-only are excluded. Source transcription, lect mapping and the split Davar nine responses are preserved. No new donor nodes, ingestion, identity edits or rebuild. Earlier Varli unresolved cases remain held.','','Research files: varli-search003–006.json; varli-primary003–005.json (full shortlisted entries and addenda); pass023-source-audit.json.','','Validation and save status: pending.']
(P/'pass023-review.md').write_text('\n'.join(lines)+'\n')
print(len(a),'accepted',dict(__import__('collections').Counter(x['record']['Language_ID'] for x in a)))
