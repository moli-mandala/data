import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in json.loads((P/'tamilnadu-remaining055.json').read_text()) if r['ID'] not in linked]
groups=[(['sola'],'sixteen','12812','ṣoḍaśa','Turner12812 section1 gives Marathi/Konkani soḷā, Prakrit sōlā and eastern sola. Katkari sola fits the contracted -l- branch; source unmarked length and nonretroflex l are preserved. It is not the analogical dental-d or e-vowel subsection.'),(['bis','vīsu'],'twenty','11616','viṁśati','Turner11616 gives regional bis/bīs and Gujarati/Marathi vīs, Konkani vīsa, through MIA vīsa-type forms. Katkari bis and Saurashtra vīsu fit these b/v regional outcomes; the source final u remains unchanged. Separate addendum14802 concerns Pashai compounds and does not change this choice.'),(['calis','cəḷīsu'],'forty','4656','catvāriṁśat','Turner4656 gives Prakrit cālīsa, regional cālis/cālīs, Gujarati cāḷīś and Marathi ċāḷīs. These support the survey lateral variants; source c is retained without assigning it a new phonetic value. Turner marks the Marathi cā- variant as Hindi-influenced, so this family link does not establish an entirely contact-free history; the survey notation does not resolve that distinction.')]
a=[];lines=['# Tamil Nadu: higher numerals','','Five reflex records across three families; standing autonomous authorization. Printed pp. 870–871 checked in the preceding pass using the source PDF images.','','| # | Survey forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,g,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in fs and r['Gloss']==g]
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]',evidence=ev+' Printed source and ingestion audit checked. Existing transcription and chapter-based Katkari mapping uncertainty retained.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {ev} |')
assert len(a)==5
h=[dict(record=r,reason='Printed sɔvvəlu is confirmed, but vv plus l requires Saurashtra sound-history or analogical evidence before choosing a ṣoḍaśa branch.') for r in rs if r['Form']=='sɔvvəlu' and r['Gloss']=='sixteen']
aud=[];allowed={'dialect mapping: Katkari follows chapter p. 326 despite contradictory Dravidian table heading','gloss/transcription: published comparative table contains editorial inconsistencies; attested source pairing preserved','transcription: embedded image restored; original orthography retained'}
for line in open('data/other/forms/raw_data/census_nepal_2026/tamil-nadu-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('installed',[])
  if len(ins)>7 and any(x['record']['Source']==ins[7] and x['record']['Gloss']==ins[3] for x in a):
   assert all(n in allowed for n in rd['review']);aud.append(row)
assert len(aud)==5
for name,val in [('decisions',dict(accepted=a,held=h)),('source-audit',aud)]: (P/f'pass056-{name}.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
lines+=['','Held: **sɔvvəlu** ‘sixteen’ requires explanation of its vv/l consonants. Ordinal records remain unassigned pending morphology/contact research.','','Discovery: tamilnadu-search005–007. Full primary articles and separate addenda: tamilnadu-primary003–006. No source or identity repair; no rebuild.','','Save and validation pending.']
(P/'pass056-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
