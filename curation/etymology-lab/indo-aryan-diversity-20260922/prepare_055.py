import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in json.loads((P/'tamilnadu-remaining055.json').read_text()) if r['ID'] not in linked]
groups=[(['ek'],'one','2462-2','*ēkka','Turner2462 section2 explicitly gives Marathi and Gujarati ek, retaining k through MIA ekka. Katkari ek selects this branch; Turner allows emphatic doubling or a MIA Sanskrit loan as the origin of the reinforced stem.'),(['sat','sə̄tu'],'seven','13139','sapta','Turner13139 gives Marathi/Gujarati sāt and Konkani sāta through MIA satta. Katkari sat and Saurashtra sə̄tu fit the regional sat- family; the reported central long vowel and final u are retained, as in the survey’s other u-final numerals.'),(['ãṭ'],'eight','941','aṣṭā','Turner941 gives Marathi/Gujarati āṭh, Bengali āṭ and nasalised aṣ forms elsewhere in the family. Katkari ãṭ fits the eight family; its nasal vowel and unaspirated retroflex stop are preserved, without claiming that Turner prints this exact Katkari spelling.'),(['nɔo','nəvvu'],'nine','6984','nava²','Turner6984 gives Marathi nav/naū, Gujarati and Konkani nav, and regional nao/nɔu forms. Katkari nɔo matches the contracted type and Saurashtra nəvvu retains v; source gemination, central vowel and final u remain unchanged.'),(['dəs','dəssu'],'ten','6227','daśa','Turner6227 gives MIA dasa and Gujarati/Marathi das beside Marathi dahā. Both source forms fit the s-retaining ten family, with central vowels and Saurashtra ss/u preserved rather than normalised.')]
a=[];lines=['# Tamil Nadu: remaining basic numerals','','Eight reflex records across five families; standing autonomous authorization. Source images checked at printed pp. 868–871 (PDF pp. 888–891).','','| # | Survey forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,g,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in fs and r['Gloss']==g]
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]',evidence=ev+' Printed source image and ingestion audit checked. Existing transcription uncertainty and chapter-based Katkari mapping are preserved.'))
 label=label.replace('*',r'\*')
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {ev} |')
assert len(a)==8,len(a)
reasons={'ɔndɛ':'The printed one response is confirmed, but its contact history and immediate donor remain unresolved; do not assign it to ēkka.','aadu':'The printed eight response is confirmed. The aa/d notation and its relationship to aṣṭā require independent Saurashtra phonological evidence before linking.'}
h=[dict(record=r,reason=reasons[r['Form']]) for r in rs if r['Form'] in reasons and r['Gloss'] in ['one','eight']]
audit=[]
allowed={'dialect mapping: Katkari follows chapter p. 326 despite contradictory Dravidian table heading','gloss/transcription: published comparative table contains editorial inconsistencies; attested source pairing preserved','transcription: embedded image restored; original orthography retained'}
for line in open('data/other/forms/raw_data/census_nepal_2026/tamil-nadu-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('installed',[])
  if len(ins)>7 and any(x['record']['Source']==ins[7] and x['record']['Gloss']==ins[3] for x in a):
   assert all(n in allowed for n in rd['review']);audit.append(row)
assert len(audit)==8,len(audit)
(P/'pass055-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(P/'pass055-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines+=['','Two held records:']+[f'- **{x["record"]["Form"]}** ‘{x["record"]["Gloss"]}’: {x["reason"]}' for x in h]+['','Earlier diyɔ two and sɔ̄tu six holds remain. Discovery: tamilnadu-search003–004; full dictionary articles/addenda: danuwar-primary006–007. No source or language-identity repairs.','', 'Save and validation pending.']
(P/'pass055-review.md').write_text('\n'.join(lines)+'\n')
print(len(a),'accepted',len(h),'held')
