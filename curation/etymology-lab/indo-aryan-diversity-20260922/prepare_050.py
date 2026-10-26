# Original13-record research generator; final decisions hold10851 hair after existing graph cycle found. Do not rerun over final decisions.
import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'regmi-thakur2016danuwar[' in r['Source'] and r['ID'] not in linked]
groups=[('body',['jiu'],'5239','jīva','Turner5239 gives Nepali jiu body/life and Maithili jiu life/soul in the jīva family. Both source body attestations fit the contracted regional form; the homonymous source tongue jiu is excluded.'),('head',['kapār'],'2744-4','*kappāla','Turner2744 section4 explicitly gives Bihari/Maithili kapār head/forehead and Awadhi kapāru head. Short initial a selects this existing branch; source kāpār with long initial ā remains separate pending discrimination from kāpāla3038.'),('hair',['raũ'],'10851','roman','Turner10851 explicitly gives Nepali raũ hair, with Maithili roā̃ and Bhojpuri rōwā̃ body hair. Preserve the survey’s general hair gloss and nasal vowel; do not silently narrow it to scalp or body hair.'),('bone',['hād̺','hār','had̺d̺i'],'13952','haḍḍa','Turner13952 gives Nepali/Maithili/Bhojpuri hāṛ bone and regional haḍḍī bone, including Hindi and northwestern comparanda. The long-vowel singleton and short-vowel geminate source forms belong to this bone family. Preserve the source dental diacritics and plain r; the entry’s proposed deeper connection with asthi is explicitly doubtful.'),('tongue',['jip','jibrā'],'5228','jihvā','Turner5228 gives regional jib/zip tongue and Nepali jib(h)ro beside Kumaoni jibṛo. These support the simple labial and r-extended source forms within the same tongue family. Preserve source final p and the unmarked dental/plain r; the separate jiu tongue response remains held.')]
a=[];lines=['# Danuwar: further body vocabulary','','Existing compiled and accepted sidecar links excluded; standing autonomous authorization.','','| # | Danuwar forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in fs and r['Gloss']==g];assert hits,(g,fs)
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]',evidence=ev+' Existing Done/East classification and broad Danuwar dialect-mapping uncertainty remain unchanged.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)} records) | **{label}**, CDIAL **{parent}** | {ev} |')
h=[]
for r in rs:
 reason=None
 if r['Form']=='chālā' and r['Gloss']=='skin':reason='Turner5005 explicitly marks Nepali chālā as a Hindi loan. Danuwar source forms require immediate-donor versus inherited-family discrimination; do not infer a route from the Nepali comparator alone.'
 elif r['Form']=='kāpār':reason='Long initial ā requires distinguishing kappāla2744-4 from kāpāla3038. The source hair-glossed kāpār additionally needs semantic/source review; no source correction performed.'
 elif r['Form']=='jiu' and r['Gloss']=='tongue':reason='Do not conflate tongue jiu with body jiu5239. A source-specific tongue development must be established before selecting5228.'
 if reason:h.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason=reason))
audit={}
for line in open('data/other/forms/raw_data/census_nepal_2026/danuwar-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('installed',[])
  if len(ins)>7:audit.setdefault(ins[7],[]).extend(rd.get('review',[]))
for x in a:
 notes=audit[x['record']['Source']];assert notes and all(n.startswith('dialect mapping:') for n in notes),(x['record'],notes)
(P/'pass050-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),onlyDialectMappingUncertainty=True,sources={x['record']['Source']:audit[x['record']['Source']] for x in a}),ensure_ascii=False,indent=2)+'\n')
(P/'pass050-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines+=['',f'{len(a)} proposed records and reflex rows; {len(h)} held. All selected source audit notes concern dialect mapping only; this uncertainty and all source spellings are preserved.','Research: homepage danuwar-search002; full dictionary articles/addenda danuwar-primary002. No new source material or source repairs.','Validation: pending.'];(P/'pass050-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
