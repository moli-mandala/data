import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'chalise2014majhi[' in r['Source'] and r['ID'] not in linked]
groups=[('fire',['āgi'],'55','agni','Turner55 gives Bihari āgi, Maithili āgⁱ and Bhojpuri āgī fire through the agni/aggi family. The five source forms match that regional outcome, retaining final i; addendum14198 on Bengali aggini-type forms does not require another branch.'),('cloud',['bād̪al'],'11567','vārdala','Turner11567 gives Prakrit vaddala cloud and Nepali bādal, with eastern bādar and Hindi bādal. Majhi bād̪al fits this family; its marked dental d is preserved.'),('star',['t̪ārā'],'5798-4','tāraka','Turner5798 section4 contains Nepali tāro and Maithili/Hindi tārā star. Majhi t̪ārā matches that regional branch; no gender is added and the source dental notation remains unchanged.'),('sand',['bālvā','bāluvā','balvā'],'11580','vālukā','Turner11580 gives Nepali bāluwā/baluwā and Kumaoni bāluwā through Prakrit vāluā. The source -lvā/-luvā variants fit this family; initial length differences are retained. The separate -autā responses are not stripped of their endings.'),('body',['dziu'],'5239','jīva','Turner5239 gives Nepali jiu body/life and Maithili jiu life/soul. Majhi dziu is the source’s voiced-affricate spelling of the corresponding contracted body word; preserve dz rather than rewriting it as j.'),('tongue',['dzibro','dzibi'],'5228','jihvā','Turner5228 gives Nepali jib(h)ro, Kumaoni jibṛo and Bhojpuri jībhⁱ tongue, alongside simpler jib forms. These support the r-extended dzibro and final-i dzibi responses within the same family; source affricate spelling, vowels and r are preserved.'),('bone',['haddi','hād'],'13952','haḍḍa','Turner13952 gives regional haḍḍī bone and long-vowel hāḍ/hāṛ forms. Source haddi and hād fit the corresponding geminate and singleton forms. Retain the source unmarked stops; Turner regards the deeper connection with asthi as doubtful. The hādek extension is excluded.'),('head',['kapāl'],'2744-4','*kappāla','Turner2744 section4 explicitly gives Nepali kapāl head and Bihari/Maithili kapār head/forehead. The two short-initial-a kapāl head responses fit this branch; long kāpāl and the hair-glossed kapāl record remain separate.')]
a=[];lines=['# Majhi: nature and body vocabulary','','Thirty-four proposed reflex records across eight families; standing autonomous authorization.','','| # | Majhi forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in fs and r['Gloss']==g];assert hits,(g,fs)
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]',evidence=ev+' Existing published-notation uncertainty and locality identities remain unchanged; no source transcription repair.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)} records) | **{label}**, CDIAL **{parent}** | {ev} |')
assert len(a)==34,len(a)
h=[]
for r in rs:
 reason=None
 if r['Form']=='balautā' and r['Gloss']=='sand':reason='The -autā formation needs source morphology before the whole response can be assigned to vālukā11580.'
 elif r['Form']=='hādek' and r['Gloss']=='bone':reason='Additional -ek requires source morphological analysis; do not silently strip it.'
 elif r['Form']=='kāpāl' and r['Gloss']=='head':reason='Distinguish long initial ā from the kappāla2744-4 branch and possible kāpāla3038 family.'
 elif r['Form']=='kapāl' and r['Gloss']=='hair':reason='Hair-glossed source response requires semantic/source review rather than copying the head assignment; no gloss repair performed.'
 if reason:h.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason=reason))
selected={x['record']['Source'] for x in a};aud={};full=[]
for line in open('data/other/forms/raw_data/selected_surveys_2026/majhi-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('row',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).extend(rd.get('review',[]))
  if len(ins)>7 and ins[7] in selected:full.append(row)
for x in a:
 notes=aud[x['record']['Source']];assert notes and all(n=='transcription: published notation retained; ambiguous symbols are not assigned guessed phonemic values' for n in notes),(x['record'],notes)
(P/'pass054-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),retainedPublishedNotation=True,rows=full),ensure_ascii=False,indent=2)+'\n')
(P/'pass054-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines+=['',f'34 records and34 reflex rows; {len(h)} records held. Full source audit rows retained; all selected review notes concern published notation.','Research: homepage majhi-search004–005. Previously read complete shortlisted articles/addenda danuwar-primary002–005 are reused, with independently checked Majhi source records. No new nodes, source repairs or rebuild.','Validation: pending.'];(P/'pass054-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
