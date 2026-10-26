import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'regmi-thakur2016danuwar[' in r['Source'] and r['ID'] not in linked]
groups=[(['fire'],['āgi'],'55','agni','Turner55 explicitly gives Bihari āgi, Maithili āgⁱ and Bhojpuri āgī fire through the agni/aggi family. All five survey āgi attestations fit this regional inherited outcome; the source final i is retained. Separately numbered addendum14198 concerns Bengali aggini-type extensions and does not require a different branch here.'),(['water','rain'],['pāni'],'8082','pānīya','Turner8082 gives Nepali pāni and Maithili/Bhojpuri pānī water, and its addendum explicitly documents water/rain polysemy in a regional pāṇi form. Preserve five water responses and two rain responses as distinct source records; their shared form does not erase the elicited rain sense.'),(['star'],['tārā','t̺ārā','tarā'],'5798-4','tāraka','Turner5798 section4 contains Nepali tāro and Maithili/Hindi tārā star, with eastern tāra/tarā forms. This is the regional tāraka branch, rather than the feminine tārā head or tārikā subsection. Source unmarked length and dental diacritic differences remain unchanged; no gender annotation is added.'),(['sand'],['bālvā','bāluvā'],'11580','vālukā','Turner11580 gives Prakrit vāluā and Nepali bāluwā/baluwā, with Kumaoni bāluwā, under vālukā. These w/v-bearing forms support the full -ukā family. The source bālvā is a contracted member of that sequence, not reassigned solely by resemblance to a shorter bālu word.'),(['cloud'],['bādal','bād̺al'],'11567','vārdala','Turner11567 gives Prakrit vaddala cloud, Nepali bādal, and eastern bādar beside Hindi bādal. Four bādal source responses fit that family, retaining the dental articulation mark where supplied. The separate megh response is excluded; it is not a variant spelling of bādal.')]
a=[];lines=['# Danuwar: nature vocabulary across five survey locations','','Twenty-six proposed reflex records across five families; standing autonomous authorization.','','| # | Danuwar forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(gs,fs,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in fs and r['Gloss'] in gs];assert hits,(fs,gs)
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]'+(';CDIAL[14198]' if parent=='55' else ''),evidence=ev+' Existing broad Danuwar language/dialect mapping uncertainty and all source locators are retained.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{" / ".join(gs)}’ ({len(hits)} records) | **{label}**, CDIAL **{parent}** | {ev} |')
assert len(a)==26,len(a)
h=[dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason='Separate megh cloud record requires its own history and inherited versus learned/contact distinction; not grouped with bādal merely because the elicited gloss matches.') for r in rs if r['Form']=='megh' and r['Gloss']=='cloud']
audit={}
for line in open('data/other/forms/raw_data/census_nepal_2026/danuwar-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('installed',[])
  if len(ins)>7:audit.setdefault(ins[7],[]).extend(rd.get('review',[]))
for x in a:
 notes=audit[x['record']['Source']];assert notes and all(n.startswith('dialect mapping:') for n in notes),(x['record'],notes)
(P/'pass051-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),onlyDialectMappingUncertainty=True,sources={x['record']['Source']:audit[x['record']['Source']] for x in a}),ensure_ascii=False,indent=2)+'\n')
(P/'pass051-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines+=['','26 records and 26 reflex rows; one megh cloud record held. Each selected source audit reports dialect mapping uncertainty only. The five survey locations and all source-defined duplicates are preserved.','Research: homepage danuwar-search003; full primary004 and separately numbered addendum14198 in primary005. Source p117 items44,45,46,48,54,55. No new nodes, source repairs or rebuild.','Validation: pending.'];(P/'pass051-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
