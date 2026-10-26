"""Render review-only language batches with stable ownership; no graph writes."""
import json,csv,collections,re,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;LAB=P.parent;ROOT=P.parents[2]
data=json.load(open(P/'proposed-decisions.json'));rows=data['accepted'];held=data['held'];articles=json.load(open(P/'primary-evidence.json'))
langs={r['ID']:r['Name'] for r in csv.DictReader(open(ROOT/'cldf/languages.csv'))}
parents={}
ids={p for r in rows for p in r.get('components',[r['parent']])}
assignment_count=sum(len(r.get('components',[r['parent']])) for r in rows)
for r in csv.DictReader(open(ROOT/'cldf/forms.csv')):
 if r['ID'] in ids:parents[r['ID']]=r
by=collections.defaultdict(lambda:collections.defaultdict(list))
for x in rows:by[x['record']['Language_ID']][(x['parent'],x['evidence'])].append(x)
owned=json.load(open(P/'review-manifest-paths.json')) if (P/'review-manifest-paths.json').exists() else {}
covered={x['record']['ID'] for x in rows+held}|{r['ID'] for r in json.load(open(P/'externally-assigned-records.json'))};unexamined=[r for r in json.load(open(P/'survey-inventory.json')) if r['ID'] not in covered]
paths={};index=['# LSI Kurux residual records research — initial assessment complete','', '**Pending review. No new assignments have been saved. No rebuild has been run.**','',f'Current checkpoint: **{assignment_count:,} proposed assignment rows**, affecting **{len({x["record"]["ID"] for x in rows}):,} records**, under **{len(ids)} parent nodes** and **{len(by)} language**. Another **{len(held):,} records** have explicit research holds. These are research proposals, not accepted links.','', 'Proposals use full dictionary articles with explicit transcription, paradigm and multi-part-response qualifications. Conflicting meanings/language attributions and incomplete responses remain on hold. Existing source spellings, glosses, record identities and accepted sidecars are preserved.','', 'Across all 177 LSI Kurux records, 24 had earlier pending proposals and five had earlier research holds. This 148-record residual has 47 new proposals and 101 new holds, with zero unexamined records. The 12 go/went clauses have ordered pronoun and verb links; copula and beat clauses remain held because their predicates lack verified etyma.','', '| Language | Proposed records | Held records | Review batches |','|---|---:|---:|---|']
for lang,groups in sorted(by.items()):
 dest=LAB/lang;dest.mkdir(exist_ok=True)
 old=owned.get(lang,[]);grouped=list(groups.items());batchlinks=[];paths[lang]=[]
 existing=[int(f.stem.split('-')[1]) for f in dest.glob('batch-*.json') if f.stem.split('-')[1].isdigit()];n=max(existing,default=0)+1
 for j in range(0,len(grouped),5):
  at=j//5
  if at<len(old):
   manifest=Path(old[at]);olddata=json.load(open(manifest));assert olddata.get('researchDirectory')==str(P),manifest
   number=olddata['batch']
  else:manifest=dest/f'batch-{n:03d}.json';number=n;n+=1;assert not manifest.exists()
  props=[];lines=[f'**Next {len(grouped[j:j+5])} proposals, ready for review. Nothing in this batch saved yet.**','',f'{langs.get(lang,lang)} · LSI Kurux residual records, 2026-09-25 · batch {number:03d}. Pending user review; autonomous research continues.','',f'| # | {langs.get(lang,lang)} | Proposed etymology | Evidence |','|---|---|---|---|']
  for k,((parent,evidence),items) in enumerate(grouped[j:j+5],j+1):
   records=[x['record'] for x in items];forms=list(dict.fromkeys((r['Form'],r['Gloss']) for r in records))
   label=', '.join('**'+f.replace('|','\\|')+'** ‘'+g.replace('|','\\|')+'’' for f,g in forms)
   article=items[0].get('evidence_parent',parent)
   page=articles[article]['Page'];link=articles[article].get('URL',f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={page}');source_label=articles[article].get('Label',f'DEDR {article[1:]}')
   components=items[0].get('components')
   kind='component' if components else items[0].get('kind','reflex')
   direct=bool(components) or kind in {'borrowed','derived'} or all('comparison' in x for x in items)
   qualification='' if direct else ' This family identification does not endorse broader cross-family ancestry claims in the source.'
   full=evidence+qualification
   proposed=f'Proposed reflex of [**DEDR {article[1:]}**]({link})'
   if kind in {'borrowed','derived'}:
    relation='Borrowed from' if kind=='borrowed' else 'Derived from'
    donor=parents[parent]
    proposed=f'{relation} **{langs.get(donor["Language_ID"],donor["Language_ID"])} {donor["Form"]}**, [{source_label}]({link})'
   if components:
    proposed='Ordered components: '+' + '.join(f'[**DEDR {part[1:]}**](https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={articles[part]["Page"]})' for part in components)
   lines.append(f'| {k} | {label} | {proposed} | {full} ({len(items)} record{"s" if len(items)!=1 else ""}.) |')
   assignments=[dict(Form_ID=x['record']['ID'],Etymon_ID=part,Kind=kind,Rank='1',Status='proposed',Source=x['citation'],Notes=x['evidence'],Pos=str(pos) if components else '') for x in items for pos,part in enumerate(x.get('components',[parent]),1)]
   props.append(dict(number=k,status='pending-review',parentId=parent,parentForm=parents[parent]['Form'],kind=kind,citation=items[0]['citation'],evidence=full,formIds=sorted(r['ID'] for r in records),records=records,assignments=assignments))
   if components:props[-1].update(componentIds=components,componentForms=[parents[part]['Form'] for part in components])
  total=sum(len(x['records']) for x in props)
  edge_total=sum(len(x['assignments']) for x in props)
  kinds=', '.join(sorted({x['kind'] for x in props}))
  lines+=['',f'Proposed assignment rows: **{edge_total}**. Affected records: **{total}**. Relation kinds: **{kinds}**. Multi-part responses retain their source order.','',f'Exact persistent IDs and complete source records: [{manifest.name}]({manifest}).', '', 'Known ambiguous loans, questioned dictionary comparisons and unresolved parent labels are excluded; see the [research holds](../kurux-lsi-20260925/research-holds.json). No new lexical material or donor nodes are introduced.']
  manifest.write_text(json.dumps(dict(language=lang,batch=number,status='pending-review',date='2026-09-25',researchDirectory=str(P),proposals=props),ensure_ascii=False,indent=2)+'\n')
  review=manifest.with_name(manifest.stem+'-review.md');review.write_text('\n'.join(lines)+'\n');paths[lang].append(str(manifest));batchlinks.append(f'[{number:03d}]({review})')
 count=sum(len(x) for x in groups.values());hc=sum(x['record']['Language_ID']==lang for x in held)
 index.append(f'| {langs.get(lang,lang)} | {count} | {hc} | '+', '.join(batchlinks)+' |')
index+=['','Candidate discovery used the existing homepage `fetchLemmaList` code and read-only database. Full shortlisted DEDR articles were read from the cached primary-source audit; [primary evidence](primary-evidence.json), [decisions](proposed-decisions.json), [holds](research-holds.json), and [inventory](inventory-counts.json) are retained.','', ('This checkpoint is incomplete: unexamined records remain in the [research queue](unexamined-inventory.json). ' if unexamined else 'Initial assessment is complete: every source record has a proposal or research hold in this or the earlier checkpoint. Pending proposals and holds are not resolved or approved. ') + 'Research holds are distinguished from records not yet examined. No rebuild, browser database refresh, importer, commit, push or deployment has been performed.']
for old_paths in owned.values():
 for old_path in old_paths:
  if old_path in {v for vs in paths.values() for v in vs}:continue
  manifest=Path(old_path);previous=json.load(open(manifest))
  assert previous.get('researchDirectory')==str(P)
  previous['status']='superseded';previous['supersededReason']='Research regrouped after new primary evidence or current-sidecar reconciliation; consult current REVIEW.md.'
  manifest.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
  review=manifest.with_name(manifest.stem+'-review.md')
  review.write_text('**Superseded research checkpoint. Do not approve this old batch.**\n\nConsult the [current review index]('+str(P/'REVIEW.md')+').\n')
(P/'REVIEW.md').write_text('\n'.join(index)+'\n');(P/'review-manifest-paths.json').write_text(json.dumps(paths,indent=2)+'\n');(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
covered={x['record']['ID'] for x in rows+held}|{r['ID'] for r in json.load(open(P/'externally-assigned-records.json'))};unexamined=[r for r in json.load(open(P/'survey-inventory.json')) if r['ID'] not in covered];(P/'unexamined-inventory.json').write_text(json.dumps(unexamined,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(rows),'assignmentRows':assignment_count,'held':len(held),'unexamined':len(unexamined),'languages':len(by),'batches':sum(map(len,paths.values()))}))
