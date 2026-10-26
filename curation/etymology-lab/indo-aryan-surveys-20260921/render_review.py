"""Render this run's exact saved manifests and research checkpoint only."""
import json,collections,re,csv
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def esc(x):return str(x).replace('|',r'\|').replace('\n',' ').replace('*',r'\*')
def bold(x):return '**'+esc(x)+'**'
primary_links={}
for pf in sorted(P.glob('primary-[0-9][0-9][0-9].json')):
 for key,items in read(pf).items():
  if items: primary_links[key]=items[0]['url']
platts_links={}
lexical_links=read(P/'lexical-reference-links.json') if (P/'lexical-reference-links.json').exists() else {}
for pf in sorted(P.glob('platts-*.json')):
 for item in read(pf):
  match=re.search(r'p\. (\d+)',item.get('text',''))
  if match: platts_links[match.group(1)]=item['url']
def citation_html(c):
 def sub(m):
  key=m.group(1).split(',')[0].strip()
  return '[CDIAL '+esc(m.group(1))+']('+primary_links[key]+')' if key in primary_links else esc(m.group(0))
 c=re.sub(r'CDIAL\[([^]]+)\]',sub,c)
 def platts(m):
  page=re.search(r'p\. (\d+)',m.group(1))
  url=platts_links.get(page.group(1)) if page else lexical_links.get(m.group(0),lexical_links.get(m.group(1)))
  return '[Platts '+esc(m.group(1))+']('+url+')' if url else esc(m.group(0))
 c=re.sub(r'platts1884\[([^]]+)\]',platts,c)
 def lexical(m):
  url=lexical_links.get(m.group(0))
  return '['+esc(m.group(0))+']('+url+')' if url else esc(m.group(0))
 return re.sub(r'bashir2023\[([^]]+)\]',lexical,c)
corrections=[read(f) for f in sorted(P.glob('correction*-retraction.json'))]
withdrawn={i for c in corrections for i in c['formIds']}
def removed_for(ledger):
 return {i for c in corrections if Path(c['originalLedger']).name==Path(ledger).name for i in c['formIds']}
saved=[]; manifests=[]; held={}
for f in sorted(P.glob('pass*-decisions.json')):
 for x in read(f).get('held',[]):held[x['record']['ID']]=x
 for x in read(f).get('accepted',[]):
  if x['record']['ID'] in withdrawn:
   held[x['record']['ID']]={'record':x['record'],'reason':next(c['reason'] for c in corrections if x['record']['ID'] in c['formIds'])}
for f in sorted(P.glob('pass*-validation.json')):
 report=read(f);saved.append(report)
 removed=removed_for(f.name.replace('-validation','-saved-assignments'))
 for m in read(f.with_name(f.name.replace('-validation','-manifest-paths'))):
  mp=Path(m);d=read(mp);manifests.append(mp)
  removed_here={r['ID'] for q in d['proposals'] for r in q['records']} & removed
  for q in d['proposals']:
   q['records']=[r for r in q['records'] if r['ID'] not in removed]
   q['formIds']=[i for i in q['formIds'] if i not in removed]
   q['assignments']=[r for r in q['assignments'] if r['Form_ID'] not in removed]
  d['proposals']=[q for q in d['proposals'] if q['formIds']]
  lines=[f'# {d["language"]}: survey etymology review, batch {d["batch"]}','',f'**Saved under the autonomous-work authorization of 2026-09-21.** {len(d["proposals"])} proposals; exact IDs and source attestations are in the [manifest]({mp}).','', '| # | Language form and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
  if removed_here:
   lines[3:3]=[f'**Corrected after the original save:** {len(removed_here)} record(s) were withdrawn during self-review and are excluded below. The linked manifest preserves the original save; the correction ledger and current holds are in the [run review]({P}/REVIEW.md).','']
  for q in d['proposals']:
   forms=list(dict.fromkeys(bold(r['Form'])+' ‘'+esc(r['Gloss'])+'’' for r in q['records']))
   lines.append('| '+str(q['number'])+' | '+', '.join(forms)+' | '+{'reflex':'Reflex of ','borrowed':'Borrowed from ','derived':'Derived from ','component':'Component: '}.get(q['kind'],'')+(' + '.join(bold(x) for x in q['componentForms']) if q.get('componentForms') else bold(q['parentForm']))+'; '+citation_html(q['citation'])+' | '+esc(q['evidence'])+f' ({len(q["formIds"])} records.) |')
  rows=[a for q in d['proposals'] for a in q['assignments']]
  lines+=['',f'{len(rows)} assignment rows on {len({r["Form_ID"] for r in rows})} records. Relation kinds: '+str(dict(collections.Counter(r['Kind'] for r in rows)))+'.','',f'[Validation]({d["validation"]}). No compiled or browser rebuild. Excluded and competing analyses are tracked in the [run review]({P}/REVIEW.md).','']
  mp.with_name(mp.stem+'-review.md').write_text('\n'.join(lines))
rows=[r for f in P.glob('pass*-saved-assignments.json') for r in read(f) if r['Form_ID'] not in removed_for(f)]
ids={r['Form_ID'] for r in rows}; held={k:v for k,v in held.items() if k not in ids}
initial=read(P/'inventory-summary.json')
remaining_initial=read(P/'remaining-initial.json')
language_names={r['ID']:r['Name'] for r in csv.DictReader((P.parents[2]/'cldf/languages.csv').open())}
start_counts=collections.Counter(r['Language_ID'] for r in remaining_initial)
saved_counts=collections.Counter(r['Language_ID'] for r in remaining_initial if r['ID'] in ids)
held_counts=collections.Counter(x['record']['Language_ID'] for x in held.values())
language_manifests=collections.defaultdict(list)
for mp in manifests:language_manifests[mp.parent.name].append(mp)
index_dir=P/'language-index';index_dir.mkdir(exist_ok=True)
progress=['# Survey etymology progress by language','',f'{len(ids):,} newly linked records out of {initial["remaining_records"]:,} unassigned at the start. Counts concern source records, not unique words. Remaining includes held and unresearched records.','',f'[Complete run review]({P}/REVIEW.md)','', '| Language / survey | Initially unassigned | Newly linked | Explicit holds | Remaining |','|---|---:|---:|---:|---:|']
for lang,total in sorted(start_counts.items(),key=lambda x:(-saved_counts[x[0]],x[0])):
 label=language_names.get(lang,lang)
 destination=index_dir/(lang+'.md')
 progress.append(f'| [{esc(label)} ({lang})]({destination}) | {total:,} | {saved_counts[lang]:,} | {held_counts[lang]:,} | {total-saved_counts[lang]:,} |')
 body=[f'# {label} ({lang})','',f'{saved_counts[lang]:,} records newly linked; {total-saved_counts[lang]:,} remain from the {total:,} initially unassigned records. {held_counts[lang]:,} have explicit current-run holds.','',f'[Progress index]({P}/language-progress.md) · [Full run review]({P}/REVIEW.md)','', '## Saved reviews','']
 for mp in language_manifests[lang]:body.append(f'- [{mp.stem}]({mp.with_name(mp.stem+"-review.md")})')
 if not language_manifests[lang]:body.append('No new assignments saved for this language in this run.')
 body+=['','## Explicit holds','']
 for x in held.values():
  r=x['record']
  if r['Language_ID']==lang:body.append(f'- {bold(r["Form"])} ‘{esc(r["Gloss"])}’ (`{r["ID"]}`): '+esc(x['reason']))
 destination.write_text('\n'.join(body)+'\n')
(P/'language-progress.md').write_text('\n'.join(progress)+'\n')
lines=['# Remaining Indo-Aryan surveys: autonomous review, 21 September 2026','', '**Autonomous work resumed without a time limit.** The original eight-hour window and its completion audit are historical; the user subsequently authorized continued work. Pass117 and pass118 apply explicit adjudications. No rebuild, commit, push, or deployment.','',f'Starting reconciliation: {initial["current_scope_records"]:,} surviving records in the previous survey scope; {initial["remaining_records"]:,} lacked a current rank-1 graph or accepted sidecar assignment. The {len(initial["missing_old_ids"])} absent historical IDs are all retired and resolve through the identity registry: 72 map within that scope, and one additional current record is already assigned. The reconciled scope therefore has {initial["current_scope_records"]+1:,} current records, with the same unassigned starting count.','',f'Newly saved: **{len(rows):,} assignment rows on {len(ids):,} records**, in {len(saved)} passes. {len(held):,} records have explicit current-run holds. {initial["remaining_records"]-len(ids):,} starting records remain unassigned by this run; this includes previously held and still-unresearched material.','', '## Saved reviews','']
pending_files=[f for f in P.glob('pass*-decisions.json') if not f.with_name(f.name.replace('-decisions','-validation')).exists()]
pending_count=sum(len(read(f).get('accepted',[])) for f in pending_files)
fly_held=sum(1 for x in held.values() if x['reason'].startswith('User approved 1697-2'))
lines.insert(8,f'Pending research batches contain {pending_count} unsaved proposals; {fly_held} complex fly responses have an approved lexical parent but need complete morphology. The cold, path, seed and fire adjudications were applied in pass118: [review]({P}/adjudication-next.md).\n')
lines.insert(8,f'[Language-by-language progress and review index]({P}/language-progress.md)\n')
for mp in manifests:
 lines.append(f'- [{mp.parent.name}, {mp.stem}]({mp.with_name(mp.stem+"-review.md")})')
if corrections:
 lines+=['','## Corrections during self-review','', 'Original save ledgers and manifests remain as historical records. Current counts and review tables exclude the following withdrawn analyses.']
 for c in corrections:
  lines.append(f'- {len(c["removedAssignments"])} rows on {len(c["formIds"])} records withdrawn: '+c['reason'])
lines+=['','## Evidence and validation','', 'Candidate discovery invokes the actual homepage `fetchLemmaList` implementation against the existing read-only browser database. Saved assignment decisions use the current compiled forms/edges plus every current source sidecar, since the browser database lags them. Search inputs/results, full shortlisted CDIAL articles and addenda, exact decisions, and helper validation ledgers are retained in this directory.','', 'Each saved pass passed exact-node and durable-ID checks, temporary graph application, repeat-application idempotence, and unchanged compiled-data/identity hashes. The shared save helper routes the new rows to their source sidecars.','', 'The global `check-sources` run after pass001 reports two existing misplaced rows: `f_sntaxhwfhm3ew → 5071` and `f_2mljkhybpnnbm → 5232`, in the Bagheli sidecar instead of Bundeli. The Bagheli sidecar was unchanged by pass001. This is a validation limitation, not a clean global pass. Those failures persisted through pass115; check-sources after pass117 exited successfully. The intervening correction was outside this run. See the per-pass `check-sources-*.log` files. A separate `new-row-placement-audit.json` confirms the placement of this run’s rows.','', 'No new donor/source ingestion has been performed. Full builds and browser database refreshes are deliberately omitted by request. Ordinary inflections remain reflex links; complex responses are only accepted when their morphology and full lexical content are accounted for.','', 'The [SSNP font audit]('+str(P/'ssnp-tongue-transcription-audit.md')+') documents printed affricates currently imported as y; source spellings and profiles remain unchanged. Full primary dictionary extracts are in `primary-*.json` and are linked in each review table.', '', '## Current-run held cases','']
for i,x in enumerate(held.values(),1):
 r=x['record'];lines.append(f'{i}. {r["Language_ID"]} {bold(r["Form"])} ‘{esc(r["Gloss"])}’ (`{r["ID"]}`): '+esc(x['reason']))
(P/'REVIEW.md').write_text('\n'.join(lines)+'\n')
(P/'current-summary.json').write_text(json.dumps({'savedRows':len(rows),'savedRecords':len(ids),'passes':len(saved),'heldRecords':len(held),'remaining':initial['remaining_records']-len(ids)},indent=1)+'\n')
print('saved rows',len(rows),'saved records',len(ids),'held',len(held))
