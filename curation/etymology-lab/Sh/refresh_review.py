"""Render research manifests; never writes accepted overlay or compiled data."""
import json,re,pickle
from pathlib import Path
# Preserve the approved review snapshot when no new batch awaits review.
_review_dir = Path(__file__).resolve().parent
_review_batches = list(_review_dir.glob('batch-*.json'))
if _review_batches and all(json.loads(f.read_text()).get('status') == 'saved' for f in _review_batches):
    print('All batches saved; preserving the approved review snapshot.')
    raise SystemExit(0)
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[2]
progress=json.loads((ROOT/'overnight-progress.json').read_text())
proposals=[]
for path in sorted(ROOT.glob('batch-*.json')):
    batch=json.loads(path.read_text())
    if batch['status']!='saved': proposals.extend(batch['proposals'])
assert len({p['number'] for p in proposals})==len(proposals)
assert len({i for p in proposals for i in p['formIds']})==sum(len(p['formIds']) for p in proposals)
pages={}
for n,page in enumerate(pickle.load(open(DATA/'data/cdial/cdial.pickle','rb')),1):
    for number in re.findall('<number>([^<]+)</number>',page):pages.setdefault(number,n)
def cell(s):return s.replace('|','\\|').replace('\n',' ')
lines=['# Shina and Brokskat: overnight review' ,'','Deadline: September 10, 2026, 09:00 Eastern. Queue: Gilgit → Dras → Brokskat → all remaining Shina.','','Approved batch 1: 50 analyses, 70 accepted overlay rows. Every analysis below remains pending user review.']
if progress.get('status')=='ready_for_user_review':
    lines+=['', 'Overnight pass finalized for user review. The scheduled continuation is paused; unresolved inventories remain open.']
queues=[('Gilgit','', '#straightforward'),('Dras','dras','dras/REVIEW.md'),('Brokskat','../bro','../bro/REVIEW.md'),('All remaining Shina','remaining','remaining/REVIEW.md')]
overview=[];all_pending=[]
for name,folder,link in queues:
    items=[q for path in sorted((ROOT/folder).glob('batch-*.json')) for batch in [json.loads(path.read_text())] if batch['status']!='saved' for q in batch['proposals']]
    all_pending.extend(items)
    if name=='Gilgit':link=f"#straightforward-{sum(q['difficulty']=='straightforward' for q in items)}-proposals"
    overview.append(f"| [{name}]({link}) | {len(items)} | {len({i for q in items for i in q['formIds']})} | "+' / '.join(str(sum(q['difficulty']==level for q in items)) for level in ['straightforward','moderate','difficult'])+' |')
lines+=['','## Review overview','',f"{len(all_pending)} pending proposals cover {len({i for q in all_pending for i in q['formIds']})} records. Each queue is ordered by difficulty; cases without proposed assignments appear in separate follow-up tables.",'','| Queue | Proposals | Records | Straightforward / moderate / difficult |','|---|---|---|---|']+overview
lines+=['','The remaining-Shina queue includes additional Gilgit and Dras attestations; these review queues do not overlap. Brokskat still has unresolved cases. The full unresolved inventory and validation limitations appear below.']
coverage_path=ROOT/'handoff-coverage.json'
if coverage_path.exists():
    coverage=json.loads(coverage_path.read_text())
    lines+=['',f"Across the full inventory, {coverage['unresolvedOrUnexamined']} records remain unresolved or unexamined: {coverage['withDocumentedFollowup']} have explicit follow-up notes and {coverage['withoutIndividualFollowup']} have no individual follow-up in those manifests. Follow-up notes range from discovery leads to checked but inconclusive source comparisons; they are not completed etymologies.", '', '[Structural audit](review-consistency-audit.json) · [Exact follow-up coverage](handoff-coverage.json)']
followup=ROOT/'approved-followup.json'
if followup.exists():
    cases=json.loads(followup.read_text())['cases']
    lines+=['','## Approved analyses requiring another look','','These published disagreements were found after approval. The saved overlay is unchanged.','','| # | Gilgit Shina | Existing analysis | New evidence and review question |','|---|---|---|---|']
    for q in cases:
        lines.append('| '+str(q['proposalNumber'])+' | **'+cell(', '.join(q['forms']))+"** ‘"+cell(q['gloss'])+"’ | CDIAL "+q['savedParent']+' | '+cell(q['evidence']+' '+q['action']+' '+q['sourceLocator'])+' |')
for level in ['straightforward','moderate','difficult']:
    subset=[p for p in proposals if p['difficulty']==level]
    lines+=['','## '+level.capitalize()+f' ({len(subset)} proposals)','','| # | Gilgit Shina | Proposed etymology | Evidence |','|---|---|---|---|']
    for p in subset:
        kind=p['kind'];parent=p['parents'][0];label='**'+p['parentLabels'][0].replace('*','\\*')+'**'
        if kind=='component':
            label='Shina components: '+' + '.join('**'+x.replace('*','\\*')+'**' for x in p['parentLabels'])
            if p.get('compositionOperation'):label+=' ('+p['compositionOperation']+')'
        elif kind=='derived': label='Derived from Shina '+label
        elif kind=='borrowed':
            donor='Domaaki' if parent=='f_mwofwwzjyqtsi' else 'Hindi-Urdu' if parent=='f_cu7ekiqv7ojh6' else 'the cited donor'
            label=('Probably borrowed from '+label) if p.get('borrowingCertainty')=='probable' else 'Borrowed from '+donor+' '+label
        if p.get('derivationType'):label+=' ('+p['derivationType']+')'
        source=p['evidenceSource'];url=p.get('primarySourceUrl')
        match=re.search(r'CDIAL\[(\d+)',source)
        if not url and match and match[1] in pages:url='https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page='+str(pages[match[1]])
        label+='; '+('['+source+']('+url+')' if url else source)
        lines.append('| '+str(p['number'])+' | **'+cell(', '.join(p['forms']))+"** ‘"+cell(p['gloss'])+"’ | "+cell(label)+' | '+cell(p['evidence'])+' |')
records=len({i for p in proposals for i in p['formIds']});kinds={k:sum(len(p['assignments']) for p in proposals if p['kind']==k) for k in ['reflex','derived','borrowed','component']}
lines+=['','## Coverage and limitations','',f'{len(proposals)} pending proposals affect {records} records. Assignment rows by kind: '+', '.join(f'{k}: {v}' for k,v in kinds.items())+'.',f"{progress['fullQueueCounts']['Gilgit Shina']['unresolved_or_unexamined']} records across all Gilgit tags remain unresolved or unexamined. The separate loan-source audit preserves {progress.get('sourceAuditedLoanRecords',0)} source claims; some now have pending proposals elsewhere, so that audit is not an additional unresolved count.",'','See [loan source audit](LOAN-REVIEW.md) for the outstanding borrowed-word cases and [difficult citation review](DIFFICULT-REVIEW.md) for primary-checked ambiguous references. Those comparisons have no proposed assignment rows and require follow-up.','','Other unresolved leads include goat (aja/avi), earth (very doubtful sumahant), flour/griddle (unresolved immediate Indic donor), sit (upaviśati/vasati convergence), willow (*veti addendum), and complex elicitation responses.','','Pending assignment validation passes with this task’s approved dependencies. Global overlay validation encounters an unrelated pre-existing missing parent f_asyswz7gcij24. The focused suite had 16 passes and one pre-existing count failure: 2603 self-reference rows versus 2604 expected. Shared compiled CLDF, identities, and the accepted overlay have not been altered by pending research.']
if progress.get('fullQueueCounts'):
    lines += ['', '## Full queue inventory', '', 'The original Gilgit review selection is narrower than all Gilgit-labelled records. The counts below include additional source-specific dialect tags and are reconciled with the accepted overlay; unresolved does not mean fully examined.', '', '| Scope | Accepted ancestry | Pending records | Unresolved or unexamined |', '|---|---|---|---|']
    for scope in ['Gilgit Shina','Dras Shina','Brokskat','Remaining Shina']:
        c=progress['fullQueueCounts'][scope]
        lines.append(f"| {scope} | {c.get('accepted_link',0)} | {c.get('pending_proposal',0)} | {c.get('unresolved_or_unexamined',0)} |")
    lines += ['', 'Exact record IDs and source tags are preserved in [full queue inventory](full-queue-inventory.json). See the separate [Dras review](dras/REVIEW.md) for its numbered proposals.', '', 'Numeral construction reference: [Shina coordination and score counting](https://koshur.org/Linguistic/2.html). Compound proposals link the attested local stems and coordinating particle; inherited numerals retain their historical etyma.']
lines += ['', '[Brokskat difficulty-triaged review](../bro/REVIEW.md)', '', '[All remaining Shina: difficulty-triaged review](remaining/REVIEW.md)']
(ROOT/'REVIEW.md').write_text('\n'.join(lines)+'\n')
loanpath=ROOT/'loan-source-audit.json'
if loanpath.exists():
    records=json.loads(loanpath.read_text())['records'];loans=['# Gilgit Shina: loan-source audit awaiting follow-up','','These are source claims, not verified new graph assignments. The cited note was reviewed; immediate donor nodes, route and sound/meaning compatibility still need resolution. No accepted assignments are proposed here.','','| Record | Gilgit Shina | Source claim | Remaining question |','|---|---|---|---|']
    for p in records:loans.append('| '+p['Form_ID']+' | **'+cell(p['Form'])+"** ‘"+cell(p['Gloss'])+"’ | "+cell(p['primarySourceClaim']+' — '+p['Source'])+' | '+cell(p['analysis'])+' |')
    (ROOT/'LOAN-REVIEW.md').write_text('\n'.join(loans)+'\n')
print(f'Rendered {len(proposals)} proposals / {records if isinstance(records,int) else len(records)} loan audit records.')
