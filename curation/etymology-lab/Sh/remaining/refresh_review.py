import json,pickle,re
from pathlib import Path
# Preserve the approved review snapshot when no new batch awaits review.
_review_dir = Path(__file__).resolve().parent
_review_batches = list(_review_dir.glob('batch-*.json'))
if _review_batches and all(json.loads(f.read_text()).get('status') == 'saved' for f in _review_batches):
    print('All batches saved; preserving the approved review snapshot.')
    raise SystemExit(0)
R=Path(__file__).resolve().parent;D=R.parents[3];pages={}
for n,page in enumerate(pickle.load(open(D/'data/cdial/cdial.pickle','rb')),1):
 for k in re.findall('<number>([^<]+)</number>',page):pages.setdefault(k,n)
p=json.loads((R/'progress.json').read_text());qs=[q for f in sorted(R.glob('batch-*.json')) for q in json.loads(f.read_text())['proposals']]
def esc(x):return x.replace('|','\\|').replace('\n',' ')
lines=['# Shina: pending overnight review','','All proposals remain pending user review. Numbering is local to the full-Shina remainder pass.']
for level in ['straightforward','moderate','difficult']:
 sub=[q for q in qs if q['difficulty']==level];lines+=['','## '+level.capitalize()+f' ({len(sub)})','','| # | Shina | Proposed etymology | Evidence |','|---|---|---|---|']
 for q in sub:
  label='**'+q['parentLabels'][0].replace('*','\\*')+'**'
  if q['kind']=='component':label='Shina components: '+' + '.join('**'+x+'**' for x in q['parentLabels'])+' ('+q['compositionOperation']+')'
  elif q['kind']=='derived':label='Derived from Shina '+label
  elif q['kind']=='borrowed':label='Borrowed from '+label
  match=re.search(r'CDIAL\[(\d+)',q['evidenceSource']);citation=q['evidenceSource']
  if match: citation='['+citation+'](https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page='+str(pages[match[1]])+')'
  lines.append('| '+str(q['number'])+' | **'+esc(', '.join(q['forms']))+'** ‘'+esc(q['gloss'])+'’ | '+label+'; '+citation+' | '+esc(q['evidence'])+' |')
lines+=['','## Coverage','',f"{len(qs)} pending proposals cover {len({i for q in qs for i in q['formIds']})} records with {sum(len(q['assignments']) for q in qs)} proposed assignment rows. {len(p['unexaminedIds'])} records remain unresolved or unexamined in the full Shina remainder.",'','Pending rows validate jointly with the earlier Shina proposals. No accepted overlay or compiled data changed. This pass includes records from all Shina dialects, including open Gilgit and Dras material.','', 'Anomalous Gurez kuṭu ear is held for source review; it was not assigned to karṇa on the basis of its English gloss.']
for filename,title in [('contact-followup.json','Contact cases'),('motion-followup.json','Motion and posture verbs'),('calendar-followup.json','Calendar terms'),('kinship-followup.json','Kinship source and etymology checks'),('landscape-followup.json','Landscape terms'),('additional-followup.json','Additional lexical checks')]:
 audit=R/filename
 if not audit.exists():continue
 data=json.loads(audit.read_text())
 cases=data['cases'];held_ids={i for case in cases for i in case['formIds']}
 lines+=['','## '+title+' held for research','',f'{len(cases)} cases cover {len(held_ids)} records. These cases have no proposed assignments. Difficulty describes the remaining review; some moderate cases have an explicit source analysis but await an eligible donor or base node.','', '| Difficulty | Shina | Records | Evidence and next check |','|---|---|---|---|']
 for case in sorted(cases,key=lambda c:['straightforward','moderate','difficult'].index(c['difficulty'])):
  evidence=esc(case['evidence'])
  source_url=case.get('sourceURL') or case.get('sourceUrl')
  if source_url:evidence+=' [Primary discussion]('+source_url+').'
  if case.get('sourceLocator'):evidence+=' Source: '+esc(case['sourceLocator'])+'.'
  lines.append('| '+case['difficulty'].capitalize()+' | **'+esc(', '.join(case['forms']))+'** ‘'+esc(case['gloss'])+'’ | '+str(len(case['formIds']))+' | '+evidence+' |')
 if data.get('sourceUrl'):lines+=['',data['evidence']+' [Source]('+data['sourceUrl']+').']
(R/'REVIEW.md').write_text('\n'.join(lines)+'\n');print('Rendered',len(qs),'remaining Shina proposals')
