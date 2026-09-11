import json,pickle,re
from pathlib import Path
# Preserve the approved review snapshot when no new batch awaits review.
_review_dir = Path(__file__).resolve().parent
_review_batches = list(_review_dir.glob('batch-*.json'))
if _review_batches and all(json.loads(f.read_text()).get('status') == 'saved' for f in _review_batches):
    print('All batches saved; preserving the approved review snapshot.')
    raise SystemExit(0)
R=Path(__file__).resolve().parent;D=R.parents[2];pages={}
for n,page in enumerate(pickle.load(open(D/'data/cdial/cdial.pickle','rb')),1):
 for k in re.findall('<number>([^<]+)</number>',page):pages.setdefault(k,n)
p=json.loads((R/'progress.json').read_text());qs=[q for f in sorted(R.glob('batch-*.json')) for q in json.loads(f.read_text())['proposals']]
def esc(x):return x.replace('|','\\|').replace('\n',' ')
lines=['# Brokskat: pending overnight review','','All proposals remain pending user review. Numbering is local to the Brokskat pass.']
withdrawn=[n for f in R.glob('batch-*.json') for n in json.loads(f.read_text()).get('withdrawnProposalNumbers',[])]
if withdrawn:lines+=['','Withdrawn proposal numbers: '+', '.join(map(str,sorted(withdrawn)))+'. Their original analyses are preserved in withdrawn-proposals.json; the unresolved evidence is in DIFFICULT-REVIEW.md.']
for level in ['straightforward','moderate','difficult']:
 sub=[q for q in qs if q['difficulty']==level];lines+=['','## '+level.capitalize()+f' ({len(sub)})','','| # | Brokskat | Proposed etymology | Evidence |','|---|---|---|---|']
 for q in sub:
  label='**'+q['parentLabels'][0].replace('*','\\*')+'**'
  if q['kind']=='component':label='Brokskat components: '+' + '.join('**'+x+'**' for x in q['parentLabels'])+' ('+q['compositionOperation']+')'
  elif q['kind']=='derived':label='Derived from Brokskat '+label
  elif q['kind']=='borrowed':label='Borrowed from '+(q.get('donorLanguageName','')+' ').lstrip()+label
  match=re.search(r'CDIAL\[(\d+)',q['evidenceSource']);citation=q['evidenceSource']
  if match: citation='['+citation+'](https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page='+str(pages[match[1]])+')'
  evidence=esc(q['evidence'])
  if q.get('sourceUrl'):evidence+=' [Primary source]('+q['sourceUrl']+')'
  lines.append('| '+str(q['number'])+' | **'+esc(', '.join(q['forms']))+'** ‘'+esc(q['gloss'])+'’ | '+label+'; '+citation+' | '+evidence+' |')
lines+=['','## Coverage','',f"{len(qs)} pending proposals cover {len({i for q in qs for i in q['formIds']})} records with {sum(len(q['assignments']) for q in qs)} proposed assignment rows. {len(p['unexaminedIds'])} Brokskat records remain unresolved or unexamined.",'','Pending rows validate jointly with this task’s earlier dependencies. No accepted overlay or compiled data changed.','', 'Excluded from this pass: unanalysed multiword elicitation responses, the unexplained LSI spelling for eight, Tibetan loan candidates without a verified immediate donor, and horse ancestry where an Iranian route competes with direct inheritance. Difficulty measures review effort, not approval status.']
lines += ['', '[Source-checked unresolved cases](DIFFICULT-REVIEW.md)']
(R/'REVIEW.md').write_text('\n'.join(lines)+'\n');print('Rendered',len(qs),'Brokskat proposals')
