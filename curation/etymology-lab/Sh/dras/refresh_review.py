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
lines=['# Dras Shina: pending overnight review','','All proposals remain pending user review. Numbering is local to the Dras pass.']
for level in ['straightforward','moderate','difficult']:
 sub=[q for q in qs if q['difficulty']==level];lines+=['','## '+level.capitalize()+f' ({len(sub)})','','| # | Dras Shina | Proposed etymology | Evidence |','|---|---|---|---|']
 for q in sub:
  k=q['parents'][0].split('-')[0];label='**'+q['parentLabels'][0].replace('*','\\*')+'**'
  if q['kind']=='borrowed':
   label='Probably borrowed from '+label
   match=re.search(r'CDIAL\[(\d+)',q['evidenceSource']);k=match[1] if match else k
  url=q.get('primarySourceUrl') or ('https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page='+str(pages[k]) if k in pages else '')
  lines.append('| '+str(q['number'])+' | **'+esc(', '.join(q['forms']))+'** ‘'+esc(q['gloss'])+'’ | '+label+'; ['+q['evidenceSource']+']('+url+') | '+esc(q['evidence'])+' |')
lines+=['','## Coverage','',f"{len(qs)} pending proposals cover {len({i for q in qs for i in q['formIds']})} records with {sum(len(q['assignments']) for q in qs)} proposed assignment rows. {len(p['unexaminedIds'])} Dras records remain unresolved or unexamined.",'','Pending rows validate jointly with this task’s earlier dependencies. No accepted overlay or compiled data changed.','', 'The initial mixed tooth/sting exclusion has been lifted after checking both source entries (proposal 6). Other exclusions include nine/drama-staging records, compounds handled in the broader Shina pass, unusual numeral endings, and kinship terms that cannot be equated solely from a shared English gloss.']
(R/'REVIEW.md').write_text('\n'.join(lines)+'\n');print('Rendered',len(qs),'Dras proposals')
