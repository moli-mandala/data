"""Exploratory exact form/gloss links from earlier review passes; not assignments."""
import collections,glob,json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==1504
def fold(s):
    s=''.join(c for c in unicodedata.normalize('NFD',s.lower()) if not unicodedata.combining(c))
    return re.sub('[^a-z]','',s)
prior=collections.defaultdict(lambda:collections.defaultdict(set))
for filename in glob.glob(str(P.parent/'*/proposed-decisions*.json')):
    for row in json.loads(Path(filename).read_text()).get('accepted',[]):
        rec=row['record'];key=(fold(rec['Form']),rec['Gloss'].lower())
        prior[key][row['parent']].add(rec['Language_ID'])
out=[]
for ix,rec in enumerate(records,1):
    ps=prior.get((fold(rec['Form']),rec['Gloss'].lower()),{})
    if len(ps)==1:
        pid=next(iter(ps));out.append({'index':ix,'recordId':rec['ID'],'form':rec['Form'],'gloss':rec['Gloss'],'parent':pid,'lects':sorted(ps[pid])})
(P/'cross-pass-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'candidates':len(out),'parentArticles':len({x['parent'] for x in out})}))
