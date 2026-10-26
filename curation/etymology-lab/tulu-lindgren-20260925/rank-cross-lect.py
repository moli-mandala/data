"""Exploratory Lindgren cognate-set lookup, never an automatic assignment."""
import collections,csv,gzip,json,re,unicodedata,difflib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==1730
prior={}
for name in ('tamil-lindgren-20260925','malayalam-lindgren-20260925','kannada-lindgren-20260925'):
    by=collections.defaultdict(set)
    for x in json.loads((P.parent/name/'proposed-decisions.json').read_text())['accepted']:
        by[x['record']['Cognateset']].add(x['parent'])
    prior[name.split('-')[0]]=by
out=[]
for ix,rec in enumerate(records,1):
    parents={lect:next(iter(by[rec['Cognateset']])) for lect,by in prior.items()
             if len(by.get(rec['Cognateset'],set()))==1}
    if parents:
        out.append({'index':ix,'form':rec['Form'],'gloss':rec['Gloss'],'cognateset':rec['Cognateset'],
                    'comparanda':parents,'consensus':next(iter(set(parents.values()))) if len(set(parents.values()))==1 else None})
(P/'cross-lect-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
target={x['consensus'] for x in out if x['consensus']}
articles={}
with gzip.open(R/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in target:
            articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert target==set(articles)
def subsection(article):
    m=re.search(r'\bTu\.\s',article)
    if not m:return ''
    rest=article[m.end():]
    end=re.search(r'\s(?:Te|Kol|Nk|Pa|Ga|Go|Konḍa|Kui|Kuwi|Br|Ma|Ir|Ko|To|Ka|Koḍ)\.\s',rest)
    return rest[:end.start()] if end else rest
def fold(s):
    s=s.lower().translate(str.maketrans({'ɨ':'u','ə':'a','ɔ':'o','ɛ':'e','ŋ':'n','ñ':'n','ṭ':'t','ḍ':'d','ṇ':'n','ḷ':'l','ɻ':'l','ṟ':'r','ṛ':'r','ś':'s','š':'s','̪':'','ː':''}))
    return re.sub('[^a-z]','', ''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c)))
for x in out:
    pid=x['consensus']
    if not pid:continue
    s=subsection(articles[pid]['Raw_Article'])
    x['tuluSubsection']=s[:320]
    x['tuluAttested']=bool(s)
    x['formMentioned']=bool(fold(x['form']) and fold(x['form']) in fold(s))
(P/'cross-lect-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'candidateRecords':len(out),'consensusRecords':sum(bool(x['consensus']) for x in out),
                  'TuluSubsectionRecords':sum(bool(x.get('tuluAttested')) for x in out),'articles':len(articles)}))
