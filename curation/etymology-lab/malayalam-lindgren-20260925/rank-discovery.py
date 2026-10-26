"""Exploratory ranked same-language homepage comparanda; never assign automatically."""
import difflib,glob,json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==680
results=[]
for f in sorted(P.glob('discovery-*-results.json'),key=lambda x:int(x.name.split('-')[1])):results+=json.loads(f.read_text())
assert len(results)==680
def fold(s):
 s=s.lower().translate(str.maketrans({'ɨ':'u','ə':'a','ɔ':'o','ɛ':'e','ŋ':'n','ñ':'n','ṭ':'t','ḍ':'d','ṇ':'n','ḷ':'l','ɻ':'l','ṟ':'r','ṛ':'r','ś':'s','š':'s','̪':'','ː':''}))
 s=''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c))
 return re.sub('[^a-z]','',s)
out=[]
for i,(rec,result) in enumerate(zip(records,results),1):
 assert result['opts']['params']['form']==rec['Form']
 form=fold(rec['Form']);gwords={w for w in re.findall(r'[a-z]{3,}',rec['Gloss'].lower()) if w not in {'the','and','with','from','something','person','animate','inanimate','singular','plural','female','male','third','second','first'}}
 c=[]
 for row in result['rows']:
  pid=row.get('etymon_id')
  if not pid or not re.fullmatch(r'd\d+[A-Za-z]?',pid):continue
  fs=difflib.SequenceMatcher(None,form,fold(row['word'])).ratio()
  rg={w for w in re.findall(r'[a-z]{3,}',row['gloss'].lower())}
  gs=len(gwords&rg)/max(1,len(gwords))
  score=round(fs*.7+gs*.3,3)
  if fs>=.55:c.append({'parent':pid,'form':row['word'],'gloss':row['gloss'],'formScore':round(fs,3),'glossScore':round(gs,3),'score':score})
 c.sort(key=lambda x:(x['score'],x['formScore']),reverse=True)
 out.append({'index':i,'form':rec['Form'],'gloss':rec['Gloss'],'candidates':c[:5]})
(P/'homepage-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('with candidate',sum(bool(x['candidates']) for x in out),'strong',sum(any(c['formScore']>=.8 and c['glossScore']>0 for c in x['candidates']) for x in out))
