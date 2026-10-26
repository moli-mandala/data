"""Exploratory Ta/Ma DEDR form/sense ranking; never automatic assignments."""
import csv,difflib,gzip,json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
def fold(s):
 s=s.lower().translate(str.maketrans({'ɔ':'o','ɛ':'e','ə':'a','ɯ':'u','ɨ':'u','ʰ':'','θ':'t','ɻ':'l','ɭ':'l','ṛ':'r','ḷ':'l','ṇ':'n','ŋ':'n','ñ':'n','ʝ':'j','ś':'s','š':'s','ḍ':'d','ṭ':'t','ᴂ':'a','ɡ':'g','̃':'','ː':''}))
 s=''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c))
 return re.sub('[^a-z]','',s)
def section(s,lang):
 m=re.search(r'\b'+lang+r'\.\s',s)
 if not m:return ''
 tail=s[m.end():];n=re.search(r'\b(?:Ta|Ma|Ir|Ko|To|Ka|Koḍ|Tu|Kor|Te|Kol|Nk|Pa|Ga|Go|Konḍa|Pe|Manḍ|Kui|Kuwi|Kur|Br|Malt|Naikri|Pengo)\.\s',tail)
 return tail[:n.start()] if n else tail
articles=[]
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for a in csv.DictReader(f):
  s=a['Raw_Article'];comp=section(s,'Ta')+' '+section(s,'Ma')
  if not comp.strip():continue
  words=re.findall(r"[A-Za-z\u00c0-\u024f\u1e00-\u1effɔɛəɯɨʰθɻɭṛḷṇŋñʝśšḍṭᴂⁱɡ̃ː·øȳ]+",comp)
  fw=[fold(w) for w in words[:130]]
  articles.append((a['Entry_ID'],comp,fw,s))
stop={'the','and','of','for','with','this','that','those','these','etc','into','from','verb','person','some','have','make','being','used','also','which','when','where','than','such','whose'}
out=[]
for ix,r in enumerate(records,1):
 f=fold(r['Form']);stem=re.sub(r'(gatu|katu|gadu|kadu|atu)$','',f)
 keys=[fold(w) for w in re.findall(r'[A-Za-z]+',r['Gloss'].lower()) if len(w)>2 and w.lower() not in stop]
 if not keys:keys=[fold(r['Gloss'])]
 c=[]
 for pid,comp,fw,full in articles:
  lower=full.lower()
  if not any(re.search(r'\b'+re.escape(k)+r'\b',lower) for k in keys if k):continue
  score=max((max(difflib.SequenceMatcher(None,x,w).ratio() for x in (f,stem)) for w in fw if 1<len(w)<30),default=0)
  if score>=.48:c.append((round(score,3),pid,comp[:300]))
 c.sort(reverse=True)
 out.append({'index':ix,'form':r['Form'],'gloss':r['Gloss'],'candidates':[{'score':s,'parent':p,'comparanda':t} for s,p,t in c[:5]]})
(P/'primary-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('records',len(out),'with candidates',sum(bool(x['candidates']) for x in out))
