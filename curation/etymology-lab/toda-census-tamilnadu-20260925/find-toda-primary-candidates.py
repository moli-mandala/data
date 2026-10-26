"""Exploratory gloss+form ranking of cached DEDR Toda subsections; no decisions."""
import csv,gzip,json,re,unicodedata,difflib
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'survey-inventory.json').read_text())
def fold(s):
 s=s.lower().translate(str.maketrans({'ɔ':'o','ɛ':'e','ə':'a','ɯ':'u','ɨ':'u','ʰ':'','θ':'t','ɻ':'l','ɭ':'l','ṛ':'r','ḷ':'l','ṇ':'n','ŋ':'n','ñ':'n','ʝ':'j','ś':'s','š':'s','ḍ':'d','ṭ':'t','ᴂ':'a','ⁱ':'i','ɡ':'g','̃':'','ː':''}))
 s=''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c))
 return re.sub('[^a-z]','',s)
langs=r'Ta|Ma|Ir|Ko|To|Ka|Koḍ|Tu|Kor|Te|Kol|Nk|Pa|Ga|Go|Konḍa|Pe|Manḍ|Kui|Kuwi|Kur|Br|Malt|Naikri|Pengo'
articles=[]
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for a in csv.DictReader(f):
  s=a['Raw_Article'];m=re.search(r'\bTo\.\s',s)
  if not m:continue
  tail=s[m.end():];n=re.search(r'\b(?:'+langs+r')\.\s',tail)
  to=tail[:n.start()] if n else tail
  words=re.findall(r"[A-Za-z\u00c0-\u024f\u1e00-\u1effɔɛəɯɨʰθɻɭṛḷṇŋñʝśšḍṭᴂⁱɡ̃ː·ɨøȳ]+",to)
  fw=[fold(w) for w in words[:200]]
  articles.append((a['Entry_ID'],to,fw,s))
stop={'to','the','and','of','a','an','as','on','for','or','in','with','he','she','it','is','this','that','those','these','something','some','one','side','etc','up','down'}
out=[]
for i,r in enumerate(rs,1):
 form=fold(r['Form']);gloss=fold(r['Gloss']);keys=[fold(w) for w in re.findall(r'[A-Za-z]+',r['Gloss'].lower()) if w.lower() not in stop and len(w)>2]
 if not keys:keys=[gloss]
 cand=[]
 for pid,to,fw,full in articles:
  lower=full.lower()
  if not any(re.search(r'\b'+re.escape(k)+r'\b',lower) for k in keys if k):continue
  best=max((difflib.SequenceMatcher(None,form,w).ratio() for w in fw if 1<len(w)<35),default=0)
  if best<.37:continue
  cand.append((round(best,3),pid,to[:350].replace('\n',' ')))
 cand.sort(reverse=True)
 out.append({'index':i,'form':r['Form'],'gloss':r['Gloss'],'candidates':[{'score':score,'parent':pid,'Toda':to} for score,pid,to in cand[:5]]})
(P/'toda-primary-candidates-full-gloss.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('queries',len(out),'candidate-rich',sum(bool(x['candidates']) for x in out))
