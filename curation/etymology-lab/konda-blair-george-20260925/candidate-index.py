import csv,json,re,unicodedata,difflib,collections
from pathlib import Path
P=Path(__file__).resolve().parent
S=json.load(open(P/'survey-inventory.json'))
E={}
with open(P.parents[2]/'cldf/edges.csv') as f:
 for r in csv.DictReader(f):
  if r['Kind']=='reflex' and r['Rank']=='1':E[r['Child_ID']]=r['Parent_ID']
D=[]
with open(P.parents[2]/'cldf/forms.csv') as f:
 for r in csv.DictReader(f):
  if r['Language_ID']=='Konda' and r['Source'].startswith('dedr') and r['ID'] in E:
   D.append((r['Form'],r['Gloss'],E[r['ID']]))
def norm(s):
 s=unicodedata.normalize('NFD',s.lower());return re.sub('[^a-z]','',s)
def formsim(a,b):
 return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()
def glossim(a,b):
 a,b=norm(a),norm(b)
 if a and a in b or b and b in a:return 1
 A=set(re.findall('[a-z]+',a));B=set(re.findall('[a-z]+',b));return len(A&B)/len(A|B) if A and B else 0
out=[]
for i,r in enumerate(S,1):
 matches=sorted(D,key=lambda x:0.68*formsim(r['Form'],x[0])+0.32*glossim(r['Gloss'],x[1]),reverse=True)[:12]
 out.append(dict(index=i,form=r['Form'],gloss=r['Gloss'],matches=[{'form':f,'gloss':g,'parent':p,'score':round(0.68*formsim(r['Form'],f)+0.32*glossim(r['Gloss'],g),3)} for f,g,p in matches]))
json.dump(out,open(P/'candidate-index.json','w'),ensure_ascii=False,indent=2)
for o in out:
 print(f"{o['index']:03} {o['form'][:21]:21} {o['gloss'][:13]:13} "+' | '.join(f"{m['form'][:15]}:{m['gloss'][:15]}:{m['parent']} {m['score']}" for m in o['matches'][:4]))
