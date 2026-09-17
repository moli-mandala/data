import json,csv,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'prepare.py').read_text();exec('def norm'+src.split('def norm',1)[1].split('families=json.loads')[0]);oldnorm=norm
def norm(s):
 s=oldnorm(s.replace('ɡ','g').replace('ǰ','j').replace('č','c').replace('ˈ','').replace('ˌ',''))
 return re.sub(r'([bcdfghjklmnpqrstvwxyzṭḍṇḷṛśṣ])ː',r'\1\1',s)
qs=[]
def add(parent,gloss,words,ev,loc=None):qs.append(dict(parent=parent,gloss=gloss,words=words.split('|'),evidence=ev,citation='CDIAL['+(loc or parent.replace('-','.',1))+']'))
add('8118-2','near','pas|pās|pāse|pase|pasi|pāsi|pass|passe','CDIAL 8118.2 explicitly groups Punjabi pās, Hindi pās, eastern pās/pāse and Gujarati pāse “near” under the pārśvatas/pārśve adverbs. This selects the adverbial branch rather than the anatomical side/rib noun pārśva.','8118.2')
add('9502','wet','bhija|bhijo|bhiji|bhijal|bhijala|bhijla|bhijlāh|bhijlah|bhijlo|bhijli|bhijela|bhijelo|bhijel','CDIAL 9502 *bhiyajyate gives Bengali bhijā, Maithili bhijlāh and Bhojpuri bhījal alongside the widespread bhij- “get wet” verb. These simple participial/result forms express wetness; no opaque auxiliary or compound response is treated as an inherited stem.')
add('10636-2','good','ramro|rāmro|ramṛo|rāmṛo','CDIAL 10636 explicitly separates the -ḍ- extension, with Western Pahari rāmṛā/ramṛo and Nepali rāmro “beautiful, good”. Jambu’s *ramyaḍ- node selects that extension rather than bare ramya; borrowing into other Nepal-area languages needs independent evidence.','10636, -ḍ- extension')
add('7150','good','nik|nīk|nika|nīka|niko|nikāh','CDIAL 7150 nikta is represented by Middle Indo-Aryan *nikka, Prakrit ṇikka, Maithili nīk/nikāh and Hindi nīkā “good”. The article separately notes Persian nēk influence on nekā; e-vowel forms are excluded from this shortlist.')
add('5731','below','tal|tala|tale|tali|taḷe|taḷ|tar|tara|tare|tol|tole','CDIAL 5731.1 tala explicitly includes adverbial Nepali tala, Bengali/Assamese tale, Bhojpuri tar and Hindi tar/tare/tale “below”. This is the base/bottom noun used as an adverb; doubled-l forms that may select *talla are not included.','5731.1')
add('13561','thread','sut|suta|suto|sūta|sūto|sutar|suttur|sūtar|sūtur|sutr|sutri','CDIAL 13561 sūtra gives Prakrit sutta, Hindi sūt, Bengali/Oriya sutā, Gujarati sutar and Dardic sūtr forms “thread”. The r-retaining Western Pahari forms explicitly marked as Punjabi loans are held; -l- extensions have a different existing parent.')
add('8857','stone','pathar|pāthar|patthar|pathara|pātthar|pathal|patthal','CDIAL 8857 prastara gives Pali/Prakrit patthara “stone” and Punjabi patthar, Western Pahari pāthar and eastern pāthar/pathara. It explicitly marks several Central/eastern patthar/pathar forms as Punjabi loans; those transmission-sensitive matches are held rather than assigned a reflex edge.')
add('142','good','accha|achha|accho|acchi|acho|achā|ācho|āchā|achyo','CDIAL 142 accha gives Punjabi acchā “good”, explicitly borrowed into standard Hindi, and Old Marwari āchyo/āchī. The route is decided locally: standard Hindi is linked to the existing Punjabi attestation; direct western āch- continuations are kept separate from unresolved regional contact forms.')
(P/'sixth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
linked={r['Child_ID'] for r in csv.DictReader((ROOT/'cldf/edges.csv').open()) if r['Rank']=='1'};linked.update(r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1')
held={x['record']['ID'] for f in ['decisions.json','second-decisions.json','third-decisions.json','fourth-decisions.json','fifth-decisions.json'] for x in json.loads((P/f).read_text())['held']};excluded=linked|held
ws=[{norm(w) for w in q['words']} for q in qs];cs=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in excluded:continue
 w=norm(r['Form']);gs={g.strip().lower() for g in r['Gloss'].split(';')}
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i,q in enumerate(qs) if w in ws[i] and gs<=set(q['gloss'].split('|'))]
 if ii:cs.append(dict(record=r,families=ii))
(P/'sixth-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
with (P/'sixth-review.txt').open('w') as f:
 for i,q in enumerate(qs):
  rs=[x['record'] for x in cs if i in x['families']];f.write(f"\n{i}. {q['parent']} {q['gloss']} ({len(rs)})\n");f.write('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs}))+'\n')
print(len(cs))
