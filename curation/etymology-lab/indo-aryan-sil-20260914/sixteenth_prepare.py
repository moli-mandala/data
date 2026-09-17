import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
qs=[]
def add(parent,gloss,words,cite,ev,query,file):
 primary=next(x for x in json.loads((P/file).read_text()) if x['word']==query)
 qs.append(dict(parent=parent,gloss=gloss.split('|'),words=words.split('|'),citation=cite,evidence=ev,primary=primary))
add('f_s4nms4eshmvdm','week','hafta|hapta|hapto|hafto|haphta|haptu|haftu|afta|hapṭa|hafṭa|haptah','platts1884[s.v. hafta];liljegren[entry LX000908]','Platts p. 1230 documents hafta “week” as Persian-derived; the existing Hindi/Urdu hafta donor is independently attested by Liljegren, whose Palula article explicitly identifies Urdu transmission. Regional hapta/hapto forms adapt f to ph/p and ordinary noun endings.','hafta','platts-research.json')
add('f_gsynajbspnc2y','man|husband','admi|adami|adimi|odmi|odami|admin','platts1884[s.v. admi]','Platts p. 33 documents Urdu/Hindi ādmī “human being, man, husband”. The survey admi/adami/adimi family has the same consonants and documented man/husband senses, with vowel insertion or rounding. The exact earlier Arabic/Persian history is not represented by a direct remote-ancestor edge.','ādmī','platts-additional-research.json')
add('f_6sqgvw7opxpy6','sky','asman|asaman|aśman|asmaan','platts1884[s.v. āsmān]','Platts p. 53 documents Urdu/Hindi āsmān/asmān “sky” as a Persian loan. The survey forms preserve the distinctive sman sequence, sometimes with an inserted vowel.','آسمان','platts-final-research.json')
add('f_7tuydfvqt7ici','body','badan|bodon|badon','platts1884[s.v. badan]','Platts p. 141 distinguishes Arabic-derived Urdu/Hindi badan “body” from Hindi badan “mouth/face” descended from vadana. Only the body sense is linked here.','badan','platts-additional-research.json')
add('f_vhi3eao4uvyjm','woman|wife','aurat|orat|aurath|orath','platts1884[s.v. aurat]','Platts p. 766 explicitly states that aurat means “woman, wife” in Urdu, distinguishing its earlier Persian/Arabic sense. The survey aurat/orat responses agree with that regional semantic development.','عورت','platts-final-research.json')
add('f_bgpkwqir4pwla','fingernail|nail|nails','nakhun|naxun|nakhon|naxun','platts1884[s.v. nakhun]','Platts p. 1112 documents Urdu/Hindi nākhun “fingernail, toenail, claw” as a Persian loan. The survey forms retain the diagnostic two-syllable khun/xun sequence, distinct from inherited nakh.','nāḵẖun','platts-additional-research.json')
add('f_ymjgts524mota','few','kam|kom','platts1884[s.v. kam];liljegren[entry kam]','Platts p. 846 distinguishes Persian-derived kam “less, little, scanty” from kām “work/desire” and Arabic interrogative kam. The existing Hindi/Urdu kam donor is also identified in Liljegren’s Palula loan analysis.','kam','platts-additional-research.json')
current={r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1'}
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};acc=[];held=[]
for rr in csv.DictReader((P/'unresearched-records.csv').open()):
 if rr['ID'] in current:continue
 r=raw[rr['ID']];gs={g.strip().lower().rstrip('?!.') for g in r['Gloss'].split(';')};w=norm(r['Form'])
 for i,q in enumerate(qs):
  if w not in {norm(v) for v in q['words']} or not gs<=set(q['gloss']):continue
  # Avoid inferring Hindi mediation where Iranian contact presents a distinct live alternative.
  if rr['clade'] in {'Kohistani','Chitrali','Shinaic','Kunar'} or r['Language_ID']=='H':continue
  if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):held.append(dict(record=r,families=[i],reason='Source uncertainty requires review before choosing the provisional regional donor.',passNumber=16));continue
  ev=q['evidence']+' Provisional immediate-donor hypothesis: Hindi/Urdu, with borrowing between neighboring Indo-Aryan languages still possible. The source establishes the lexical family, not the direction of transfer into this particular survey locality. Saved as borrowed under the user’s explicit preference to link supported families despite unresolved cross-IA transmission; audit the donor route.'
  acc.append(dict(record=r,family=i,parent=q['parent'],kind='borrowed',citation=q['citation'],evidence=ev))
(P/'sixteenth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'sixteenth-primary-articles.json').write_text(json.dumps({q['parent']:[dict(url=q['primary']['url'],text=q['primary']['text'])] for q in qs},ensure_ascii=False,indent=1));(P/'sixteenth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','sixteenth-decisions.json').replace('sixth','sixteenth').replace("Kind='reflex'","Kind=x.get('kind','reflex')").replace("grouped[(x['parent'],x['citation'],x['evidence'])]","grouped[(x['parent'],x['citation'],x['evidence'],x.get('kind','reflex'))]").replace('((parent,cite,ev),ys)','((parent,cite,ev,kind),ys)').replace("kind='reflex',citation=cite","kind=kind,citation=cite")
(P/'sixteenth_save.py').write_text(s)
print('accepted',len(acc),'held',len(held))
for i,q in enumerate(qs):
 xs=[x for x in acc if x['family']==i];print(q['parent'],len(xs),'; '.join(sorted({x['record']['Language_ID']+' '+x['record']['Form'] for x in xs})))
