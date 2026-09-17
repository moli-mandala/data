import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};eligible={r['ID'] for r in csv.DictReader(open(P/'unresearched-records.csv'))};eligible.update(x['record']['ID'] for x in json.loads((P/'audit-records.json').read_text()))
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
donor='f_br7axlvon4m3a';parts=['f_a72ti56gttur4','f_udz5ccpb6imfc']
ev='Platts p. 1095 s.v. mūṅg explicitly lists mūṅg-phalī as the groundnut Arachis hypogaea, separately from mūṅg-kī phalī the mung-bean pod. CDIAL 10198 gives Hindi mū̃g mung bean; CDIAL 9051 gives Hindi phalī pod. The whole groundnut compound therefore has two ordered Hindi lexical components, not a single mudga ancestor.'
borrow='Platts p. 1095 identifies Hindi mūṅg-phalī groundnut; the Hindi Saraiyya survey records muŋphali (kannauji p. 72). The whole matching compound is linked to that existing Hindi attestation as a provisional regional borrowing route. The attestation village is not asserted to be the historical donor locality; intermediate Indo-Aryan transmission remains open. Both components of the Hindi donor are linked in this same pass.'
qs=[dict(parent=parts[0],citation='Platts[p. 1095];CDIAL[10198,9051]',evidence=ev),dict(parent=donor,citation='Platts[p. 1095];kannauji[p. 72]',evidence=borrow)]
acc=[];held=[]
for fid in sorted(eligible):
 r=raw[fid]
 if r['Gloss']!='groundnut':continue
 w=norm(r['Form']).replace(' ','').replace('ɸ','ph')
 # These shapes were inspected in the full groundnut inventory: nasal assimilation,
 # contracted mung-, ph/f, l/retroflex lateral and r in the pod component.
 if not re.fullmatch(r'm[uūo]+[mnŋṇgɡhũ]*[pf](?:h)?(?:a|ə|e|ʌ)?(?:l|ḷ|r|ṛ)+i',w):continue
 if r['Language_ID']=='H':
  acc.append(dict(record=r,family=0,parent=parts[0],components=parts,kind='component',citation=qs[0]['citation'],evidence=ev))
 else:acc.append(dict(record=r,family=1,parent=donor,kind='borrowed',citation=qs[1]['citation'],evidence=borrow+' Survey compound: '+r['Form']+'.'))
assert donor in {x['record']['ID'] for x in acc}
(P/'groundnut-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'groundnut-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
print('records',len(acc),'rows',sum(len(x.get('components',[])) or 1 for x in acc));print('Hindi',[x['record']['Form'] for x in acc if x['kind']=='component']);print('Forms',sorted({x['record']['Form'] for x in acc}))
