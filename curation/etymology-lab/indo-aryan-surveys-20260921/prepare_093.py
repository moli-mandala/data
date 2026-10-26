import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('10234','urine',['mūṭra'],208,'Trail and Cooper explicitly derive muṭra urine from mūtra; Turner cites the Kalasha form in this entry.'),
('7087','kill',['nāś-'],213,'Trail and Cooper give naśik kill and explicitly cite nāśayati. This is the causative kill stem, distinct from intransitive die.'),
('7027','die',['nāśi-','ānāśī-','ānāśi-'],214,'Trail and Cooper explicitly derive naśik die from naśyati and illustrate anaśaw he died. The survey a-prefixed forms belong to this verbal paradigm, not the separate naṣṭa adjective.'),
('7467-2','sit down',['nīs-','nis-'],218,'Trail and Cooper explicitly cite *nisīdati, T-7467.2, and give the nis- present paradigm. Turner also cites Kalasha nīsim. The precise second subsection is retained.'),
('7914','go',['pār-'],230,'Trail and Cooper explicitly derive parik go from paraiti. Only the par- stem is assigned here; the suppletive past gaḷa is from a different root.'),
('8012','see',['pāśī-'],231,'Trail and Cooper explicitly derive paśik see from paśyati, matching the survey verbal stem.'),
('2981','walk',['kāsi-','kāsī-'],151,'Trail and Cooper explicitly derive kasik walk from kasati. The survey forms preserve this verbal stem.'),
('10187-2','tree',['mūṭ'],208,'Trail and Cooper give muṭ tree with etymology *muṭṭha, T-10187.2. Turner’s second subsection explicitly includes the Kalasha and Phalura tree forms; the general preceding branch is not substituted.'),
('9661','brother',['bāyā'],34,'Trail and Cooper explicitly derive baya brother from bhrātṛ. The survey forms match the cited Kalasha reflex.'),
('12555','dry',['ṣūṣtā'],294,'Trail and Cooper explicitly derive ṣuṣṭa dry from *śuṣṭa. The survey forms match this adjective, including its retroflex cluster.'),
('10323','fat',['mɛ̃','mɛ'],198,'Trail and Cooper explicitly derive me fat from medas. Turner cites Rumbur meʰ and Urtsun mɛ̃, supporting the recorded dialect variation.'),
('9882','monkey',['mɛ̣̄kā','mɛ̣̄̃ka','makụ̄yɛk'],198,'Trail and Cooper place mạkọyak monkey and its Urtsun variant mạka under markaṭa. These survey forms are compared with that explicitly identified local family, retaining their vowel and diminutive variation.'),
('4104','snake',['gok','gokʰ'],108,'Trail and Cooper explicitly derive gok snake from gavedhuka. The two velar-initial survey forms match this local reflex; the divergent cuana dialect synonym is held separately.'),
('986','we (incl.)',['ābī'],None,'Turner asmad explicitly lists Kalasha ābi we. The survey’s inclusive label is retained without asserting an inherited inclusive/exclusive distinction.')]
a=[]
for i,(parent,gl,forms,page,ev) in enumerate(groups):
 for r in rs:
  if r['Language_ID']=='Kal' and r['ID'] not in done and r['Gloss']==gl and r['Form'] in forms:
   a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]'+(f';trail-cooper1999[printed p. {page}, PDF p. {page+25}]' if page else ''),evidence=ev,family=i))
held=[]
for r in rs:
 if r['Language_ID']=='Kal' and r['ID'] not in done:
  reason=None
  if r['Gloss']=='bone':reason='Turner identifies borrowing into the Dardic group; an immediate donor is not securely resolved.'
  if r['Gloss']=='sit down' and r['Form']=='nas-':reason='The vowel differs from the inspected nis- paradigm; retain pending direct form-level confirmation.'
  if r['Gloss']=='snake' and r['Form']=='cūānā':reason='Trail and Cooper list cuana as an Urtsun variant under gok, but the divergent form needs an explicit etymological bridge.'
  if reason:held.append(dict(record=r,reason=reason))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass093-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n')
(P/'pass093-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
