import json
from pathlib import Path
P=Path(__file__).resolve().parent
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
a=[];held=[]
simple={'ūḍ','ūḍe','ūḍtɔ','ūḍīyo','ūḍḍe','ūḍḍɛ','ūṛdā','uḍḍɛ','ūḍlū','ūḍḷo','ūḍana','ūdla','ūdlanī','ūḍlanī','ūṛla','ūdūla','ūḍtaī','ūḍ, ūḍīyū','ūḍ, ūḍyɔ','ūḍ, ūḍyo'}
gone={'ūḍgayɔ','uḍagyɔ','ūḍgyo','ūḍgyɔ','uḍgyɔ','uḍgɔ','uḍagayɔ','ūḍ-gīyo','ūḍ-goyo','ūḍīgayo','ūḍīgiyo','ūḍīgyo','ūḍgayo','ūḍagayo','uḍigayā','ūḍigayā','uḍay go','ūḍ, ūḍigayo','ūḍ, ūḍigoyo','ūḍ, ūḍgyo','ūḍ, ūḍgayɔ','uḍo, uḍgɛyo','uḍ, uḍīgyo','ūḍ, ūḍīgɔ','ūḍ, ūḍigɔ'}
for fn,needle in [('018','Addenda 5'),('028','10247 explicitly'),('020','2528 explicitly')]:
 for h in json.loads((P/f'pass{fn}-decisions.json').read_text())['held']:
  if needle not in h['reason'] or h['record']['ID'] in done:continue
  r=h['record'];f=r['Form']
  if fn=='028':a.append(dict(record=r,parent='10247',kind='reflex',citation='CDIAL[10247 and addenda];CDIAL[10191]',evidence='User adjudication: retain mūrdhan (10247) as the head-family parent, explicitly allowing contamination by muṇḍa shaved (10191). Turner lists the unaspirated muṇḍ/mūṛ/muṛiyā family and regional extended forms, but notes possible derivation from or contamination by muṇḍa. The saved parent represents the approved analysis, not a claim that the alternative has been disproved; local nominal extensions are retained.'))
  elif fn=='020':a.append(dict(record=r,parent='2528',kind='reflex',citation='CDIAL[2528,addenda]',evidence='User adjudication: follow Turner’s analysis of kab/kabe when as interrogative remodeling of the ēvam ēva series, explicitly crossed with ka. Turner gives Bengali kabe, Hindi kab/kabhī and western Pahari kεbε/kabhia. Platts’s competing kiyat-based explanation with jab/tab analogy remains an alternative; this is not an unmodified sound-law reflex of ēvam ēva.'))
  elif f in simple:a.append(dict(record=r,parent='1697-2',kind='reflex',citation='CDIAL[1697,section 2 and addenda 5]',evidence='User adjudication selects uḍḍayate (1697-2) for the uḍ-/uṛ- fly family, retaining the later *uddayati alternative in the notes. These complete responses contain the fly stem and ordinary regional inflection, or paired forms of that same verb, with no separately identified auxiliary.'))
  elif f in gone:a.append(dict(record=r,components=['1697-2','4008'],citation='CDIAL[1697,section 2 and addenda 5];CDIAL[4008]',evidence='User adjudication selects uḍḍayate (1697-2) for fly despite the addendum’s *uddayati alternative. The complete response additionally contains gayo/gyo/go, the gone auxiliary under gata (4008); both components are saved in that order. Where the source pairs a bare fly imperative with its past expression, the second component belongs only to the past member.'))
  else:held.append(dict(record=r,reason='User approved 1697-2 for the fly lexical stem. This whole response still requires its auxiliary, paired-answer, causative, or local inflection analysis; no incomplete root-only assignment has been saved.'))
(P/'pass117-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a),'records',sum(len(x.get('components',[])) or 1 for x in a),'rows',len(held),'held')
