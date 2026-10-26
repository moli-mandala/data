import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};priorheld={x['record']['ID'] for f in P.glob('pass*-decisions.json') for x in json.loads(f.read_text()).get('held',[])};a=[];held=[]
for r in rs:
 if r['ID'] in done or r['Language_ID']!='Kho':continue
 f,g=r['Form'],r['Gloss'];x=None;reason=None
 if g=='sleep' and f in ['ore','ūre']:
  x=('807','CDIAL[807];bashir2023[s.v. oreék]','Turner avarati explicitly gives Khowar orói sleep and its verb oréik to sleep. Bashir confirms oreék/oróik. These simple survey ore/ūre sleep responses match that verbal family; the unrelated poris response is excluded.')
 if g=='salt' and f=='trūpʰ':
  x=('5934','CDIAL[5934];CDIAL[5848,comparative cross-reference];bashir2023[s.v. trup]','Turner tr̥pra (5934) explicitly gives Khowar trup salt, also radish. Bashir confirms trup/thrup. Although Bashir cites T5848, the full tīvrá entry only cross-references Khowar trup to tr̥pra; 5934 is therefore the exact parent. Final aspiration in the survey is retained, not silently normalized.')
 if g=='die' and f=='obri-':
  x=('10383','CDIAL[10383 and addenda];bashir2023[s.v. brik]','Turner mriyate explicitly gives Khowar bri-, brik and past obristai; Bashir gives obrítay beside older obrístam. The survey obri- is this documented prefixed verbal stem, not the distinct birdu dead participle under mr̥ta 10278.')
 if g=='run' and f=='de':
  x=('6802-2','CDIAL[6802,section 2];bashir2023[s.v. deék]','Turner section 2 dhāvayati explicitly gives Khowar deik both make run and intransitive run/gallop. Bashir distinguishes intransitive deék from transitive deík. The survey de run is matched to this documented de- family; choosing the causative historical subsection does not change its current intransitive meaning.')
 if g=='hungry' and f in ['ʦ̣ūī','ʦ̣ī','ʦ̣ūyī','ʦ̣hūī']:
  x=('3716-3','CDIAL[3716,section 3];bashir2023[s.v. c̣huy]','Turner explicitly separates Khowar c̣hui hunger under *kṣudhikā (section 2) from c̣hui hungry under *kṣudhin (section 3). The survey adjective belongs to section 3; Bashir confirms c̣huy hungry beside c̣huí hunger. The local renderings differ in aspiration and vowel notation but preserve the diagnostic retroflex affricate and predicate meaning.')
 if g=='leaf':reason='Bashir c̣han cites T4979, but Turner explicitly reports chādana as the proposed source while filing it under chadana and says the retroflex affricate remains unexplained. Do not resolve the vowel/subsection and consonant problems by a bare reference number.'
 if g=='meat' and f in ['pʰūśūr','pūśūr']:reason='Full CDIAL7985 paśutā and 8217 piśita give competing analyses of Khowar pušur meat. Bashir cites 7985, but the correspondence and competing proposal remain unresolved; no single parent accepted.'
 if g=='speak' and f=='lū':reason='Bashir lu is a noun word/speech; speak is lu dik. Turner10934 also marks the comparison tentative. The survey bare lu under speak does not authorize silently adding the missing verbal component.'
 if g=='black' and f in ['ṣā́','ṣā']:reason='Bashir s.v. ṣa explicitly gives an Iranian source (Morgenstierne1936). No supported immediate donor lexeme has been resolved; do not assign Sanskrit śyāma from appearance alone.'
 if g=='rice' and f in ['grīny','grinc']:reason='Bashir grinǰ rice explicitly has Iranian origin, with no immediate donor resolved. grīny also falls in the SSNP j-to-y import problem. No remote Sanskrit link is accepted.'
 if x:a.append(dict(record=r,parent=x[0],kind='reflex',citation=x[1],evidence=x[2]))
 elif reason and r['ID'] not in priorheld:held.append(dict(record=r,reason=reason))
(P/'pass116-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print('accepted',len(a),'held',len(held))
