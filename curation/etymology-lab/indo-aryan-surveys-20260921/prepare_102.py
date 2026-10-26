import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];held=[]
groups=[
('6097','spit',['tʰuk','tʰũk','ṭʰuk','tʰukyo','tʰukla','tʰukʎa','tʰukilak','tʰuklio'],'Turner *thukk explicitly gives Nepali thuknu, Maithili thukab and the corresponding regional spittle noun. The selected simple survey verbs, including ordinary finite endings, match this family. Dang initial retroflex ṭ is retained as source notation. Negative or multi-verb responses are excluded.'),
('5384','take out',['jʰik','jʰikla','jʰikilik','jʰiklio','jʰikyo','jik'],'Turner *jhikk first branch explicitly gives Nepali jhiknu to pull in, take out or off. These adjacent Nepal survey verbs match its form and exact sense, including ordinary finite endings and the Dang unaspirated spelling. No nasal *jhiṅk branch or additional negative component is assumed.'),
('1174','bring',['ān','ānla','ānlik','ānyo'],'Turner ānayati first branch gives the regional ān bring family, including Bihari ānal, Hindi ānnā and Assamese/Bengali ān-. The selected bare and finite survey responses preserve this root. The le/lei plus bring compounds and negative clauses require additional analysis and are excluded.'),
('6886','wash',['dʰo','dʰoyo','ḍʰo','dʰoila','do̤ilak','dʰɔilio','dʰoile','ḍʰu'],'Turner *dhauvati first branch explicitly includes Nepali dhunu, Maithili dhoeb, Bhojpuri dhowal and Hindi dho-/dhov-. These regional simple wash forms and finite derivatives match that family. The source retroflex and breathy spellings are preserved; compounds and the less clear dhai-/dhe- forms are excluded.'),
('4578','climb up',['caḍ','carilak','carlo','cardyo','coṛyo','coriyo','coryo','coḍ','cʰār','tsārla','tsarla'],'Turner *caḍhati first branch explicitly gives Nepali caṛnu, Bhojpuri caṛhal and regional affricate caḍ-/caṛ- climb verbs. These survey forms are compared with that regional family, retaining the source ts/c and aspirated variants and ordinary finite endings. Longer and negative responses are excluded.')]
for parent,gloss,forms,ev in groups:
 for r in rs:
  if r['ID'] not in done and r['Gloss']==gloss and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=ev))
for r in rs:
 if r['ID'] not in done and r['Gloss']=='come down' and r['Form'] in ['jʰar','jaryo','jʰārla','jʰarla','jʰārilak','jʰarlo']:
  held.append(dict(record=r,reason='CDIAL 5328 explicitly allows Nepali jharnu to derive either from *jhaṭati or *jharati 5346. The survey does not choose between those histories; long-vowel transitive-looking variants add uncertainty.'))
(P/'pass102-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a),len(held))
