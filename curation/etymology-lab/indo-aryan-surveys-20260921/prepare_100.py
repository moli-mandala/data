import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];held=[]
groups=[('3120',['katʰiyā','kāṭʰiyā','kaṭiyā','kaṭʰūā','kaṭ','ka.to','kāṭi'],'Turner kāṣṭha includes kāṣṭhikā small piece of wood, Hindi kāṭh/kāṭhī and Oriya kāṭha/kāṭhi. These survey wood/fuel forms fit that family with local short vowels, deaspiration and ordinary nominal extensions. The aspirated voiced kāḍhiya spelling is held separately.'),('1584',['indan','indanā'],'Turner indhana fuel gives Punjabi innhaṇ and Hindi ī̃dhan. The Gojri/Kaithal firewood responses fit this fuel-noun family with local denasalization and deaspiration; they are not the questionable Kalasha tripod sense mentioned in the same entry.'),('5313',['jarna','jārnā','jaranā','jarnā'],'Turner *jvālana explicitly gives Hindi jāran firewood and Bhojpuri jāran. The eastern Nepal-survey jarna/jārnā forms fit this kindling/fuel family with local vowel syncope and final-vowel variation. The distinct jvalana burning/inflammation noun and aspirated jhar- forms are not merged here.')]
for i,(parent,forms,ev) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='firewood' and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=ev,family=i))
for r in rs:
 if r['ID'] not in done and r['Gloss']=='firewood' and r['Form'] in ['kāḍʰiyā','jʰarnā']:held.append(dict(record=r,reason='Possible wood/fuel family, but the recorded consonant difference needs direct dialect evidence.'))
(P/'pass100-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a),len(held))
