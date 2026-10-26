import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('14028','hammer',['hatʰoṛīyā','hatɔriyā','haṭoṛā','aṭʰorā','aṭʰoṛī','hātvāṛā','haṭoṛo','aṭʰoṛo','tʰoṛī','tʰoṛā','āṭoṛa','aṭʰoṛo / aṭʰoṛī','hocro','hāṭāuri','hotrʰo','atur','hātevaḷī','hathevḍū','hātevaṛī','hātevḍū','hāthɛūḍu','hethoḍī','āthɔḍū','hātheḍū','hāthɛūḍo','hathoūḍā','hāthaūḍo','atūḍū','atura','tʰoḍi','haṭoḍā','tʰauḍā','haṭʰori','tʰauḍi','haṭʰāũḍi','haṭʰoḍi','hatɔḍali','ṭʰoṛā','haṭeḍɔ','haṭhaḍu','haṭhoḍi','āṭhuri','haṭoḍi','hāṭhoḍi','ātuḍi','hataᵘdi'],'Turner *hastakūṭa explicitly gives Punjabi/Lahnda hathoṛā, Nepali hothro, Bengali hātuṛi, Gujarati hathɔṛɔ/athɔṛɔ and Marathi hatoḍā. These survey hammer forms fit that established family, including initial h-loss, retroflex r/ḍ and regional vowel contraction. The severely shortened ṭhoṛā forms are qualified comparisons with the local hathoṛā family; unrelated hammer responses and mixed alternatives are excluded.'),
('10223','pestle',['mūso','musɔ','muhaḷo','mūhū̃','mūhaḷū','mūhava','mūhɔv','mūsaī','mūhūl','mohole','mahalo','mahlo','muhalo','muhayõ','muhava'],'Turner musala explicitly supplies Gujarati musḷũ, Marathi musaḷ, Punjabi muhlā/mohlā and Sindhi muhurī pestle. The h/s forms and local contractions in these survey responses are compared with that regional musala family, with s > h and reduction preserved in the source spelling. Turner treats the word as probably non-Aryan in ultimate origin; this link does not invent an immediate Dravidian donor.'),
('10223-2','pestle',['muçaṇ'],'Turner’s second musala branch, *muṣala/*muśala, explicitly contains Jaunsari mūśṛī and Western Pahari muśḷ. The Jaunsari muçaṇ response is a qualified comparison with this specifically palatal family, preserving its local final consonant; the precise extension is not claimed to be solved.'),
('10199-2','hammer',['mõgarī'],'Turner’s second mudgara branch *mōdgara explicitly includes Hindi mogrā/mogrī hammer and Gujarati mogrī mallet. The Bundeli nasalized mõgarī fits that o-vowel family, with nasalization as recorded rather than silently normalized.'),
('10199-2','pestle',['mogrā'],'Turner’s second branch *mōdgara explicitly gives Prakrit moggara pestle and Hindi mogrā hammer. The Kaithal mogrā pestle matches this branch and its attested tool meaning.')]
a=[]
for i,(parent,gloss,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']==gloss and r['Form'] in forms:
   a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence,family=i))
# The final consonant of Jaunsari muçaṇ is not an ordinary match to the cited muśḷ/mūśṛī; keep it unresolved.
h=[];keep=[]
for x in a:
 if x['family']==2:h.append(dict(record=x['record'],reason='Jaunsari muçaṇ resembles the palatal *muśala family (10223-2), but the final ṇ/extension is not established by the primary comparanda.'))
 else:keep.append(x)
a=keep
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass081-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass081-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
