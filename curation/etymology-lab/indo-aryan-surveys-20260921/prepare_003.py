import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
original={r['ID']:r for r in json.loads((P/'ssnp-next-original-records.json').read_text())}
font='The legacy-font j in this survey represents printed ǰ, as verified visually in its Hindko and Gujari appendices; ssnp conversion currently turns it into y. The analysis follows the original affricate, preserves the current spelling, and does not posit a y-to-j sound change. '
groups=[
dict(parent='5086',gloss=['root'],forms=['yaṛ'],citation='CDIAL[5086,1];rensch-hallberg-oleary1992[item 63, pp. 218, 266]',evidence=font+'The exact root columns were checked in the scan. Section 1 jáṭā gives Lahnda jaṛṛ, Punjabi jaṛ/jaṛh and Hindi/Marwari jaṛ “root”; the short-vowel survey form fits this branch, not section 3 *jāṭā. Turner’s discussion of possible Dravidian/Munda prehistory remains unresolved by this family link.'),
dict(parent='242',gloss=['today'],forms=['ay','āy','āy / ay'],citation='CDIAL[242];rensch-hallberg-oleary1992[item 123, pp. 225, 273]',evidence=font+'CDIAL adya gives Prakrit ajja, Lahnda/Punjabi ajj and Hindi/Gujarati āj “today”. Both the short and long vowel alternatives belong to this same family; the Gujari printed column also confirms CAK āǰ / aǰ. The additional Hindko āīte response is excluded because its final material needs separate analysis.'),
dict(parent='13095',gloss=['right'],forms=['sayā','saya','saye','saya / sayā','sayī','sayo','soyī','soyī / soyo','sayɛ / sayo','sayyo / sayyī'],citation='CDIAL[13095];rensch-hallberg-oleary1992[item 138, pp. 226, 274]',evidence=font+'CDIAL explicitly lists Lahnda/Awankari/Punjabi sajjā “righthand” and Sindhi sāj̄o in sajya, distinguishing this family from savya “left”. Masculine/feminine and agreement variants, including both members of the selected slash responses, share the same stem. Gujari printed soǰi/soǰo and sajjo/sajji were checked; their vowel and gemination notation is retained.'),
dict(parent='5098',gloss=['man','husband'],forms=['yaṇā','yaṇa','yano','yaṇo','yanā','yaṇā̃','yano / yaṇā'],citation='CDIAL[5098];rensch-hallberg-oleary1992[items 102, 113]',evidence=font+'Full jána explicitly gives Lahnda jaṇā “person, husband”, Awankari jaṇā “person”, Sindhi j̄aṇo and Marwari jaṇo. This directly supports both survey senses and the local n/ṇ and final-vowel variants. Responses mixing this stem with ādmī, bandā or xasam are excluded; both alternatives in yano / yaṇā are accounted for.'),
dict(parent='9361',gloss=['(you) run!'],forms=['bʰāy','bʰay','bʰay / bʰāy'],citation='CDIAL[9361,1];rensch-hallberg-oleary1992[item 196, p. 281]',evidence=font+'CDIAL section 1 bhajyate gives Lahnda bhajjaṇ, Awankari bhajjuṇ and Punjabi bhajjṇā “flee”, beside bhāj- forms in other languages. These Gujari imperatives retain the affricate and belong to section 1, unlike g-final bhāg forms in section 2. Both slash variants are the same stem. Turner discusses an alternative ultimate flee-root and its coalescence with the break family; the link does not settle that deeper dispute.'),
dict(parent='9361-2',gloss=['(you) run!'],forms=['bʰāgna'],citation='CDIAL[9361,2];rensch-hallberg-oleary1992[item 196, p. 281]',evidence='This Gujari infinitival response has g, not the legacy-font j of the neighbouring bhaj imperatives. Full CDIAL section 2 bhagna explicitly gives Punjabi bhaggᵃnā, Hindi bhāgnā, Nepali bhāgnu and Gujarati bhāgvũ “flee”. The infinitival ending is ordinary verbal morphology, not an additional compound component. Regional transmission and Turner’s deeper break/flee-root discussion remain qualified.'),
]
acc=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] in original and r['Gloss'] in g['gloss'] and r['Form'] in g['forms']:
   acc.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
held=[]
for r in rs:
 if r['ID'] in original and r['Gloss']=='child' and r['Form'] in ['yātak','yākat','yātek','yākat / yātak']:
  held.append(dict(record=r,candidates=['5182'],reason='The original onset is affricate j, not current y. CDIAL 5182 includes Sanskrit jātaka “new-born son”, but the full prose does not provide a matching regional jākat/jātak comparator. Retained intervocalic t/k and metathesis require checking regional derivation or transmission before choosing reflex versus borrowing; do not assign from the Sanskrit resemblance alone.'))
assert len(acc)==len({x['record']['ID'] for x in acc})
(P/'pass003-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass003-decisions.json').write_text(json.dumps({'accepted':acc,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(acc),[(g['parent'],sum(x['family']==i for x in acc)) for i,g in enumerate(groups)],'held',len(held))
