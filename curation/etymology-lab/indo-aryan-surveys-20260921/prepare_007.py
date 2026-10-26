import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
groups=[
dict(parent='f_xqxbqhxafwpsu',gloss=['knife','knife (to cut meat)'],forms=['cākū','caku','cākʰū','chakku','chokkū','cokū','tsakū','tsokū','tsoku'],citation='platts1884[s.v. ćāqū, p. 418];kannauji[p. 62]',evidence='Platts explicitly labels ćāqū “clasp-knife, penknife” Persian; the existing Hindi cākū attestation in the Kannauji survey supplies an attested regional donor with q adapted to k. The selected c/ts, aspiration, gemination and vowel variants preserve the whole knife word. This follows the previous review’s provisional Hindi/Urdu transmission policy, leaving local Gujarati, Pashto or other intermediaries unresolved; mixed slash responses and metathesized kācū are excluded.'),
dict(parent='f_cdzupphnjmodm',gloss=['fat','fat (of meat)'],forms=['cɛrvī','carbi','cɛrbī / cɛrvī','cerabī','cɛrbī','carbī'],citation='platts1884[s.v. ćarbī, p. 429];kannauji[p. 75]',evidence='Platts derives ćarbī “fat, grease, suet, tallow” from Persian ćarb; the existing Hindi carbī attestation supports the whole regional loan word. Pothwari cɛrvī and Gujari cɛrbī/cɛrvī retain this consonant sequence with b/v alternation; both members of the selected slash response belong to the same family. The borrowed link preserves the provisional regional Hindi/Urdu route rather than asserting direct Persian contact for each language.'),
dict(parent='f_cdko5w5xxg5qg',gloss=['hot','hot (water)','hot (weather)'],forms=['garam','garām'],citation='platts1884[s.v. garm, garam, p. 905];kannauji[p. 88]',evidence='Platts explicitly records Persian garm and vernacular garam “hot, warm”; the linked Hindi garam survey attestation is the existing regional donor. These whole adjectives are linked as regional loans with unresolved local transmission, not as direct descendants of cognate Sanskrit gharma. Responses in -i “heat” and combinations with the inherited tatta adjective are excluded.'),
dict(parent='f_dbuiao2qkdgv4',gloss=['whole','whole (unbroken)'],forms=['sābūt','sabūt'],citation='platts1884[s.v. s̤ubūt, s̤abūt, p. 368]',evidence='Platts gives vernacular s̤abūt and explicitly the adjective “entire”; the existing Hindi whole/entire entry supports these u-vowel responses as regional loans. The graph identifies the Hindi/Urdu donor hypothesis, without asserting direct Arabic contact. The a/i-vowel sābat/sābit family and sarva-derived sab forms are not silently folded into this donor.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['Gloss'] in g['gloss'] and r['Form'] in g['forms']:
   a.append(dict(record=r,parent=g['parent'],kind='borrowed',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass007-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass007-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
