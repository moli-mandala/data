"""Second researched pass, including exact primary-page resolution of SSNP j."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
groups=[
dict(parent='5228',gloss='tongue',languages=['awan','srk'],forms=['yīb'],citation='CDIAL[5228];rensch-hallberg-oleary1992[p. 212, item 10]',evidence='The printed Hindko appendix p. 212, item 10, was visually checked in the SIL scan: the form is ǰib, not a palatal-approximant yib. Its legacy text extraction is jib; the current ssnp conversion maps plain j to y. CDIAL jihvā explicitly lists Awankari jībh and other regional jib/jībh forms. Link the original attestation to this family, retaining current Form yīb and flagging the transcription repair separately; no y-to-j sound change or source correction is asserted here.'),
dict(parent='5228',gloss='tongue',languages=['Goj'],forms=['yīb','yīv','yīv / yīb'],citation='CDIAL[5228];rensch-hallberg-oleary1992[p. 260, item 10]',evidence='The Gujari appendix p. 260, item 10, was visually checked in the SIL scan. It prints ǰib in KUN/CHT/DIR/SSW/TSW/GLT/KGH/SHZ/NAK/GJW, ǰiv in SAK, and ǰiv / ǰiβ in CAK. The current y onset and CAK final b are legacy-font conversion issues. Both CAK alternatives continue the same jihvā family: compare CDIAL Prakrit jibbhā, regional jib/jībh and Kashmiri zev. The slash response is analysed in full, not split into separate records. Preserve current spellings and qualify regional transmission.'),
dict(parent='9758-2',gloss='fish',languages=['awan'],forms=['macī','macʰī','maccī'],citation='CDIAL[9758,2]',evidence='Turner explicitly assigns Lahnda Awankari macchī to section 2 *matsiya, rather than the main matsya branch (which also contains feminine fish forms in other languages). These Awankari responses are the same i-final fish word, with source gemination and aspiration differences retained. The same-language comparator resolves the earlier section hold; this decision is not automatically extended to neighbouring languages.'),
dict(parent='1900',gloss='get up',forms=['uṭʰ'],citation='CDIAL[1900,1]',evidence='Full *ut-sthāti section 1 gives Prakrit uṭṭhaï, Nepali uṭhnu and the eastern Hindi uṭh-/ūṭh- rise verb. These bare uṭh responses are imperatives/stems of “get up”, not the past participle *ut-sthita (1907). Responses containing negative particles, auxiliaries or perfect endings are excluded from this rule.'),
dict(parent='1907',gloss='get up',languages=['Dotyali'],forms=['uṭʰyo','utʰyo'],citation='CDIAL[1907,1]',evidence='Full *ut-sthita section 1 gives Prakrit uṭṭhiya, Hindi uṭhā “stood up” and Old Marwari ūṭhyo. Dotyali uṭhyo is the matching perfect form, so the inherited participial node 1907 is selected rather than the present stem 1900. The source dental/retroflex notation is retained.'),
dict(parent='997',gloss='mother',languages=['M'],forms=['āy'],citation='CDIAL[997]',evidence='Turner explicitly gives Marathi āī “mother” in the *āī nursery-word family. Survey āy represents the same final glide; the meaning is exact. No claim about an Indo-European root or the competing Dardic āryikā analysis is made for this Marathi form.'),
dict(parent='12629',gloss='broom',languages=['kul'],forms=['śun'],citation='CDIAL[12629]',evidence='The śodhana article and addendum give Western Pahari Kotgarhi śv̄̃ṇ and Kiunthali śū̃hṇ “broom”, beside Jaunsari śuain and Hindi sonī. Kullui śun belongs to this geographically and semantically specific contracted broom family; source absence of nasal-vowel and aspiration marking is retained.'),
dict(parent='13551',gloss='needle',languages=['Bhili','Bhilali','PauriBareli','Noiri','DungraBhili'],forms=['hūī','hūi','hūe','huye'],citation='CDIAL[13551,1]',evidence='Full sūcī section 1 gives Prakrit sūī, Hindi sūī, Marathi sūī and Gujarati soy “needle”. The selected Bhil hūī/hūi/hūe/huye forms retain the contracted vowel/glide sequence with regional s-to-h development; they do not retain the medial affricate/nasal defining section 2 *sūñcī. Regional contact versus inheritance is not resolved by the family link.'),
]
acc=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['Gloss']==g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   acc.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
held=[]
for r in rs:
 if r['Gloss']=='wet' and r['Form'] in {'sijyā','sijā','sijʸā'}:
  held.append(dict(record=r,candidates=['13393','13933'],reason='CDIAL explicitly cross-references sicyate (13393) and svidyati (13933) for Lahnda/Punjabi sijj- “wet”; the Pothwari sijā/sijyā forms do not distinguish them. Retain both candidates rather than choosing from similarity.'))
 if r['Gloss']=='wet' and r['Form'] in {'sinna','sinnā','sina','sinə','sino','siy','siyi','siyī','siya'}:
  held.append(dict(record=r,candidates=['13798','13934'],reason='Full CDIAL snih (13798) and svinna (13934) explicitly offer competing derivations of regional sinnā/sinnhā “wet”. h loss, contraction or a same-gloss match does not resolve the historical parent.'))
assert len(acc)==len({x['record']['ID'] for x in acc})
(P/'pass002-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass002-decisions.json').write_text(json.dumps({'accepted':acc,'held':held},ensure_ascii=False,indent=1)+'\n')
print('Selected',len(acc),'records; held',len(held),[(g['parent'],sum(x['family']==i for x in acc)) for i,g in enumerate(groups)])
