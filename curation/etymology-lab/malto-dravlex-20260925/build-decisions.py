"""Materialize this review-only Malto checkpoint from audited source records."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
rows = json.loads((P / 'inventory.json').read_text())
raw = json.loads((P / 'source-record-evidence.json').read_text())
assert len(rows) == len(raw) == 63

proposals = {
    13: ('d1458', 'The source explicitly compares āikā with Malto qáye “become dry” and Kurux xaikā “dry”; the full DEDR article prints both. Its initial-vowel and final-vowel differences remain visible in the source forms.'),
    25: ('d5516', 'The full article gives Malto mene (menj-) “hear, ask”, Kurux mennā, and Central Dravidian ven- reflexes. Source mēna preserves the m-n frame and exact sense, with vowel length differing from the older citation.'),
    26: ('d698', 'The full article lists Malt. ? ugli “heart, mind”; source huglī has the same u-gl-i sequence with initial h and long final vowel. The article marks ugli with a question mark, so the family affiliation remains explicitly tentative.'),
    54: ('d1', 'The whole demonstrative article lists Malto áh/ár/áth, adjective á, for distant “that”, and discusses initial h in related forms. Source ādu and hād both belong to the distal-demonstrative series, though neither exactly repeats the cited inflection.'),
    55: ('d1', 'The whole demonstrative article lists Malto áh/ár/áth, adjective á, for distant “that”, and discusses initial h in related forms. Source ādu and hād both belong to the distal-demonstrative series, though neither exactly repeats the cited inflection.'),
    59: ('d2654', 'The source annotation explicitly quotes Malt. kuṛni “warm, hot”; the full DEDR article places it with Malt. kuṛe “burn” and Kurux kuṛnā “grow warm”. Source kuḍni differs in the medial retroflex but matches the ku-?ni adjective and sense.'),
    60: ('d187', 'The source quotes Malt. amu “water” but misprints its locator as DED 18. Full DEDR 187 contains Malt. amu and Tamil am/ām “water”; modern hamdu retains am- with initial h and added d, so this is a qualified family proposal with the corrected article number.'),
}
specific_holds = {
    3: ('d4411', 'DravLex groups bēḍo with Telugu pedda and other “big” terms; DEDR 4411 has those comparanda but no Malto reflex. The b-/ḍ form could reflect contact rather than a secure inherited reflex.'),
    6: ('d4750', 'DEDR 4750 has Malt. maṛg̣ro “black”, but the source transcribes maḍharo and its annotation calls this an apparent singleton. The medial sequence and derivation are unresolved.'),
    10: (None, 'The source explicitly flags badlī as a loan and suggests Hindi bādal; a documented immediate donor for this exact form was not verified.'),
    11: ('d2426', 'The source annotation cites DEDR 2426, which contains Malt. keye “die”, but keəānākeəidāh is an undelimited multi-part response. The parts and their individual relationships need clarification.'),
    12: (None, 'The source calls kucco a strong lookalike to Brahui kučak, but gives no DEDR entry or borrowing direction; DEDR 1796 concerns a dog call, not this noun.'),
    15: (None, 'The source flags bhumi as a loan and cites Sanskrit bhūmi; the immediate donor into Malto is not established.'),
    22: ('d848', 'DEDR 848 gives Malt. éṛu “good”, whereas source raḍu differs at the onset and medial consonant; no supported derivation bridges them.'),
    24: ('d2023', 'DEDR 2023 gives Malt. qeqe “hand”; source ṭeṭu is distinct in both consonants and is not explicitly assigned to this article.'),
    27: ('d4044', 'The source cites DEDR 4044 for bajpo “kill”, but the full article has no Malt. reflex and its Pa./Kui striking forms do not establish the b-j stem.'),
    28: ('d17', 'The source cites DEDR 17, which gives Malt. áge “know”; hāhpo has a different stem shape and no demonstrated derivation.'),
    37: (None, 'This response combines bēḍo “big” with pahāḍi “mountain/hill”; its components need separate donor and family analysis.'),
    38: (None, 'The source flags nāmi as a loan and compares Sanskrit nāman-, but no immediate donor has been verified.'),
    40: (None, 'The source explicitly identifies rāti as a loan; no immediate donor form has been established.'),
    49: ('d1407', 'DEDR 1407 gives Malt. qandre “sleep”, close to source andarpo, but the source response andarpo kōḍa is multiword and kōḍa remains unanalyzed.'),
}
accepted = []
held = []
for i, (record, evidence) in enumerate(zip(rows, raw), 1):
    assert record['Source'].find('dravlex:' + evidence['ID']) >= 0
    if i in proposals:
        parent, note = proposals[i]
        accepted.append(dict(record=record, parent=parent, kind='reflex',
                             citation=record['Source'] + ';dedr[' + parent[1:] + ']',
                             evidence=note, comparison='Full DEDR article and source annotation checked.',
                             status='pending-review'))
    else:
        candidates, reason = specific_holds.get(i, (None, 'The source annotation and homepage search did not establish a supported DEDR family or an immediate donor for this form.'))
        if evidence['Loan'] == 'true' and i not in specific_holds:
            reason = 'The source explicitly flags this as a loan, but the immediate donor into Malto was not verified.'
        held.append(dict(record=record, candidates=[candidates] if candidates else [], reason=reason))

decisions = dict(status='pending-review', accepted=accepted, held=held)
assert len(accepted) + len(held) == 63
(P / 'proposed-decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted': len(accepted), 'held': len(held)}))
