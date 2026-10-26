"""Materialize review-only Tamil DravLex decisions from audited source rows."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'inventory.json').read_text())
raw = json.loads((P / 'source-record-evidence.json').read_text())
assert len(records) == len(raw) == 35

proposals = {
    4: ('d1448', 'The full article prints Ta. kātu “ear” and Malayalam kātu, Irula ka·du. Source kādə has the same kād/t stem and meaning, with a voiced medial stop in its survey transcription.'),
    7: ('d3682', 'The full article prints Ta. niṟai “become full, be replete” with past -nt-. Source nir̪ainta is the corresponding completed/past form; its inflection does not require a separate derived node.'),
    8: ('d4992', 'The full article prints Ta. mur̤u “all, entire, whole” and Malayalam mur̤u; source muḷu “full” matches this lateralized whole/full family.'),
    9: ('d2017', 'The full article begins Ta. kēḷ “hear, listen” with kēṭp-/kēṭṭ- stems; source kēḷə is the same hearing verb with a final survey vowel.'),
    11: ('d3419', 'The full article prints Ta. teri “be understood, know” and its -v- present stem. Source teriyum is an inflected continuation of that verb, not a separate etymon.'),
    14: ('d1366', 'The short full article prints Ta. kar̤uttu “neck, throat” and Malayalam kar̤uttu. Source kaḷuttə has the same neck stem with the survey’s final vowel and liquid notation.'),
    15: ('d2552', 'The full article explicitly lists Ta. iravu, ira, irā, rā “night”. Source irāvə and rāvə preserve the two vocalic shapes in this one family.'),
    16: ('d2552', 'The full article explicitly lists Ta. iravu, ira, irā, rā “night”. Source irāvə and rāvə preserve the two vocalic shapes in this one family.'),
    21: ('d664', 'The full article explicitly prints Ta. uruṇṭai “ball, anything round, roundness”, matching source urunḍai in form and sense.'),
    22: ('d2855', 'The full article prints Ta. col (colv-/colluv-, colli-) “say, speak, tell”. Source collə and sollə are ordinary survey variants of this verb.'),
    23: ('d2855', 'The full article prints Ta. col (colv-/colluv-, colli-) “say, speak, tell”. Source collə and sollə are ordinary survey variants of this verb.'),
    24: ('d4091', 'The full article begins Ta. pār (-pp-, -tt-) “see, look at, examine”; source pārə is the same verb with a final survey vowel. The article also contains a separate brahman sense, which is not claimed here.'),
    27: ('d1298', 'The full article prints Ta. kal “stone, pebble, boulder” and Malayalam kallu; source kallə directly matches the geminate stone form.'),
    29: ('d3687', 'The full article gives Ta. nīntu (nīnti-) “swim” and Malayalam nīntuka; source neentə is the same nīnt- swimming stem in survey transcription.'),
    30: ('d1', 'The whole distal-demonstrative article lists Ta. adjective anta and pronoun atu “that”. Source anta and adʰə are article-family forms even though the voiced/aspirated spelling of adʰə is not printed verbatim.'),
    31: ('d1', 'The whole distal-demonstrative article lists Ta. adjective anta and pronoun atu “that”. Source anta and adʰə are article-family forms even though the voiced/aspirated spelling of adʰə is not printed verbatim.'),
    32: ('d410', 'The whole proximal-demonstrative article lists Ta. itu “this” and adjective inta. Source idʰə continues the i- proximal family with a voiced/aspirated survey spelling.'),
    33: ('d5517', 'The full article explicitly gives Ta. vetuvetu “become lukewarm” and vetuvetuppu “lukewarmness”. Source vedəvedəppə matches the reduplicated warm form, with voiced survey stops.'),
    34: ('d5496', 'The full article begins Ta. veḷ/veṇ “white” and lists veḷḷai “whiteness”; source veḷḷə matches the white stem and gloss.'),
    35: ('d4635', 'The full article gives Ta. mañcaḷ “turmeric; yellow colour as that of turmeric” and cognates in Malayalam, Kannada, and Tulu. Source manjaḷ is the same yellow/turmeric word.'),
}

specific_holds = {
    1: ([], 'The source labels toppai “belly” self-identified but supplies no cognate or DEDR locator; a supported family was not verified.'),
    2: ([], 'The source flags nagam “claw” as a loan and cites a distant Indo-European comparison; its immediate donor into Tamil remains unspecified.'),
    3: (['d2426'], 'The source cites the cā/cett- “die” article for sattə, but sattə po is a two-word response with an unanalyzed po auxiliary; it is not treated as one inherited stem.'),
    5: ([], 'The source gives sāpədə “eat” without a DEDR locator; the colloquial stem and its derivation were not verified in a full primary article.'),
    6: (['d5173'], 'Tamil rekke “feather” lacks a cited DEDR parent. Article 5173 treats Telugu rēku “petal” and Kolami/Kuwi leaf forms, not a Tamil inherited feather noun; contact direction is unclear.'),
    10: ([], 'The source flags idayam “heart” as a loan without an immediate donor attestation.'),
    12: (['d3419'], 'terinjuko includes the teri- “know” stem but also a -ko element; its morphological composition and relation kind need analysis before attaching the whole response.'),
    13: ([], 'The source flags cantiran̪ “moon” as a loan; no immediate donor form was established.'),
    17: ([], 'The source flags rātri “night” as a loan; it is not grouped under inherited iravu/rā in DEDR 2552.'),
    18: ([], 'The source flags man̪itan̪ “person” as a loan; no immediate donor form was established.'),
    19: ([], 'The source flags manuṣan “person” as a loan; no immediate donor form was established.'),
    20: ([], 'The source flags vaṭṭam “round” as a loan; no immediate donor was verified despite superficially similar roundness words.'),
    25: ([], 'The source gives ukkārə “sit” without a DEDR locator; no supported family or donor was verified.'),
    26: ([], 'The source flags naccatiram “star” as a loan and compares Sanskrit nakṣatra, but its immediate donor was not identified.'),
    28: ([], 'The source flags sūryan “sun” as a loan; the immediate donor into Tamil was not verified.'),
}
assert set(proposals).isdisjoint(specific_holds)
assert set(proposals) | set(specific_holds) == set(range(1, 36))
accepted = []
held = []
for i, (record, evidence) in enumerate(zip(records, raw), 1):
    assert 'dravlex:' + evidence['ID'] in record['Source']
    if i in proposals:
        parent, note = proposals[i]
        assert evidence['Loan'] == 'false'
        accepted.append(dict(record=record, parent=parent, kind='reflex', citation=record['Source'] + ';dedr[' + parent[1:] + ']', evidence=note, status='pending-review'))
    else:
        candidates, reason = specific_holds[i]
        held.append(dict(record=record, candidates=candidates, reason=reason))
decisions = dict(status='pending-review', accepted=accepted, held=held)
(P / 'proposed-decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted': len(accepted), 'held': len(held)}))
