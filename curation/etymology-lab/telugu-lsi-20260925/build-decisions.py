"""Materialize the first review-only Telugu LSI checkpoint."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'inventory.json').read_text())
source = json.loads((P / 'source-record-evidence.json').read_text())
assert len(records) == len(source) == 133

father = ('d3067', 'The full article gives Te. taṇḍri “father”. The LSI taṇḍrĭ/taṇḍrŭ forms with -ki, plural -lu, genitive -la, and allative -ku are ordinary case or number forms of this same father stem, not independent derived etyma.')
bull = ('d815', 'The full article explicitly gives Te. eddu (plural eḍlu) “ox, bullock, bull”. LSI yeddŭ and plural yeḍlŭ reproduce both stems with initial y in the survey spelling.')
deer = ('d2504', 'The full article gives Te. jiṅka “antelope” and Kannada jiṅke. LSI jiŋkă and plural jiŋkă-lŭ match the noun, though the survey’s “male deer” gloss is narrower than the printed generic antelope sense.')
question = ('d5151', 'The full interrogative article prints Te. evãḍu “who/which man”, ēmi “what”, and enduku “why”. The LSI yevaḍŭ, yēmĭ, and yendukŭ are these article-family forms with survey initial y and final reduced vowels.')
proposals = {
    1: ('d784', 'The full article explicitly prints Te. enimidi “eight”; LSI yenimidi matches it with initial y in the survey spelling.'),
    2: father,
    3: father,
    6: father,
    7: father,
    8: father,
    10: ('d3136', 'The full article prints Te. tal(l)i “mother” beside Tamil taḷḷai and Malayalam taḷḷa. LSI tallĭ is the same noun.'),
    19: bull,
    20: bull,
    21: ('d334', 'The full article gives Te. āvu “cow” with old plural ālu. LSI āvŭ-lŭ is a transparent modern plural of that cow noun; the plural ending is inflectional.'),
    28: ('d5087', 'The full article explicitly gives Te. mēka “goat” alongside Kannada mēke and Kolami me·ke. LSI mēkă-lŭ adds the ordinary plural ending to that noun.'),
    29: deer,
    31: deer,
    38: ('d796', 'The full article gives Te. ettu “height, elevation” and the adjective “high, elevated”. LSI yettŭ matches this high/height form with initial y in survey spelling.'),
    39: ('d4841', 'The full article explicitly gives Te. mī̃diki “up, upwards” within the mī- upper/top family. LSI mīdikĭ is the same directional form with nasalization omitted in the survey spelling.'),
    43: ('d5020', 'The full article gives Te. mundaṭa(n) “in front, before” in the mundu “front” family. LSI mundaṭă directly matches the adverb.'),
    44: ('d5488', 'The full article gives Te. ven(u)ka “back, rear, behind” and related back terms. LSI venukă matches the directional form directly.'),
    45: question,
    46: question,
    47: question,
    50: ('d333', 'The full article gives Te. avunu “yes” in the agu/avu “be, become” family. LSI avunŭ matches the cited affirmative form.'),
}
component_proposals = {
    15: (['d4616', 'd1711'], 'The first article gives Te. moga “male”; the second gives Te. guṟṟamu “horse”. LSI singular and plural responses express male + horse in that order; -lu is ordinary plural inflection. The horse article notes a possible Indo-Aryan origin, which this analysis does not resolve.'),
    17: (['d4616', 'd1711'], 'The first article gives Te. moga “male”; the second gives Te. guṟṟamu “horse”. LSI singular and plural responses express male + horse in that order; -lu is ordinary plural inflection. The horse article notes a possible Indo-Aryan origin, which this analysis does not resolve.'),
    22: (['d4616', 'd1796'], 'The full articles give Te. moga “male” and Te. kukka “dog”. LSI singular and plural responses combine those stems in source order, with -lu as ordinary plural inflection.'),
    24: (['d4616', 'd1796'], 'The full articles give Te. moga “male” and Te. kukka “dog”. LSI singular and plural responses combine those stems in source order, with -lu as ordinary plural inflection.'),
    26: (['d4616', 'd5087'], 'The full articles explicitly give Te. moga “male” and Te. mēka “goat”. LSI mogă mēkă combines those two stems in source order.'),
}

holds = {
    4: (['d3067'], 'The father stem is identifiable, but taṇḍrĭ daggirĭ nuñcĭ is a three-part “from a father” phrase; daggirĭ and nuñcĭ need separate analysis.'),
    5: (['d3067'], 'iddarŭ taṇḍrŭ-lŭ combines the numeral “two” with an inflected father noun; the whole response is not one father reflex.'),
    9: (['d3067'], 'taṇḍrŭ-la daggirĭ nuñcĭ is a multi-part “from fathers” phrase; the spatial and ablative elements need separate analysis.'),
    11: ([], 'maniśĭ “man” has a possible Indo-Aryan route, but no supported immediate donor or full DEDR family was established.'),
    12: ([], 'No full DEDR article directly supports LSI āḍadĭ “woman”; its relation to gender modifier āḍă and any donor remain unresolved.'),
    13: ([], 'kumārtĕ “daughter” may be a loan or remodelled kinship term; a supported immediate donor was not established.'),
    14: ([], 'kumārtĕ-lŭ is a plural of the daughter term, whose underlying family or immediate donor remains unresolved.'),
    16: (['d1711'], 'āḍă gurramŭ means “mare”; DEDR 1711 gives the horse noun but the female modifier needs independent analysis.'),
    18: (['d1711'], 'āḍă gurramŭ-lŭ combines gender modifier, horse noun, and plural ending; no complete component analysis is ready.'),
    23: (['d1796'], 'āḍă kukkă “bitch” is a two-part response. DEDR 1796 gives Te. kukka “dog”, but the female modifier needs its own supported analysis.'),
    25: (['d1796'], 'āḍă kukkă-lŭ combines the female modifier, dog noun, and plural; no complete component analysis is ready.'),
    27: (['d5087'], 'āḍă mēkă “female goat” is a two-part response; the female modifier has not been sourced even though DEDR 5087 prints mēka “goat”.'),
    30: ([], 'irrĭ “female deer” does not match the DEDR 2504 jiṅka antelope noun; a separate animal or gender term remains unidentified.'),
    32: ([], 'mañcivāḍŭ “good man” combines adjective and masculine noun; a secure parent for the full expression was not established.'),
    33: ([], 'mañcivārŭ “good men” combines adjective and plural masculine noun; a secure full component analysis was not established.'),
    34: ([], 'mañcidĭ “good woman” combines adjective and feminine demonstrative/pronominal material; its components remain unresolved.'),
    35: ([], 'mañcĭ strī-lŭ “good women” combines an adjective and a plural word for women; donor and component analysis remain unresolved.'),
    36: ([], 'ceḍḍă cinnavāḍŭ “bad boy” is a multi-part adjectival response with no complete component etymology.'),
    37: ([], 'ceḍḍă āḍapillă “bad girl” is a multi-part adjectival response with no complete component etymology.'),
    40: ([], 'kindikĭ “down” is a directional form, but no matching full DEDR article or securely sourced base was verified.'),
    41: ([], 'daggiră “near” lacks a verified full DEDR family or immediate donor.'),
    42: ([], 'dūramŭ “far” may involve Indo-Aryan dūra, but its immediate donor into Telugu was not verified.'),
    48: (['d410'], 'LSI iŋkă is glossed “and”; the similarly spelled DEDR 410 forms belong to the proximal demonstrative and do not establish this conjunction.'),
    49: ([], 'gānĭ “but” has no verified DEDR family or immediate donor; superficially similar kāni in DEDR 333 means “bad, wrong”.'),
}
assert set(proposals).isdisjoint(holds)
assert set(component_proposals).isdisjoint(set(proposals) | set(holds))
assert set(proposals) | set(component_proposals) | set(holds) == set(range(1, 51))
accepted = []
held = []
for i, record in enumerate(records[:50], 1):
    assert record['Source'].split(';')[0].startswith('grierson-lsi1928')
    if i in proposals:
        parent, note = proposals[i]
        accepted.append(dict(record=record, parent=parent, kind='reflex', citation=record['Source'] + ';dedr[' + parent[1:] + ']', evidence=note, status='pending-review'))
    elif i in component_proposals:
        components, note = component_proposals[i]
        citation = record['Source'] + ''.join(';dedr[' + x[1:] + ']' for x in components)
        accepted.append(dict(record=record, parent=components[0], components=components, kind='component', citation=citation, evidence=note, status='pending-review'))
    else:
        candidates, reason = holds[i]
        held.append(dict(record=record, candidates=candidates, reason=reason))
(P / 'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review', accepted=accepted, held=held), ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted': len(accepted), 'held': len(held), 'unexamined': len(records) - 50}))
