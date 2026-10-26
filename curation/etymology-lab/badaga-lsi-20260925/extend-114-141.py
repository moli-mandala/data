"""Finish initial research of the LSI Badaga residual source without saving links."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
decisions = json.loads((P / 'proposed-decisions.json').read_text())
assert len(decisions['accepted']) == 101 and len(decisions['held']) == 12

proposals = {
    114: (['d3986'], 'DEDR 3986(a) gives Kannada pal “tooth”; linked Badaga hallu “tooth, teeth” directly belongs to this article, matching the LSI form and sense.'),
    115: (['d4707'], 'DEDR 4707 gives Tamil mayir and neighbouring Kota/Toda mi·r “hair”; LSI Badaga meiru has the matching m-r hair stem, although the exact Badaga spelling is not printed in the article.'),
    116: (['d1892'], 'DEDR 1892 directly gives Kannada kūdal “hair”; linked Badaga kūdalu has the same form and meaning.'),
    117: (['d4494'], 'DEDR 4494 gives Kannada poṭṭe “belly”; linked Badaga hoṭte “stomach, belly” is assigned to that article. LSI hoṭṭĕ retains its aspirated survey spelling.'),
    118: (['d5488'], 'DEDR 5488 gives Kannada bennu “back”; linked Badaga bennu has the same body-part meaning.'),
    119: (['d1278'], 'DEDR 1278(a) gives Kannada kabbiṇa “iron” in the black/iron family; linked Badaga kabbuṇa “iron” matches LSI.'),
    120: (['d2596'], 'DEDR 2596 directly gives Kannada cinna “gold”; linked Badaga cinna “gold” confirms this LSI sense, distinct from homonymous cinna “small” in DEDR 2594.'),
    121: (['d5496'], 'DEDR 5496(a) gives Tamil veḷḷi “silver” and Kannada beḷḷi in the white/silver family; LSI belli is the survey form without retroflex notation.'),
    127: (['d1711'], 'DEDR 1711(a) gives Kannada kudure “horse”; the linked Badaga dictionary has kudure “horse, pony”, matching LSI.'),
    132: (['d2170'], 'DEDR 2170 directly gives Kannada kotti “male or female cat”; linked Badaga kotti has the cat sense, not the unrelated tuft/crest homonym.'),
    137: (['d4559'], 'DEDR 4559 gives Kannada hottu “time, sun”; linked Badaga hottu has the sun sense. The same spelling also has an unrelated “pod” homonym in DEDR 4491.'),
    138: (['d3213'], 'DEDR 3213 gives Kannada tiṅgaḷ “moon”; LSI tiggalu omits the written nasal, so this is a qualified family-level comparison alongside separate tiŋgalu.'),
    139: (['d3213'], 'DEDR 3213 gives Kannada tiṅgaḷ “moon”; LSI tiŋgalu matches the nasal stem with a survey final vowel.'),
    140: (['d1514', 'd2929'], 'LSI kiccu nerpu “fire” combines Kannada kiccu “fire” in DEDR 1514 with a nerpu form compared to Tamil neruppu “fire” in DEDR 2929; the second component lacks the dictionary geminate and remains qualified.'),
}
holds = {
    122: 'LSI vokaliga “cultivator” resembles a Kannada social/occupational label; no exact DEDR family or immediate donor node has been verified.',
    123: 'LSI āḍukāra “shepherd” appears built on goat āḍu (DEDR 5152) plus an occupational -kāra element, but the latter and the whole derivation remain unresolved.',
    124: 'LSI dēvaru “God” resembles a Sanskritic deva form; the immediate donor and graph parent are not established, so no remote Sanskrit leap is proposed.',
    125: 'LSI piśāci “Devil” is a learned-looking South Asian term, but the immediate donor attestation and path are not established.',
    126: 'LSI āle “House” matches a Badaga dictionary house term, but DEDR āle articles 382, 384 and 387 mean banyan, hail/water and sugar-cane press; no secure ancestry is established.',
    128: 'LSI klatte “Ass” is near Kannada katte and Toda katy in DEDR 1364, but its extra l is not explained by an attested Badaga comparator or source note.',
    129: 'LSI voṭṭe “Camel” lacks a matching DEDR animal term or established immediate donor.',
    130: 'Third LSI dana “Cow” record retained separately; no secure Badaga cow etymon or immediate donor was established for the same source form earlier.',
    131: 'Third LSI hasu “Cow” record retained separately; linked Badaga hasu “hunger” (DEDR 3825) is a different homonym and not cow evidence.',
    133: 'LSI hakkilu “Bird” matches linked Badaga dictionary bird forms, but those point to unlinked akki variants rather than a validated etymon; immediate ancestry is unresolved.',
    134: 'LSI hūva “Cock” has no secure DEDR bird family match or immediate donor.',
    135: 'LSI hunja “Cock” lacks a verified DEDR match or immediate donor; do not conflate it with different bird terms merely by the gloss.',
    136: 'LSI sime batu “Duck” is a multiword response, and neither lexical element has a verified etymon or immediate donor in the checked articles.',
    141: 'LSI penki “Fire” has no secure DEDR match; its resemblance to other regional fire terms is insufficient to assign an article.',
}

assert set(proposals) | set(holds) == set(range(114, 142))
for i, (parents, evidence) in proposals.items():
    record = records[i - 1]
    row = dict(record=record, parent=parents[0], kind='component' if len(parents) > 1 else 'reflex',
               citation=record['Source'] + ''.join(f';dedr[{p[1:]}]' for p in parents),
               evidence=evidence, comparison='Full DEDR article, linked Badaga comparanda and exact survey record checked.',
               status='pending-review')
    if len(parents) > 1:
        row['components'] = parents
    decisions['accepted'].append(row)
decisions['held'] += [dict(record=records[i - 1], candidates=[], reason=reason) for i, reason in holds.items()]
order = {r['ID']: i for i, r in enumerate(records)}
decisions['accepted'].sort(key=lambda x: order[x['record']['ID']])
decisions['held'].sort(key=lambda x: order[x['record']['ID']])
assert len({x['record']['ID'] for x in decisions['accepted'] + decisions['held']}) == 141
(P / 'proposed-decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
(P / 'research-holds.json').write_text(json.dumps(decisions['held'], ensure_ascii=False, indent=2) + '\n')
(P / 'unexamined-inventory.json').write_text('[]\n')
parents = {part for items in decisions['accepted'] for part in items.get('components', [items['parent']])}
articles = {}
with gzip.open(P.parents[2] / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in parents | {'d3825', 'd1018', 'd2594', 'd4491', 'd1364', 'd382', 'd384', 'd387'}:
            articles[row['Entry_ID']] = row
assert parents <= articles.keys()
(P / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
print('complete', len(decisions['accepted']), 'proposals', len(decisions['held']), 'holds', sum(len(x.get('components', [x['parent']])) for x in decisions['accepted']), 'rows')
