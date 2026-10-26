"""Extend the research-only LSI Badaga checkpoint through residual record 75."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
decisions = json.loads((P / 'proposed-decisions.json').read_text())
assert len(decisions['accepted']) == 41 and len(decisions['held']) == 9

proposals = {
    53: (['d333'], 'DEDR 333 includes assent forms in the “become/be so” family (Tamil ām, Kannada āṃ, Kodagu a·ku); linked Badaga hā is explicitly glossed “yes, affirmation”, matching LSI hā despite the Badaga initial h.'),
    55: (['d2559'], 'DEDR 2559 gives Kannada illa “no, not”, beside Malayalam illa; LSI illei retains its survey final-vowel spelling.'),
    56: (['d2559'], 'DEDR 2559 directly gives Kannada illa “no, not”; the LSI form and response sense match.'),
    57: (['d4572'], 'DEDR 4572 directly gives Kannada hōgu “go”; the linked Badaga dictionary also has hōgu with this meaning.'),
    58: (['d4572'], 'DEDR 4572 gives Kannada pō/pōgu “go”; LSI po is the short survey response.'),
    59: (['d1835'], 'DEDR 1835 gives Kannada kuḷḷu/kuḷir “sit”; linked Badaga kuḷi “sit” belongs to this article, matching LSI kuli with the lateral mark omitted.'),
    60: (['d474'], 'DEDR 474 gives Kannada ibbattu/ippattu “twenty” in the two-numeral family; LSI ibbatta differs in final-vowel notation.'),
    61: (['d2826'], 'DEDR 2826 gives Kannada ay-vattu “fifty”; LSI eivattu is the survey rendering of the same numeral.'),
    62: (['d5160'], 'DEDR 5160 gives Kannada nān “I” and related first-person obliques; LSI nānu is the same nominative family.'),
    63: (['d5160'], 'DEDR 5160 gives Kannada nān “I”; LSI na is a shorter elicitation form, retained without adding an unwritten final nasal.'),
    64: (['d5160'], 'DEDR 5160 gives first-person en-/nan- oblique stems; LSI yenna “my” includes the en(n)- possessive base with a survey initial glide.'),
    65: (['d5154'], 'DEDR 5154 contains the eŋga/naŋga first-person-plural family; LSI aŋla “we” is a qualified stem comparison with a lateral plural ending. Inclusivity is not supplied by LSI.'),
    66: (['d5154'], 'DEDR 5154 gives Kodagu eŋga “we”; LSI yeŋla has the same eŋ- base with survey y- and plural ending. Inclusivity is not supplied by LSI.'),
    67: (['d5154'], 'LSI aŋa “our” preserves the aŋ- stem of its separate aŋla “we” response; DEDR 5154 covers the eŋga/naŋga first-person-plural oblique family.'),
    68: (['d5154'], 'LSI yeŋa “our” preserves the eŋ- stem of its separate yeŋla “we” response; DEDR 5154 explicitly gives eŋga and its oblique eŋga-.'),
    69: (['d3684'], 'DEDR 3684 gives Kannada nī(n)- “thou” with nin(n)- oblique; linked Badaga ninna is “your, thy” as in LSI.'),
    70: (['d3688'], 'DEDR 3688 gives Kannada nīngaḷ “you” plural; LSI nīŋla is a close Badaga survey plural form.'),
    71: (['d5270'], 'DEDR 5270 gives Kannada bar-/bā “come”; linked Badaga bā is the same imperative/elicitation stem.'),
    72: (['d3675'], 'DEDR 3675 directly gives Kannada nil “stand”; LSI nil has the same form and sense.'),
    73: (['d1942', 'd4572'], 'LSI keṭṭ hogu “die” has Kannada keṭṭa/keḍu “be spoiled, die” in DEDR 1942 plus hōgu “go” in DEDR 4572; ordered components preserve both verbs and the source’s metaphorical whole-phrase gloss.'),
    74: (['d3098'], 'DEDR 3098 explicitly gives Kannada tar/tā “give”, with a note that Badaga has the give-to-first-or-second-person sense; LSI ta is the bare elicited stem.'),
    75: (['d1041'], 'DEDR 1041 gives Kannada ōḍu “run”; LSI vōḍ has an initial survey glide and the same running sense.'),
}
holds = {
    51: 'LSI u “and” is a grammatical conjunction; the short spelling does not distinguish a native clitic from a borrowed one, and no safe lexical parent is established.',
    52: 'LSI āle “but” is homonymous with linked Badaga āle “house” and another āle “young”, neither semantically relevant; the conjunction needs separate evidence.',
    54: 'LSI haudu “yes” resembles a wider Kannada affirmative, but the exact derivation from a Dravidian verb or demonstrative is not established by the checked DEDR articles.',
}

assert set(proposals) | set(holds) == set(range(51, 76))
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
seen = {x['record']['ID'] for x in decisions['accepted'] + decisions['held']}
assert len(seen) == 75
(P / 'proposed-decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
(P / 'research-holds.json').write_text(json.dumps(decisions['held'], ensure_ascii=False, indent=2) + '\n')
(P / 'unexamined-inventory.json').write_text(json.dumps(records[75:], ensure_ascii=False, indent=2) + '\n')
parents = {part for items in decisions['accepted'] for part in items.get('components', [items['parent']])}
articles = {}
with gzip.open(P.parents[2] / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in parents | {'d3825', 'd1018'}:
            articles[row['Entry_ID']] = row
assert parents <= articles.keys()
(P / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
print('through75', len(decisions['accepted']), 'proposals', len(decisions['held']), 'holds', len(records) - 75, 'unexamined')
