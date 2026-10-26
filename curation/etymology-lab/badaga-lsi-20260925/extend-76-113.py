"""Research-only analysis of the LSI Badaga person and verb paradigms."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
decisions = json.loads((P / 'proposed-decisions.json').read_text())
assert len(decisions['accepted']) == 62 and len(decisions['held']) == 13

pronouns = {
    'nā': ('d5160', 'first-person singular nā-/nān- in DEDR 5160'),
    'nānu': ('d5160', 'first-person singular nā-/nān- in DEDR 5160'),
    'nāvŭ': ('d3647', 'Kannada nāvu “we” in DEDR 3647'),
    'ni': ('d3684', 'second-person singular nī-/nin- in DEDR 3684'),
    'ava': ('d1', 'remote third-person ava- in DEDR 1'),
    'avu': ('d1', 'remote third-person av- in DEDR 1'),
    'avaka': ('d1', 'remote third-person av- in DEDR 1'),
    'yeŋla': ('d5154', 'first-person plural eŋga/naŋga family in DEDR 5154'),
    'nīŋla': ('d3688', 'second-person plural nīngaḷ family in DEDR 3688'),
    'niŋla': ('d3688', 'second-person plural nīngaḷ family in DEDR 3688'),
    'nīŋal': ('d3688', 'second-person plural nīngaḷ family in DEDR 3688'),
}

accepted = []
for i in range(76, 114):
    record = records[i - 1]
    words = record['Form'].split()
    if i == 88:
        parents = ['d4534']
        evidence = 'Linked Badaga hui “beat, strike” is assigned to DEDR 4534, whose Kannada puy/poy means “beat”; the identical LSI hui has that sense.'
    elif i >= 108:
        parents = ['d3688' if i == 108 else 'd1']
        if i == 108:
            evidence = 'LSI niŋa “your” is a possessive response from the nīŋ- second-person-plural family in DEDR 3688; the ending is retained.'
        else:
            evidence = f'LSI {record["Form"]} “{record["Gloss"]}” preserves the remote third-person av- demonstrative/pronoun family in the full DEDR 1 article; case and plural endings remain in the survey form.'
    else:
        assert len(words) == 2, (i, words)
        pkey = words[0].lower()
        assert pkey in pronouns, (i, pkey)
        pron_parent, pron_ev = pronouns[pkey]
        verb_parent = 'd480' if i <= 87 else 'd4534' if i <= 95 else 'd4572'
        verb_ev = {
            'd480': 'Kannada ir-/idd- “be” in DEDR 480',
            'd4534': 'Kannada poy/puy “beat” and linked Badaga hui in DEDR 4534',
            'd4572': 'Kannada hōgu/pō “go” in DEDR 4572',
        }[verb_parent]
        parents = [pron_parent, verb_parent]
        evidence = (f'LSI {record["Form"]} “{record["Gloss"]}” has ordered pronoun + verb components: '
                    f'{pron_ev}; {verb_ev}. Person/tense inflection is retained without inventing suffix etyma.')
        if i in {96, 97, 98, 100, 101}:
            evidence += ' The survey verb stem differs from the cited Kannada head and is a family-level paradigm comparison.'
        if i in {79, 85, 92, 105}:
            evidence += ' The eŋ-/yeŋ- plural stem is compared with the whole DEDR 5154 family; LSI does not specify clusivity.'
    row = dict(record=record, parent=parents[0], kind='component' if len(parents) > 1 else 'reflex',
               citation=record['Source'] + ''.join(f';dedr[{p[1:]}]' for p in parents),
               evidence=evidence, comparison='Exact LSI response, linked Badaga comparison, and full DEDR articles checked.',
               status='pending-review')
    if len(parents) > 1:
        row['components'] = parents
    accepted.append(row)

decisions['accepted'] += accepted
assert len({x['record']['ID'] for x in decisions['accepted'] + decisions['held']}) == 113
(P / 'proposed-decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
(P / 'unexamined-inventory.json').write_text(json.dumps(records[113:], ensure_ascii=False, indent=2) + '\n')
parents = {part for items in decisions['accepted'] for part in items.get('components', [items['parent']])}
articles = {}
with gzip.open(P.parents[2] / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in parents | {'d3825', 'd1018'}:
            articles[row['Entry_ID']] = row
assert parents <= articles.keys()
(P / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
print('through113', len(decisions['accepted']), 'proposals', len(decisions['held']), 'holds', len(records)-113, 'unexamined')
