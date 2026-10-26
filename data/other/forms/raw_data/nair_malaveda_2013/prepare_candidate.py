"""Complete source-local Malaveda proposal; never writes canonical files."""
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'dialects.py').exists())
sys.path.insert(0, str(DATA))
from dialects import dialect_tag
from tags import GRAMMATICAL_TAGS, GENDER_TAGS
VALID_TAGS = GRAMMATICAL_TAGS | GENDER_TAGS

SOURCE = 'nair2013malaveda'
LECT = dialect_tag('Malaveda', 'nair2013-malaveda', 'Veda (Nair 2013; locality unresolved)')
SCOPED_TAGS = {
    'p41:accusative-examples:w2': (['2pl'], 'imma is the accusative example of the explicit second-person plural imm- base on p40.'),
    'p41:genitive-examples:w1': (['3pl', 'poss'], 'The printed gloss their supplies third-person plural possession; no gender is specified.'),
    'p41:genitive-examples:w2': (['kinship'], 'The printed sister genitive is an explicitly glossed kinship example.'),
    'p41:genitive-examples:w4': (['2sg', 'poss'], 'iññ-aṭe uses the explicit second-person singular oblique base iññu on p40; of you is a possessive genitive.'),
    'p41:genitive-examples:w5': (['3sg', 'f', 'poss'], 'The printed gloss of her supplies third-person singular feminine possession.'),
    'p41:locative-examples:w1': (['kinship'], 'The segmented sister locative is an explicitly glossed kinship example.'),
    'p41:locative-examples:w2': (['kinship'], 'The arrowed surface sister locative retains the same printed kinship gloss scope.'),
    'p41:locative-examples:w7': (['3sg', 'f'], 'The printed gloss in her supplies third-person singular feminine; locative remains distinct from possession.'),
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = 'f4d4c82f6e33728e80ba3a3746c603d648a2d75511e3042ad43d80911363d1e4'
assert sha(HERE / 'physical-units-first-reading.jsonl') == expected
units = [json.loads(line) for line in (HERE / 'physical-units-first-reading.jsonl').read_text().splitlines()]
rows, audit = [], []
symbol_inventory = {}
for unit in units:
    page = unit['printed_and_pdf_page']
    block = unit['unit_key'].split(':', 2)[2]
    for witness in unit['witnesses']:
        is_target = witness['disposition'] == 'target'
        record = {'source_unit_key': unit['unit_key'], 'printed_page': page, 'pdf_page': page,
                  'witness': witness, 'source_claim': unit['source_claim'],
                  'status': 'proposed' if is_target else 'excluded',
                  'exclusion_reason': '' if is_target else ('documentary community label' if witness['disposition'] == 'documentary' else 'explicit comparison control excluded by survey policy'),
                  'language_mapping': 'Malaveda source-named base; no independently established Glottocode; no inferred item-level locality' if is_target else witness['language_scope'],
                  'review': 'owner original-page reread and native-text crosscheck; independent output audit pending',
                  'uncertainty': []}
        if is_target:
            form = witness['literal_first_reading']
            assert unicodedata.is_normalized('NFC', form)
            tags = witness['provisional_tags'].split()
            scoped = SCOPED_TAGS.get(witness['witness_key'].split(':', 1)[1])
            if scoped:
                tags.extend(scoped[0])
                record['scoped_grammar_evidence'] = scoped[1]
            assert set(tags) <= VALID_TAGS, set(tags) - VALID_TAGS
            gloss = witness['printed_gloss']
            # Case and number labels are structured, not lexical definitions.
            gloss = re.sub(r'\s*\((?:Acc\.?|Dat\.?|Pl\.?)\)', '', gloss)
            gloss = re.sub(r'\s+Dat\.?$', '', gloss).strip()
            if 'uncertain' in tags:
                record['uncertainty'].append({'type': 'source_gloss_or_grammar', 'reason': witness['note'] or unit['source_claim']})
            if block == 'third-person':
                tags.append('uncertain')
                record['uncertainty'].append({'type': 'source_deixis', 'reason': unit['source_claim']})
            if block == 'personal-terminations' and not witness['note'] and 'n' in tags:
                tags.append('3sg' if 'sg' in tags else '3pl')
                tags = [t for t in tags if t not in {'sg', 'pl'}]
            notes = witness['note']
            if block == 'conditional':
                record['uncertainty'].append({'type': 'source_grammar_gloss_conflict', 'reason': unit['source_claim']})
            row = ['Malaveda', '', form, gloss, '', '', notes,
                   f'{SOURCE}[p. {page}, {block}, {witness["witness_key"].rsplit(":", 1)[1]}]',
                   '', '', witness['witness_key'], '', '', '', ' '.join(dict.fromkeys(tags + [LECT]))]
            if block == 'copula' and witness['witness_key'].endswith(':w1'):
                row[9] = 'Nair conjectures that Veda aata may be related to Malayalam –aak and aaṇ. No exact etymon, derivation or donor relation is established.'
                record['uncertainty'].append({'type': 'source_etymological_conjecture', 'reason': row[9]})
                row[14] = ' '.join(dict.fromkeys(tags + ['uncertain', LECT]))
            if block == 'lexical-list' and form in {'meeri', 'tali'}:
                row[9] = f'Nair compares this form with Malayalam {"maari" if form == "meeri" else "tall"}; no exact donor or etymon is asserted.'
            assert len(row) == 15
            rows.append(row)
            record['entry_key'] = row[10]
            record['proposed_row'] = row
            for char in form:
                symbol_inventory[char] = symbol_inventory.get(char, 0) + 1
        audit.append(record)
    if not unit['witnesses']:
        audit.append({'source_unit_key': unit['unit_key'], 'printed_page': page, 'pdf_page': page,
                      'status': 'documentary_only', 'source_claim': unit['source_claim'],
                      'reason': 'phonological prose or reported bibliography, no printed form witness'})
assert len(units) == 57 and len(rows) == 223
assert len({r[10] for r in rows}) == 223
assert sum(r['status'] == 'excluded' for r in audit) == 15
by_key = {r[10]: r for r in rows}
graph = []
def propose(page, block, child, parent, reason):
    ck, pk = f'{SOURCE}:p{page}:{block}:w{child}', f'{SOURCE}:p{page}:{block}:w{parent}'
    graph.append({'child_key': ck, 'child_form': by_key[ck][2], 'variant_of_key': pk,
                  'head_form': by_key[pk][2], 'printed_page': page, 'source_reason': reason,
                  'status': 'proposed; independent relation review required'})
propose(41, 'dative-free-variants', 2, 1, 'Source explicitly calls ennaaru and enne free variants.')
propose(41, 'locative-markers', 2, 1, 'Source u/ē slash pair under free variation statement; retain pair scope.')
propose(41, 'locative-markers', 4, 3, 'Source –ilu/ī slash pair under free variation statement; retain pair scope.')
for child, parent in [(4, 3), (7, 6), (9, 8), (11, 10), (13, 12), (15, 14), (16, 14)]:
    propose(42, 'personal-terminations', child, parent, 'Comma-separated alternates in one personal termination cell; row II (Mas.) typo remains uncertain, not repaired.')
propose(42, 'imperative-plural-markers', 2, 1, 'Source gives imperative plural suffixes -mi- or -miiru.')
propose(42, 'imperative-plural-examples', 2, 1, 'Corresponding plural imperatives printed together under that alternating-suffix scope.')
assert len(graph) == 12
review = HERE / 'independent-review/independent-original-output-review-pass1.json'
assert sha(review) == '7d9cee33549a6d62220d643844e8d5c5dd0a38f08348e23918a81982825ec762'
independent = json.loads(review.read_text())
assert independent['graph_review']['source_supported'] == 12
for edge in graph:
    child, parent = by_key[edge['child_key']], by_key[edge['variant_of_key']]
    assert child[2] == edge['child_form'] and parent[2] == edge['head_form']
    assert child[0] == parent[0] and child[10] != parent[10]
    child[11] = parent[10]
    edge['status'] = 'source-explicit variant accepted after independent original review'
    for record in audit:
        if record.get('entry_key') == child[10]:
            record['accepted_variant'] = edge
with (HERE / 'proposed.csv').open('w', newline='') as stream:
    csv.writer(stream).writerows(rows)
(HERE / 'proposal-audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
(HERE / 'graph-proposal.json').write_text(json.dumps({'proposals': graph,
    'held_non_edges': ['Segmented-to-surface locative arrows are morphological analyses, pending relation-type evidence.',
                      'Two negative participle examples share a gloss but are not expressly called free variants.',
                      'Gender/case/tense suffix lists are not all automatically variant links.',
                      'Cross-language copula, numeral and lexical comparisons assert no exact donor or etymon.'],
    'applied_edges': 12, 'independent_review_sha256': sha(review)}, ensure_ascii=False, indent=2) + '\n')
(HERE / 'symbol-inventory.json').write_text(json.dumps({'conversion': False,
    'reason': 'Source uses mixed literal Roman typesetting; no independently defensible phonemic/house conversion. PUA lower dots and tildes are represented as Unicode, drawn lowlines as U+0332; meaningful spacing and dashes remain.',
    'drawn_lowlines': sum(r[2].count('\u0332') for r in rows),
    'symbols': [{'character': c, 'codepoint': f'U+{ord(c):04X}', 'name': unicodedata.name(c, 'UNNAMED'), 'occurrences': n} for c, n in sorted(symbol_inventory.items())]}, ensure_ascii=False, indent=2) + '\n')
(HERE / 'proposal-status.json').write_text(json.dumps({'status': 'complete source-local candidate; independent review and metadata/installation gates pending',
    'first_reading_sha256': expected, 'csv_sha256': sha(HERE / 'proposed.csv'), 'audit_sha256': sha(HERE / 'proposal-audit.jsonl'),
    'physical_blocks': 57, 'physical_witnesses': 238, 'target_rows': 223, 'comparison_exclusions': 10,
    'documentary_name_exclusions': 5, 'audit_records': len(audit), 'graph_proposals': 12, 'applied_edges': 12,
    'canonical_changes': False, 'full_build_and_browser': 'deferred by user instruction'}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'rows': len(rows), 'audit_records': len(audit), 'graph_proposals': len(graph), 'csv_sha256': sha(HERE / 'proposed.csv')}))
