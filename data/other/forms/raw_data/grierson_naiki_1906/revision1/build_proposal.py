"""Generate a review-only Naiki proposal; never install canonical source files.

Every inventory unit receives a disposition. This deliberately preserves aligned
tokens literally until their grammatical and punctuation decisions are reviewed.
"""
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line]


def main():
    inventory = read('whole-source-inventory.jsonl')
    inventory_by_key = {unit['key']:unit for unit in inventory}
    grammar = {r['source_unit_key']: r for filename in (
        'prose-grammatical-proposal.jsonl', 'pronoun-grammatical-proposal.jsonl',
        'specimen-grammatical-proposal.jsonl')
        for r in read(filename)}
    rows, audit = [], []
    for unit in inventory:
        key, kind, evidence = unit['key'], unit['kind'], unit['evidence']
        decision = grammar.get(key, {})
        record = dict(source_unit_key=key, kind=kind, printed_page=unit['printed_page'],
                      evidence=unit, grammatical_decision=decision,
                      emitted_keys=[], independent_review_pending=True)
        if kind in {'ambiguous-attribution-context', 'context-only-free-translation'}:
            record['disposition'] = 'retain-context-without-lexical-emission'
            record['reason'] = ('Ambiguous collective language attribution.' if kind ==
                                'ambiguous-attribution-context' else
                                'Free translation supplies context, not additional aligned forms.')
        elif decision.get('disposition') == 'exclude-control':
            record.update(disposition='exclude-control', reason='Explicit comparative control language.')
        else:
            tags = list(decision.get('tags', []))
            notes = []
            citation = f'grierson1906naiki[p. {unit["printed_page"]}]'
            uncertainties = [u for u in [unit.get('uncertainty'),decision.get('uncertainty')] if u]
            if kind == 'prose-group':
                forms = evidence['raw_form_group'].split(' / ')
                gloss = decision['proposed_gloss']
                notes = [decision.get('source_analysis', ''), decision.get('source_caveat', '')]
            elif kind == 'pronoun-table-cell':
                forms, gloss = [evidence['raw_form']], decision['proposed_gloss']
            elif kind == 'extra-chapter-witness':
                forms, gloss = [evidence['raw_form']], ''
                tags = ['suffix', 'gen']
                notes = ['Source explicitly attributes this genitive suffix to Naiki and Kolami.']
            elif kind == 'aligned-token':
                forms, gloss = [decision['proposed_form']], decision['proposed_gloss']
                record['metadata_review_pending'] = decision['metadata_review_pending']
                record['punctuation_review_pending'] = False
            elif kind in {'whole-utterance','whole-translated-block'}:
                forms,gloss=[evidence['form']],evidence['free_translation']
                tags=['sentential','multiword-expression']
                notes=['Complete printed utterance; no separately paired free translation.' if not gloss else
                       'Printed free translation applies to this complete source block; speaker labels remain provenance.']
                if evidence['speaker_label_evidence']:
                    notes.append('Source speaker label: '+', '.join(r['form'] for r in evidence['speaker_label_evidence']))
                citation=';'.join(f'grierson1906naiki[p. {page}]' for page in evidence['printed_pages'])
                if gloss and 575 not in evidence['printed_pages']:
                    citation+=';grierson1906naiki[p. 575, free translation]'
                for child in evidence['source_token_keys']:
                    for u in [inventory_by_key[child].get('uncertainty'),grammar[child].get('uncertainty')]:
                        if u:
                            uncertainties.append(dict(source_token_key=child,reason=u))
            else:
                raise AssertionError(kind)
            if uncertainties:
                record['typed_uncertainties'] = uncertainties
                if 'uncertain' not in tags:
                    tags.append('uncertain')
            if kind != 'extra-chapter-witness':
                tags.append('dialect:Naiki:naiki_chanda_lsi1906:Chanda')
            for index, form in enumerate(forms, 1):
                child_key = key if len(forms) == 1 else f'{key}:alternate:{index}'
                row = ['Naiki', '', form, gloss, '', '',
                       ' '.join(n for n in notes if n),
                       citation, '', '',
                       child_key, '', '', '', ' '.join(tags)]
                row = [unicodedata.normalize('NFC', value) for value in row]
                assert len(row) == 15 and row[2] and '\ufffd' not in ''.join(row)
                rows.append(row)
                record['emitted_keys'].append(child_key)
            record['disposition'] = 'propose-literal-attestation'
            record['reason'] = 'Distinct physical evidence retained; no cross-attestation deduplication or inferred graph links.'
        audit.append(record)
    assert len(inventory) == len(audit) == 699
    assert len({r['source_unit_key'] for r in audit}) == 699
    assert len({r[10] for r in rows}) == len(rows)
    assert sum(r['disposition'] == 'exclude-control' for r in audit) == 5
    assert sum(r['disposition'] == 'retain-context-without-lexical-emission' for r in audit) == 8
    assert sum(len(r['emitted_keys']) for r in audit) == len(rows)
    with (HERE / 'whole-review-proposal.csv').open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (HERE / 'whole-review-proposal-audit.jsonl').write_text(''.join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in audit))
    status = dict(stage='Unfrozen review proposal; not installed',
                  inventory_units=len(inventory), proposed_rows=len(rows),
                  dispositions=dict(Counter(r['disposition'] for r in audit)),
                  sha256=hashlib.sha256((HERE / 'whole-review-proposal.csv').read_bytes()).hexdigest(),
                  aligned_metadata_pending=sum(r.get('metadata_review_pending', False) for r in audit),
                  pending=['Independent final utterance punctuation review',
                           'Independent whole-source and fresh seeded original/output audit',
                           'Language mapping confirmation and dialect registration',
                           'Independent review of prepared sound profile, bibliography and settings; canonical registration',
                           'Canonical installation and deferred compiled validation'],
                  transcription='Literal source transcription retained; no distinct IPA asserted.',
                  graph='No inferred etymon, variant, borrowing or derivation links.',
                  canonical_changed=False)
    (HERE / 'whole-review-proposal-status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
