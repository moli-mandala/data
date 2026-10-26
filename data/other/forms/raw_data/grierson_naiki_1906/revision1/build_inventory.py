"""Unify all recovered evidence before interpretation or canonical installation.

This deliberately emits no installed CSV. Grouped prose, controls and free
translation remain distinguishable from aligned lexical occurrences.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read_jsonl(name):
    return [json.loads(line) for line in (HERE/name).read_text().splitlines() if line]

def main():
    records=[]
    for name,kind in [('prose-reviewed.jsonl','prose-group'),
                      ('pronoun-table-reviewed.jsonl','pronoun-table-cell'),
                      ('extra-chapter-first-reading.jsonl','extra-chapter-witness'),
                      ('contextual-claims.jsonl','ambiguous-attribution-context')]:
        for row in read_jsonl(name):
            records.append(dict(key=row['source_unit_key'],kind=kind,
                printed_page=row['printed_page'],evidence_file=name,evidence=row,
                independent_review_pending=True))
    first={r['source_unit_key']:r for r in read_jsonl('specimens-first-reading.jsonl')}
    review=json.loads((HERE/'grayscale-corrections.json').read_text())
    uncertainties={(r['page'],r['line'],r['word']):r for r in review['uncertainties']}
    for row in read_jsonl('specimens-second-reading.jsonl'):
        p,l,w=row['page'],row['line'],row['word']
        key=f'grierson1906naiki:p{p}:specimen:line{l}:word{w}'
        original=first.pop(key)
        assert (row['first_form'],row['first_gloss']) == (original['raw_form'],original['raw_gloss'])
        records.append(dict(key=key,kind='aligned-token',printed_page=p,
            evidence_file='specimens-second-reading.jsonl',evidence=row,
            original_evidence=original,uncertainty=uncertainties.get((p,l,w)),
            independent_review_pending=True))
    assert not first
    for row in read_jsonl('reviewed-whole-utterances.jsonl'):
        records.append(dict(key=row['source_unit_key'],
            kind='whole-'+row['kind'], printed_page=row['printed_pages'][0],
            evidence_file='reviewed-whole-utterances.jsonl', evidence=row,
            independent_review_pending=True))
    translation=json.loads((HERE/'specimen-II-free-translation.json').read_text())
    for n,paragraph in enumerate(translation['paragraphs'],1):
        records.append(dict(key=f'grierson1906naiki:p575:free-translation:paragraph{n}',
            kind='context-only-free-translation',printed_page=575,
            evidence_file='specimen-II-free-translation.json',evidence=paragraph,
            independent_review_pending=True))
    assert len({r['key'] for r in records}) == len(records)
    counts=Counter(r['kind'] for r in records)
    assert counts == {'prose-group':84,'pronoun-table-cell':11,
        'extra-chapter-witness':1,'ambiguous-attribution-context':1,
        'aligned-token':537,'context-only-free-translation':7,
        'whole-utterance':51,'whole-translated-block':7}
    text=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records)
    (HERE/'whole-source-inventory.jsonl').write_text(text)
    result=dict(stage='recovered evidence; not installed lexical rows',
        records=len(records),counts=dict(counts),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        typed_transcription_uncertainties=len(uncertainties),
        controls='Prose dispositions retained; no control-language lexical emission',
        pending='Independent review; grammar and sound profile; final output and metadata; focused tests; installation')
    (HERE/'whole-source-inventory-status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
