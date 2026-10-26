"""Validate and key owner transcriptions for review, without installation."""
import ast
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
import import_source as legacy

P = Path(__file__).resolve().parent
R = P / 'whole-source-recovery'
DATA = P.parents[4]


def main():
    with legacy.OUTPUT.open(newline='') as stream:
        baseline = list(csv.reader(stream))
    regenerated, _ = legacy.generate()
    assert regenerated == baseline, 'Existing glossary regeneration differs from canonical'
    assert len(baseline) == 329 and len({r[10] for r in baseline}) == 329
    baseline_record = dict(rows=329, csv_sha256=hashlib.sha256(legacy.OUTPUT.read_bytes()).hexdigest(),
                           entry_keys=[r[10] for r in baseline],
                           status='existing glossary only; preserve all keys during whole-source integration')
    baseline_path = R/'legacy-baseline-20260927.json'
    if baseline_path.exists():
        assert json.loads(baseline_path.read_text()) == baseline_record
    else:
        baseline_path.write_text(json.dumps(baseline_record,ensure_ascii=False,indent=2)+'\n')
    registered = set()
    for node in ast.parse((DATA / 'tags.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {'GENDER_TAGS', 'GRAMMATICAL_TAGS'}
            for t in node.targets
        ):
            registered.update(ast.literal_eval(node.value))
    units, keys, hashes = [], set(), {}
    for path in sorted(R.glob('p*-first-reading.tsv')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='') as stream:
            for raw in csv.DictReader(stream, delimiter='\t'):
                assert None not in raw and all(v is not None for v in raw.values()), raw
                page = int(raw['page'])
                assert 87 <= page <= 105
                key = f"bailey1920kagani:whole:p{page}:{raw['section']}:{raw['column']}:{raw['item']}"
                assert key not in keys, key
                keys.add(key)
                forms = [unicodedata.normalize('NFC', s) for s in raw['forms'].split('|')]
                assert all(f.strip() == f and f and '\ufffd' not in f for f in forms)
                tags = raw['tags'].split()
                assert set(tags) <= registered, (key, set(tags)-registered)
                units.append(dict(source_unit_key=key, printed_page=page, pdf_page=page+26,
                    section=raw['section'], column=raw['column'], item=raw['item'],
                    forms=forms, gloss=raw['gloss'], tags=tags,
                    source_analysis=raw['source_analysis'], raw_transcription=raw,
                    review='owner original-page first reading; independent audit pending',
                    image_evidence=f'tmp/kagani-whole/p{page}.png'))
    assert sum(u['printed_page'] == 88 for u in units) == 14
    assert sum(u['printed_page'] == 95 for u in units) == 52
    assert sum(u['printed_page'] == 96 for u in units) == 41
    assert sum(u['printed_page'] == 105 for u in units) == 28
    story = json.loads((R/'p105-story-first-reading.json').read_text())
    tokens = [pair for line in story['lines'] for pair in line['pairs']]
    assert len(tokens) == 34 and all(len(pair) == 2 and all(pair) for pair in tokens)
    covered = [i for span in story['whole_expression_spans']
               for i in range(span['first_token'],span['last_token']+1)]
    assert covered == list(range(1,35)), covered
    for name in ['p105-story-first-reading.json', 'p95-96-controls.json',
                 'p105-controls.json', 'tone-encoding-first-review.md']:
        hashes[name] = hashlib.sha256((R/name).read_bytes()).hexdigest()
    report = dict(status='partial working transcription; not installed or complete',
        source_units=len(units), candidate_forms=sum(len(u['forms']) for u in units),
        page_units=dict(Counter(u['printed_page'] for u in units)),
        second_story=dict(aligned_tokens=len(tokens),whole_expressions=3,
                          review=story['review']),
        input_sha256=hashes, validation='schema, unique source keys, registered tags and NFC only; not a lexical audit',
        existing_glossary=baseline_record['csv_sha256'],
        remaining='Remaining chapter transcription, full-source reconciliation, sound-profile validation, independent audit and installation; full compiled gates deferred.')
    (R/'owner-first-readings.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
    (R/'owner-first-readings-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
