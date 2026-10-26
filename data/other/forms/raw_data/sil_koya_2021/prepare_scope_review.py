"""Record source-segment scope decisions after the first failed output audit.

This writes a review sidecar only. Frozen output is not regenerated here.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'
# Explicit keyed decisions from reading every target segment beside its source
# translation and neighboring segments. These are not punctuation heuristics.
CLIPPED = '''
B.1:p17:1 B.1:p17:2 B.1:p17:7 B.1:p17:8
B.1:p17:10 B.1:p17:11 B.1:p17:12 B.1:p17:13
B.1:p18:1 B.1:p18:2 B.1:p18:4 B.1:p18:5 B.1:p18:6 B.1:p18:7
B.1:p18:10 B.1:p18:11 B.1:p18:18 B.1:p18:19
B.2:p20:1 B.2:p20:2 B.2:p20:18 B.2:p20:19 B.2:p21:1
B.2:p21:9 B.2:p21:10
B.4:p24:3 B.4:p24:4 B.4:p24:10 B.4:p24:11 B.4:p25:4 B.4:p25:5
B.5:p26:8 B.5:p26:9 B.5:p26:11 B.5:p26:12 B.5:p26:16 B.5:p26:17
B.5:p27:5 B.5:p27:6
'''.split()
# The free translation ends in a temporal fragment, despite printed full stops.
AMBIGUOUS = {'B.6:p28:7'}


def main():
    source = OUT / 'narrative-reviewed-analysis.jsonl'
    units = [json.loads(s) for s in source.read_text().splitlines()]
    assert len(units) == 205
    known = {u['source_unit_key'].removeprefix('devagnanavaram2021:whole:') for u in units}
    assert set(CLIPPED) | AMBIGUOUS <= known
    decisions = []
    for unit in units:
        short = unit['source_unit_key'].removeprefix('devagnanavaram2021:whole:')
        fragment = short in CLIPPED or short in AMBIGUOUS
        tag = 'multiword-expression' if fragment else 'sentential'
        reason = ('Printed segment has a clipped boundary or continues a neighboring segment; may also contain complete clauses. No standalone-sentence claim.'
                  if short in CLIPPED else
                  'Free translation includes a temporal fragment; conservative multiword scope, without resolving target-language clause completeness.'
                  if short in AMBIGUOUS else
                  'Printed target segment and paired translation represent a complete question, sentence, or sequence of sentences.')
        decisions.append({'source_unit_key': unit['source_unit_key'],
                          'original': unit['original_linguistic_text'], 'source_translation': unit['gloss'],
                          'scope_tag': tag, 'retain_interr': unit['kind'] == 'test-question',
                          'reason': reason, 'reviewer': 'root'})
    report = {'status': 'whole narrative scope review; proposed corrections, not installed',
              'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'counts': dict(Counter(d['scope_tag'] for d in decisions)),
              'policy': 'Scope tags describe preserved printed units, not inferred internal morphology. Clipped units remain separate with exact source keys and translations. Supplemental lexical terms are outside this narrative-scope review.',
              'comparative_policy': 'Tag space-separated vere vere and vera vera attestations multiword-expression; do not infer a word boundary inside veGeveGe.',
              'decisions': decisions}
    (OUT / 'narrative-scope-review-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
