"""Root first reading of Table B.2, checked against rendered printed pp.41–42.

This is source evidence for independent reconciliation, not installed forms.
Zero prefixes remain audit-only. Slash alternatives and co-occurrence groups
are retained without constructing unattested whole verbs.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA = 'aedf47151ebea80852cb7ed66d085fc18310aa3f4d35e189c698b624f3cbef5a'
SITES = [
    ('Bhokraha', 'Sunsari', 41, 1,
     ['-a', '-as/-ɑs/-ʌs', '-ia', '-iʌs', '-d͡ʒas/-t͡ʃʌs/-t͡ʃʰəs/-t͡ʃʰʌs'],
     [('ama-', '-a'), ('ma-', '-a ; -ʌs ; -ɑʔa/-ʌʔa/-ʔa ; -d͡ʒas'),
      ('mã-', '-a/-ã'), ('maʔ-', '-a ; -ʌs ; -aʔa/-ʌʔa')]),
    ('Lochani', 'Morang', 41, 1,
     ['-a', '-as/-ʌs', '-ia', '-ias/-iɑs/-iʌs', '-d͡ʒas/-t͡ʃas/-t͡ʃʰas/-t͡ʃʰɑs/-t͡ʃʰʌs'],
     [('ama-', '-a/-ã; -aʔa/-ʌʔa/-ʔa; -t͡ʃas'), ('amaʔ-', '-a ; -aʔa/-ʌʔa'), ('ma-', '-d͡ʒas; -ʌs')]),
    ('Siddhapur', 'Parsa', 41, 1,
     ['-a', '-as/-ʌs/-ɑs', '-ia', '-iɑs/-iʌs', '-asɛ/-ʌsə', '-d͡ʒas/-t͡ʃas/-t͡ʃɑs/-t͡ʃʰas/-t͡ʃʰɑs/-t͡ʃʰʌs'],
     [('ama-', '-a'), ('m-/ma-', '-a ; -ʔa/-ʔæ ; -ʌs ; -ʌʔa'),
      ('mã-', '-a'), ('maʔ-', '-a/-ʌ ; -ʔa ; -ɑʔa/-ʌʔa')]),
    ('Tokla', 'Jhapa', 41, 2,
     ['-a', '-as/-ʌs/-ɑs', '-d͡ʒas/-d͡ʒɑs', '-ia', '-ias/-iɑs/-iʌs', '-t͡ʃas/-t͡ʃʰas/-t͡ʃʰɑs/-t͡ʃʰʌs'],
     [('am-', '-a/-ʌ'), ('ama-/amʌ', '-a/-ʌ ; -ã ; -aʔa/-ʌʔa ; -ʔa'), ('amaʔ-', '-a ; -aʔa/-ʌʔa')]),
    ('Pathibari', 'West Bengal', 42, 1,
     ['-ɑ', '-d͡ʒəs', '-ənɑ/-ʌnɑ', '-əs/-ʌs', '-iəs/-iʌs',
      '-kat͡ʃes/-kat͡ʃɛs/-kət͡ʃɑs/-kʌt͡ʃʌs/-kit͡ʃa',
      '-tskɑt͡ʃʰəs/-t͡ʃʰkat͡ʃʌs/-t͡ʃkət͡ʃʰʌs', '-t͡ʃkai̯', '-t͡ʃʰɑs/-t͡ʃʰəs/-t͡ʃʌs'],
     [('ma-', '-ɑ ; -ɪke ; -ke ; -ʔoke ; -əke ; -ake'),
      ('maʔ-', '-eke/-ɛkɛ ; -ɪke ; -ke'),
      ('maʔa-\n/maʔə-', '-ke ; -eke/-ɛke ; -ʔokke\n-ə ; -ike/-ɪke/-ʔike ; -ʌke')]),
    ('Ranchi', 'Jharkhand', 42, 1,
     ['-əs/-ɪs/-us/-ʌs', '-iʌs/-ĩʌs', '-d͡ʒʌs', '-t͡ʃʰəs/-t͡ʃʰʌs/-t͡ʃʌs'],
     [('amba-\n/ambɑ-', '-a/-ɑ ; -ʌ ; -əʔa/-ʌʔa ; -ɪʔa\n-ʔa ; -kɑ')]),
]

def records():
    for site, region, page, zeros, suffixes, negatives in SITES:
        base = dict(site=site, region=region, printed_page=page,
                    pdf_page_index=page+7, source_sha256=SHA,
                    table='B.2', reviewer='root', status='first-reading-awaiting-reconciliation',
                    scope_note='Reference for this elicited dataset; author disclaims detailed morphological analysis.')
        prefix = f'kurux-nepal-2022:p{page}:B2:{site.lower()}'
        for i in range(zeros):
            yield dict(base, key=f'{prefix}:past:zero:{i+1}', kind='zero-prefix',
                       raw='∅-', disposition='audit-only', category='3SG simple past',
                       note='Physical zero-marker witness; do not emit empty lexical form or infer suffix-specific pairing.')
        for i, raw in enumerate(suffixes, 1):
            yield dict(base, key=f'{prefix}:past:suffix:{i}', kind='suffix-group',
                       raw=raw, category='3SG simple past', alternatives=raw.split('/'))
        for i, (pre, suf) in enumerate(negatives, 1):
            uncertain = (site == 'Pathibari' and i == 3) or site == 'Ranchi'
            note = ('Slash prefix group and adjacent suffix lines retained together; alternative-specific pairing is not established.'
                    if uncertain else 'Prefix and suffix lists matched by logical group order after line wrapping.')
            if site == 'Bhokraha' and i >= 2:
                note += ' The ma- suffix -d͡ʒas wraps onto the next baseline; physical baseline alone cannot establish pairing.'
            if site == 'Tokla' and i == 2:
                note += ' The second prefix alternative amʌ has no terminal hyphen in print; do not silently add one.'
            yield dict(base, key=f'{prefix}:negative:group:{i}', kind='prefix-suffix-cooccurrence-group',
                       raw_prefix=pre, raw_suffix=suf, category='2SG negative imperative',
                       uncertainty=['cooccurrence-scope'] if uncertain else [], note=note)
    for i, raw in enumerate(['-hem', '-kuna', '-kunri', '-kun', '-n̤u'], 1):
        yield dict(key=f'kurux-nepal-2022:p41:inline-suffix:{i}', kind='inline-suffix',
                   printed_page=41, pdf_page_index=48, source_sha256=SHA, raw=raw,
                   prompt_ids=[221, 222] if i == 5 else [190], gloss='at' if i == 5 else '',
                   site=None, reviewer='root', status='first-reading-awaiting-reconciliation',
                   note='Omitted from lexical similarity calculations, not an exclusion from source recovery. No site assignment printed here.')
    yield dict(key='kurux-nepal-2022:p42:B2:footnote-a', kind='qualified-source-claim',
               printed_page=42, pdf_page_index=49, source_sha256=SHA,
               raw='One wordlist consultant noted that suffixes ending with ‘-as’ are male and ‘-a’ are female.',
               claims=[{'ending':'-as', 'label':'male'}, {'ending':'-a', 'label':'female'}],
               attribution='one unnamed wordlist consultant', disposition='qualified-evidence-only',
               note='Do not propagate gender tags to all suffixes, sites, or verbs.',
               reviewer='root', status='first-reading-awaiting-reconciliation')

if __name__ == '__main__':
    rows = list(records())
    assert len({r['key'] for r in rows}) == len(rows)
    out = HERE / 'morphology-first-reading.jsonl'
    out.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))
    print(f'{len(rows)} first-reading evidence units; no canonical outputs modified')
