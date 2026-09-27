"""Expand reviewed lexical evidence into draft units; never install a CSV.

Run from any working directory. Every physical entry receives an audit record.
Ambiguous suffix scope is held for review instead of guessed. The output is an
intermediate: comparisons, graph relationships and grammatical tags need review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent


def nfc(text):
    return unicodedata.normalize('NFC', text)


def expand(records):
    units, audit = [], []
    for record in records:
        key = record['entry_key']
        forms = record.get('alternate_forms') or [record['transcribed_form']]
        if not forms or len(forms) != len(set(forms)):
            raise ValueError(f'Empty or duplicate alternate list: {key}')
        if any(any(c in form for c in '(),') for form in forms):
            raise ValueError(f'Unexpanded source notation: {key}')
        emitted, pending = [], []

        def add(form, unit_key, kind, evidence, **extra):
            if not form or '\ufffd' in form:
                raise ValueError(f'Invalid form: {unit_key}')
            unit = dict(entry_key=unit_key, physical_entry_key=key,
                        form=nfc(form), unit_kind=kind,
                        language_id='OllariGadaba',
                        printed_page=record['printed_page'],
                        pdf_page=record['pdf_page'], column=record['column'],
                        source_pos=record['source_pos'], gloss=record['gloss'],
                        form_evidence=evidence,
                        status='draft-not-installed',
                        uncertainties=record.get('uncertainties', []),
                        pronunciation_overrides=record.get('pronunciation_overrides', []),
                        **extra)
            units.append(unit)
            emitted.append(unit_key)

        for i, form in enumerate(forms):
            add(form, key if i == 0 else f'{key}:variant:{i + 1}',
                'headword' if i == 0 else 'printed-alternate',
                record['source_head'])
        for i, morph in enumerate(record['morphology'], 1):
            kind, raw = morph['kind'], morph['raw']
            if kind == 'suffix':
                if len(forms) != 1:
                    pending.append(dict(annotation_index=i, annotation=morph,
                                        reason='suffix-scope-across-alternate-stems',
                                        candidate_bases=forms))
                    continue
                if not raw.startswith('-') or forms[0].endswith('-'):
                    raise ValueError(f'Unresolved suffix notation: {key}')
                form = forms[0] + raw[1:]
            elif kind in ('full-form', 'stem'):
                form = raw
            elif kind == 'ending-replacement':
                if len(forms) != 1 or not forms[0].endswith(morph['replace']):
                    raise ValueError(f'Invalid ending replacement: {key}')
                form = forms[0][:-len(morph['replace'])] + morph['with']
                if form != morph['expanded_form'] or not morph.get('evidence'):
                    raise ValueError(f'Unverified ending replacement: {key}')
            else:
                pending.append(dict(annotation_index=i, annotation=morph,
                                    reason='unresolved-morphology-notation'))
                continue
            add(form, f'{key}:inflection:{i}', 'inflection', raw,
                morphology_label=morph['label'], expansion_kind=kind,
                supporting_evidence=morph.get('evidence'))
        audit.append(dict(physical_entry_key=key,
                          status='draft-expanded' if not pending else 'draft-partially-expanded',
                          emitted_keys=emitted, pending_morphology=pending,
                          lexical_review=record['lexical_review'],
                          comparison_review=record['comparison_review'],
                          source_record=record))
    keys = [u['entry_key'] for u in units]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate child key')
    return units, audit


def build(output):
    files = sorted(HERE.glob('reviewed-p*.json'))
    records = [r for f in files for r in json.loads(f.read_text())]
    if {r['printed_page'] for r in records} != set(range(48, 78)):
        raise ValueError('Incomplete vocabulary-page review')
    units, audit = expand(records)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('lexical-units.jsonl', units), ('expansion-audit.jsonl', audit)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
    summary = dict(status='draft-not-installed', physical_records=len(records),
                   lexical_units=len(units),
                   headwords=sum(u['unit_kind'] == 'headword' for u in units),
                   printed_alternates=sum(u['unit_kind'] == 'printed-alternate' for u in units),
                   inflections=sum(u['unit_kind'] == 'inflection' for u in units),
                   pending_morphology=sum(len(a['pending_morphology']) for a in audit),
                   input_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                   limitations=['No CSV installed.', 'POS and sense labels are source evidence, not final tags.',
                                'Printed alternates and inflections are not yet assigned graph relation types.',
                                'Comparative prose and fresh source-to-output audit remain pending.'])
    (output / 'expansion-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
