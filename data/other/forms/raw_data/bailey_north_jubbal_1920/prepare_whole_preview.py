"""Assemble all recovered units for review; deliberately has no install mode."""
import ast
import copy
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

import import_source as legacy

HERE = Path(__file__).resolve().parent
RECOVERY = HERE / 'whole-source-recovery'
INPUTS = ['reviewed-grammar-p172-182-20260927.json',
          'sentences-p183-first-reading-20260927.json',
          'sentences-p184-first-reading-20260927.json',
          'shared-scope-review-20260927.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows, audit = legacy.generate()
    with legacy.OUTPUT.open(newline='') as stream:
        canonical = list(csv.reader(stream))
    assert rows == canonical, 'Legacy regeneration differs from canonical; reconcile first'
    old_count = len(rows)
    assert old_count == 330
    registered = set()
    for node in ast.parse((legacy.DATA / 'tags.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'GENDER_TAGS', 'GRAMMATICAL_TAGS'} for t in node.targets):
            registered.update(ast.literal_eval(node.value))
    currency = json.loads((RECOVERY / 'currency-cross-reference-20260927.json').read_text())
    hashes = {str(path.relative_to(HERE)): sha(path) for path in [
        HERE/'full-transcription.tsv', HERE/'import_source.py', Path(__file__).resolve(),
        RECOVERY/'currency-cross-reference-20260927.json']}
    controls, counts = [], {}
    for name in INPUTS:
        path = RECOVERY / name
        hashes[str(path.relative_to(HERE))] = sha(path)
        data = json.loads(path.read_text())
        units = data['source_units']
        counts[name] = len(units)
        controls.extend(dict(input_file=name, raw=c) for c in data.get('controls', []))
        for raw in units:
            unit = copy.deepcopy(raw)
            forms = unit.get('forms', unit.get('expanded_form_proposals'))
            assert forms and all(isinstance(f, str) and f.strip() for f in forms), unit
            tags = list(unit.get('tags', []))
            if 'multiword' in tags:
                tags[tags.index('multiword')] = 'multiword-expression'
            # No registered stative tag: retain exact source category in analysis.
            if 'stative' in tags:
                tags.remove('stative')
                unit['source_analysis'] = unit.get('source_analysis', '') + ' Source category: stative participle.'
            if 'sentential' in tags and 'multiword-expression' not in tags:
                tags.append('multiword-expression')
            assert not (set(tags) - registered), (unit['source_unit_key'], tags)
            page = unit['printed_page']
            key = unit['source_unit_key']
            citations = f'{legacy.SOURCE}[p. {page}, {key.split(":", 2)[2]}]'
            analysis = unit.get('source_analysis', unit.get('note', ''))
            if key in currency['from_source_keys']:
                analysis += ' ' + currency['source_analysis']
                citations += f';{legacy.SOURCE}[p. 167, note to sentence 13]'
                if 'uncertain' not in tags:
                    tags.append('uncertain')
                unit['uncertainty_type'] = 'source-analysis'
            emitted = []
            for index, form in enumerate(forms, 1):
                child_key = key if index == 1 else f'{key}:answer{index}'
                emitted.append(child_key)
                rows.append(['Barari', '', unicodedata.normalize('NFC', form),
                             unit.get('gloss', ''), '', '', analysis.strip(), citations,
                             '', '', child_key, '', '', '', ' '.join(tags)])
            audit.append(dict(source_cell_key=key, status='proposed-not-installed',
                              input_file=name, source_review=unit, entry_keys=emitted,
                              normalized_tags=tags))
    assert len({r[10] for r in rows}) == len(rows)
    assert rows[:old_count] == canonical
    assert all(len(r) == 15 and r[2] and '\ufffd' not in ''.join(r) for r in rows)
    target = HERE / 'whole-unfrozen-preview.csv'
    with target.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    audit_path = HERE / 'whole-unfrozen-preview-audit.jsonl'
    audit_path.write_text(
        ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in audit))
    report = dict(status='unfrozen review preview; not installed and not approved',
                  input_sha256=hashes, canonical_sha256=sha(legacy.OUTPUT),
                  legacy_rows_unchanged=old_count, proposed_rows=len(rows),
                  added_rows=len(rows)-old_count, source_unit_counts=counts,
                  total_audit_units=len(audit), controls=controls,
                  preview_sha256=sha(target), audit_sha256=sha(audit_path),
                  pending=['Independent sentence/shared review', 'Legacy glossary reconciliation',
                           'Sound profile and provenance reconciliation', 'Frozen output audit',
                           'Focused importer tests and installation', 'Deferred full-build gates'])
    (HERE / 'whole-unfrozen-preview-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    assert sha(legacy.OUTPUT) == report['canonical_sha256']
    print(json.dumps({k: report[k] for k in ['status', 'legacy_rows_unchanged', 'proposed_rows', 'added_rows', 'total_audit_units']}))


if __name__ == '__main__':
    main()
