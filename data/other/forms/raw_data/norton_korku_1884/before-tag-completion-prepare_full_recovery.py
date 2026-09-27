"""Reconcile reviewed whole-source evidence without altering historical proposals."""
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
DIALECT = 'dialect:ko:norton-1884:Korku%20%28Norton%201884%29'


def read(name):
    return json.loads((P / name).read_text())


def lines(name):
    return [json.loads(s) for s in (P / name).read_text().splitlines()]


def origin(key):
    return re.sub(r':(?:alt|slot|resolution|recovered).*$', '', key)


def tags(row, values):
    row[14] = ' '.join(dict.fromkeys(row[14].split() + values.split()))


def note(row, value):
    if value and value not in row[6]:
        row[6] = (row[6] + ' ' + value).strip()


def citation(unit):
    section = 'English–Kor' if ':english-kor:' in unit['entry_key'] else 'Kor–English'
    return f"cust1884korku[p. {unit['printed_page']}, {section}, {unit['column']} column, item {unit['item']}]"


def build():
    # This reviewed intermediate is historical and intentionally not overwritten.
    rows = list(csv.reader((P / 'expression-proposal.csv').open()))
    audit = lines('expression-proposal-audit.jsonl')
    before = {r[10]: copy.deepcopy(r) for r in rows}
    byunit = {u['entry_key']: u for u in audit}
    correction_evidence = read('full-forward-reconciled.json')
    for fix in correction_evidence['corrections']:
        unit = byunit[fix['entry_key']]
        matched = [r for r in rows if origin(r[10]) == fix['entry_key'] and r[2] == fix['old_form'].casefold()]
        assert len(matched) == 1, fix
        row = matched[0]
        row[2] = fix['form']
        unit.setdefault('previous_selected_forms', copy.deepcopy(unit['selected_forms']))
        unit['selected_forms'] = [fix['form'] if s.casefold() == fix['old_form'].casefold() else s for s in unit['selected_forms']]
        unit.setdefault('literal_corrections', []).append(fix)

    added = lines('held-recovery-final.jsonl') + correction_evidence['omitted_forms']
    for addition in added:
        key = addition['entry_key']
        unit = byunit[key]
        existing = [r for r in rows if origin(r[10]) == key]
        fkey = key if not existing else key + ':recovered' + str(1 + sum(':recovered' in r[10] for r in existing))
        assert not any(r[10] == fkey for r in rows)
        t = addition.get('tags', '')
        if isinstance(t, list):
            t = ' '.join(t)
        row = ['ko', '', addition['form'], addition['gloss'], '', '', addition.get('note', ''), citation(unit), '', '', fkey, '', '', '', DIALECT]
        tags(row, t)
        rows.append(row)
        unit.setdefault('previous_status', unit['status'])
        unit.setdefault('historical_note', unit.get('note', ''))
        unit['selected_forms'].append(addition['form'])
        unit['status'] = 'selected'
        unit.setdefault('recovered_evidence', []).append(addition)
        unit['note'] = 'Whole source cell represented after original-image review; historical decisions retained separately.'

    # An old reverse citation cannot stay attached to a forward form that was corrected
    # to a different literal spelling. Retain all old keys and the reverse attestation.
    bykey = {r[10]: r for r in rows}
    detached = []
    for unit in audit:
        if ':kor-english:' not in unit['entry_key']:
            continue
        for oldkey in list(unit.get('reused_keys', [])):
            old = before[oldkey]
            row = bykey[oldkey]
            if old[2] == row[2]:
                continue
            cit = citation(unit)
            assert cit in row[7].split(';'), (oldkey, cit)
            row[7] = ';'.join(c for c in row[7].split(';') if c != cit)
            key = unit['entry_key']
            fkey = key if key not in bykey else key + ':recovered' + str(1 + sum(origin(k) == key for k in bykey))
            restored = ['ko', '', old[2], unit['gloss'], '', '', '', cit, '', '', fkey, '', '', '', DIALECT]
            rows.append(restored)
            bykey[fkey] = restored
            unit['reused_keys'].remove(oldkey)
            unit['selected_forms'].append(old[2])
            unit['status'] = 'selected_and_reused' if unit['reused_keys'] else 'selected'
            evidence = dict(old_reused_key=oldkey, entry_key=fkey, form=old[2], reason='Corrected forward spelling differs from this literal reverse witness; retain separate attestation.')
            unit.setdefault('reuse_reconciliation', []).append(evidence)
            detached.append(evidence)

    metadata = read('forward-p165-168-metadata-reviewed.json') + correction_evidence.get('metadata', [])
    for item in metadata:
        unit = byunit[item['entry_key']]
        unit.setdefault('source_metadata', []).append(item)
        for row in rows:
            if origin(row[10]) != item['entry_key'] or (item.get('form') and row[2] != item['form']):
                continue
            tags(row, item.get('tags', ''))
            note(row, item.get('note', ''))
    for item in read('forward-f-qualifiers-reviewed.json'):
        unit = byunit[item['entry_key']]
        unit['source_qualifier_evidence'] = item
        matched = []
        for row in rows:
            if origin(row[10]) == item['entry_key'] and row[2] in item['forms']:
                tags(row, 'uncertain')
                note(row, item['note'])
                matched.append(row[2])
        assert sorted(matched) == sorted(item['forms']), (item, matched)

    # The old numeric cross-reference never carried a real key or citation.
    one = byunit['cust1884korku:numbers:p177:item01']
    target = next(r for r in rows if r[10] == 'cust1884korku:english-kor:p169:left:item36')
    assert target[2] == 'mī,ā' and target[3] == 'one'
    num_citation = 'cust1884korku[p. 177, Numbers, item 1]'
    if num_citation not in target[7].split(';'):
        target[7] += ';' + num_citation
    tags(target, 'num')
    one['reused_keys'] = [target[10]]
    one['reuse_reason'] = 'Identical literal form and one gloss; numeral attestation and citation retained.'
    for key in correction_evidence['reviewed_keys']:
        unit = byunit[key]
        unit.setdefault('historical_note', unit.get('note', ''))
        unit['note'] = 'Full forward source cell reviewed against original; literal spelling and explicit alternatives reconciled.'
        unit['full_cell_reviewed'] = True

    # Preserve every raw source unit and expose its exact final child-key mapping.
    for unit in audit:
        keys = [r[10] for r in rows if origin(r[10]) == unit['entry_key']]
        unit['entry_keys'] = keys + list(unit.get('reused_keys', []))
    assert len(audit) == 958 and len(byunit) == 958
    assert len({r[10] for r in rows}) == len(rows)
    assert set(before) <= {r[10] for r in rows}
    assert not any(u['status'] in {'held', 'excluded_sentence'} for u in audit)
    assert all(len(r) == 15 and r[2] for r in rows)
    return rows, audit, detached


def write():
    rows, audit, detached = build()
    with (P / 'full-recovery-proposal.csv').open('w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)
    (P / 'full-recovery-proposal-audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False, sort_keys=True) + '\n' for a in audit))
    report = dict(status='proposal; independent final review pending', rows=len(rows), audit_units=len(audit), detached_reverse_reuses=detached,
                  hashes={n: hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('full-recovery-proposal.csv', 'full-recovery-proposal-audit.jsonl')})
    (P / 'full-recovery-proposal-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    write()
