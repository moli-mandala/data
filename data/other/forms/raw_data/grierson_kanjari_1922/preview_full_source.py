"""Reproduce the complete reviewed Kanjari inventory; write proposals only."""
import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = 'grierson1922lsi11'
spec = importlib.util.spec_from_file_location('kanjari_table_grammar', HERE / 'table_grammar.py')
grammar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grammar)
LECTS = {
    'Sitapur': ('kanjari_sitapur', 'Sitapur'),
    'Belgaum': ('kanjari_belgaum', 'Belgaum'),
    'Aligarh': ('kanjari_aligarh', 'Aligarh'),
    'Etawah': ('kanjari_etawah', 'Etawah'),
    'Farrukhabad': ('kanjari_farrukhabad', 'Farrukhabad'),
    'Kuchbandhi/Bahraich': ('kanjari_kuchbandhi_bahraich', 'Kuchbandhi-Bahraich'),
    'Bahraich Kuchbandhi': ('kanjari_kuchbandhi_bahraich', 'Kuchbandhi-Bahraich'),
}

def read(name):
    with (HERE / name).open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert all(r['review_status'] == 'second_reading_original_complete' for r in rows)
    return rows

def nfc(s):
    return unicodedata.normalize('NFC', s.strip())

def dialect(lect):
    key, label = LECTS[lect]
    return f'dialect:Kanjari:{key}:{label}'

def row(key, form, gloss, citation, lects=(), tags=(), notes=''):
    return ['Kanjari', '', nfc(form), gloss, '', '', notes, citation, '', '', key,
            '', '', '', ' '.join(dict.fromkeys([*(dialect(l) for l in lects), *tags]))]

def generate():
    out, audit = [], []
    table = read('table-staged.tsv')
    assert [int(r['prompt']) for r in table] == list(range(1, 242))
    for c in table:
        item = int(c['prompt'])
        for lect in ('sitapur', 'belgaum'):
            key = f'{SOURCE}:{lect}:{item}'
            raw = nfc(c[lect])
            # These cells explicitly enumerate alternatives; all other commas
            # and spaces remain inside the printed expression.
            variants = raw.split(',') if lect == 'sitapur' and item in {33, 187} else raw.split(';')
            variants = [nfc(s) for s in variants if s.strip()]
            if lect == 'sitapur' and item == 187:
                variants[1] = 'Wō ' + variants[1]
            keys = []
            notes = 'The shared printed subject Wō is expanded for the second alternative.' if lect == 'sitapur' and item == 187 else ''
            for j, form in enumerate(variants):
                k = key if j == 0 else f'{key}:answer{j+1}'
                keys.append(k)
                tags = grammar.explicit_table_tags(item)
                local_notes = notes
                if item == 66 and lect == 'sitapur':
                    local_notes = 'The original final mark matches the compact first-i dot, unlike the adjacent Jhuraī macron. Literal Nimāni is retained, distinct from prose nīmānī.'
                out.append(row(k, form, c['gloss'], f'{SOURCE}[p. {c["page"]}, item {item}]',
                               [lect.capitalize()], tags, local_notes))
            audit.append(dict(c, section='table', source_cell_key=key, raw_cell=raw,
                              source_lect=lect, entry_keys=keys, reuse_entry_keys=[],
                              english_printed_page=int(c['page'])-2,
                              uncertainty='',
                              status='ingested' if keys else 'source_blank'))

    for c in read('prose-staged.tsv'):
        key = f'{SOURCE}:prose:p{c["page"]}:{c["unit"]}'
        keys = []
        # Kheri is explicitly described as Hindostani, not Kanjari (pp97/105).
        # The contradictory p101 argot example is preserved in full evidence.
        control = c['lects'] == 'Kheri'
        bound = c['page'] == '119' and c['unit'] == 'genitive'
        lects = [l for l in c['lects'].split(';') if l in LECTS]
        notes = c['note']
        if c['lects'] == 'Kirkpatrick quotation':
            notes += ' Quoted by Grierson from W. Kirkpatrick, A Vocabulary of the Pasi Boli or Argot of the Kunchbandiya Kanjars (1911), pp277ff; no specimen locality is assigned.'
        if not control and not bound:
            for j, form in enumerate(c['forms'].split(';')):
                k = key if j == 0 else f'{key}:answer{j+1}'
                keys.append(k)
                out.append(row(k, form, c['gloss'], f'{SOURCE}[p. {c["page"]}, prose {c["unit"]}]',
                               lects, c['tags'].split(), notes))
        audit.append(dict(c, section='prose', source_cell_key=key, entry_keys=keys, reuse_entry_keys=[],
                          status='non_target_Hindostani_control' if control else 'inventory_only_bound_genitive_suffix' if bound else 'ingested'))

    seen = {}
    for c in read('specimen-staged.tsv'):
        key = f'{SOURCE}:specimen:{c["unit"]}'
        form, gloss = nfc(c['form']), c['gloss']
        pair = (c['lect'], form, gloss)
        citation = f'{SOURCE}[p. {c["page"]}, specimen {c["specimen"]}, line {c["line"]}, aligned word {c["word"]}]'
        if pair in seen:
            old = out[seen[pair]]
            old[7] += ';' + citation
            keys, reuse = [], [old[10]]
        else:
            seen[pair] = len(out)
            keys, reuse = [key], []
            notes = c['note'] if c['note'] != 'Aligned Roman/English source token, no inferred lemma.' else ''
            out.append(row(key, form, gloss, citation, [c['lect']], notes=notes))
        audit.append(dict(c, section='specimen', source_cell_key=key, entry_keys=keys, reuse_entry_keys=reuse,
                          status='reused_identical_lect_form_gloss' if reuse else 'ingested'))
    assert len(audit) == 482 + 239 + 1508
    keys = {r[10] for r in out}
    assert len(keys) == len(out)
    assert all(k in keys for a in audit for k in a['entry_keys'] + a['reuse_entry_keys'])
    return out, audit

if __name__ == '__main__':
    rows, audit = generate()
    with (HERE / 'full-preview.csv').open('w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)
    with (HERE / 'full-preview-audit.jsonl').open('w') as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    print(f'{len(audit)} source units; {len(rows)} proposed forms; no canonical writes')
