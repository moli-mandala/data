"""Reconcile frozen paired first readings with the explicit review sidecar.

Writes source-local reviewed derivatives only. No canonical installation.
"""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_tsv(name):
    with (P / name).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(name, rows):
    (P / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    review = json.loads((P / 'whole-paired-second-reading-20260927.json').read_text())
    grammar = load_tsv('whole-paired-grammar-first-reading-20260927.tsv')
    assert len(grammar) == 303
    for row in grammar:
        row['original_reading'] = dict(row)
        row['review_corrections'] = []
    for fix in review['corrections']:
        found = [r for r in grammar if int(r['page']) == fix['page'] and r['section'] == fix['section'] and r['item'] == fix['item'] and r[fix['field']] == fix['old']]
        assert len(found) == 1, fix
        found[0][fix['field']] = fix['new']
        found[0]['review_corrections'].append(fix)
    for row in grammar:
        scope = ('rampur' if any(row[f] for f in ['rampur_sg', 'rampur_pl']) else '') + ('baghi' if any(row[f] for f in ['baghi_sg', 'baghi_pl']) else '')
        scope = scope or 'shared'
        row['source_unit_key'] = f"paired:grammar:{row['page']}:{row['section']}:{row['item']}:{scope}"
        row['review_status'] = 'owner original-page second reading complete; final output audit pending'
    for section, form, gloss, gender in [('house-heading', 'gauhṛ', 'house', 'masculine'), ('girl-heading', 'tshōṭi', 'girl, daughter', 'feminine')]:
        grammar.append(dict(page='131', section=section, item='1', gloss=gloss, rampur_sg='', rampur_pl='', baghi_sg='', baghi_pl='', shared_form=form, source_label=f'noun; {gender}; separate printed heading', note='Distinct printed heading spelling, retained separately from the following paradigm.', source_unit_key=f'paired:grammar:131:{section}:1:shared', original_reading=None, review_corrections=[], review_status='additional heading attestation confirmed in owner second reading', evidence=review['additional_attestations']))
    assert len({r['source_unit_key'] for r in grammar}) == len(grammar)
    sentences = load_tsv('whole-paired-sentences-first-reading-20260927.tsv')
    for row in sentences:
        row['original_reading'] = dict(row)
        row['review_corrections'] = []
    for fix in review['sentence_review']['corrections']:
        found = [r for r in sentences if int(r['page']) == fix['page'] and r['item'] == fix['item'] and r['lect'] == fix['lect']]
        assert len(found) == 1 and found[0][fix['field']] == fix['old']
        found[0][fix['field']] = fix['new']
        found[0]['review_corrections'].append(fix)
    for row in sentences:
        row['source_unit_key'] = f"paired:sentence:{row['page']}:{row['item']}:{row['lect'].lower()}"
        row['review_status'] = 'owner original-page second reading complete; final output audit pending'
    write('whole-paired-grammar-reviewed-20260927.jsonl', grammar)
    write('whole-paired-sentences-reviewed-20260927.jsonl', sentences)
    report = dict(status='reviewed source-local derivatives; expansion and final audit pending', grammar_groups=len(grammar), sentence_groups=len(sentences), commentary_groups=19, introduction_groups=27, frozen_grammar_groups=303, added_headings=2, corrected_grammar_fields=len(review['corrections']), corrected_sentence_forms=len(review['sentence_review']['corrections']), inputs={n:sha(P/n) for n in ['whole-paired-grammar-first-reading-20260927.tsv','whole-paired-sentences-first-reading-20260927.tsv','whole-paired-second-reading-20260927.json']})
    (P/'whole-paired-reviewed-summary-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
