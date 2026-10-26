"""Record root's final source-stage acceptance after original-page reread."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection = json.loads((HERE / 'fresh20-selection-pass3.json').read_text())
assert selection['csv_sha256'] == sha(PACKAGE / 'proposed.csv') == 'a5df733b3122dfa932163c0b66a2a2c5b60bccd0b12844a42203ce9f88ea28ec'
assert selection['audit_sha256'] == sha(PACKAGE / 'proposal-audit.jsonl') == 'a906f95e89dc0587aac00fd01e04563578282b63f14fca7a3d4b7216a308b580'
current = list(csv.reader((PACKAGE / 'proposed.csv').open()))
by_key = {r[10]: r for r in current}
assert len(current) == len(by_key) == 206
sample = []
for item in selection['sample']:
    row = item['proposed_row']
    assert row == by_key[row[10]]
    sample.append({'entry_key': row[10], 'form': row[2], 'gloss': row[3], 'tags': row[14], 'verdict': 'pass',
                   'original_observation': 'Original literal glyphs and source semantic gloss agree; printed grammar is structured or deliberately retained as qualified residual commentary where no exact house tag exists.'})
assert len(sample) == 20
assert 'comitative' in by_key['nair2013bettakurumba:p12:diagnostic-case-markers:03'][14].split()
assert 'comitative' in by_key['nair2013bettakurumba:p13:sociative-sentence:01'][14].split()
assert 'instr' not in by_key['nair2013bettakurumba:p13:sociative-sentence:01'][14].split()
variants = [r for r in current if r[11]]
assert len(variants) == 3 and all(r[11] in by_key and r[11] != r[10] for r in variants)
report = {'reviewer': 'root', 'status': 'pass: source-stage installation approved for exact reviewed candidate',
    'installation_approved': True, 'result': 'pass', 'material_errors': 0,
    'seed': selection['seed'], 'csv_sha256': selection['csv_sha256'], 'audit_sha256': selection['audit_sha256'],
    'source_pdf_sha256': selection['source_pdf_sha256'],
    'sampled_rows': 20, 'sample_passes': 20, 'sample_material_errors': 0, 'sample': sample,
    'original_pages_reviewed': list(range(12, 19)),
    'targeted_repair_review': {
        'dash_and_notation_occurrences': 28,
        'source_lowline_rectangles': 20,
        'underlined_letter_occurrences': 23,
        'comitative_tag_rows': 2,
        'evidence': ['dash-notation-corrections.json', 'lowline-vector-census.json', 'revision2-owner-reconciliation.json', 'comitative-tag-continuity.json'],
        'finding': 'Changed literal dash occurrences and lowline classes agree with original pp12–18; lowlines in segmented analysis as well as surfaces are retained. Plain d/underdotted d and source -un/-um differences remain distinct.'},
    'graph_review': {'source_explicit_variant_edges': 3, 'closed_actual_endpoints': 3,
                     'source_support': 'Original p18 three slash cells, correctly scoped by person/number/tense; no invented arrows-as-variants.'},
    'source_uncertainty': ['p16 blank speaking target remains audit-only', 'p17 ba-nd-um versus prose -un unresolved',
                          'p14 source causative-case meaning because of remains qualified source commentary, not silently reclassified as a verb derivation',
                          'p12 and p13 instrumental marker spelling disagreement preserved'],
    'limits': 'Independent source-stage review only. Full CLDF pipeline, compiled graph/reference/identity survival checks, full suite and browser refresh remain deferred. Guarded installation and canonical focused verification must still run.'}
(HERE / 'root-original-review-final-20260927423.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passes': 20, 'material_errors': 0, 'variants': 3,
                  'report_sha256': sha(HERE / 'root-original-review-final-20260927423.json')}))
