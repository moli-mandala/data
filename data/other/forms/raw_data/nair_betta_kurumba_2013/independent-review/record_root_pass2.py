"""Root's independent fresh revision2 review; one sociative tag class remains."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
selection = json.loads((HERE / 'fresh20-selection-pass2.json').read_text())
assert selection['csv_sha256'] == 'c1f0933fe8f0ccc69ce8996ffd743a0960baedef4a81ce9a92bd9dd5345ddbaa'
assert selection['audit_sha256'] == 'd17eddccef9d9cf45713e35815545a7a010f023c867513942ad9ecaf558c6fd3'
rows = []
for item in selection['sample']:
    row = item['proposed_row']
    fail = row[10] == 'nair2013bettakurumba:p12:diagnostic-case-markers:03'
    rows.append({'entry_key': row[10], 'form': row[2], 'gloss': row[3], 'tags': row[14],
                 'literal_and_gloss': 'pass', 'verdict': 'repair' if fail else 'pass',
                 'original_observation': 'Source explicitly sociative; function was removed from gloss but not retained as canonical comitative tag.' if fail else 'Printed literal form and cleaned semantic gloss agree; scoped grammar agrees.'})
assert len(rows) == 20 and sum(r['verdict'] == 'repair' for r in rows) == 1
report = {'reviewer': 'root', 'status': 'one remaining sociative/comitative tag class needs repair',
    'seed': selection['seed'], 'csv_sha256': selection['csv_sha256'], 'audit_sha256': selection['audit_sha256'],
    'source_pdf_sha256': selection['source_pdf_sha256'], 'sampled_rows': 20,
    'sample_passes': 19, 'sample_material_errors': 1, 'literal_and_gloss_passes': 20, 'sample': rows,
    'repairs_confirmed': ['Marker glosses blank while raw source labels remain audited',
                          'Acc./Dat./Sg./Pl. labels moved from glosses into scoped tags',
                          'Relative participles tagged separately from verbal participles',
                          'Literal –i, –beliya and diagnostic -ooda agree with original occurrence glyphs',
                          'Source p16 timbu has literal segmented t̲in-p-u retained in note',
                          'Source p17 ba-nd-um contradiction remains visible',
                          'p18 personal table alignment and blank lexical marker glosses agree'],
    'required_followup': ['Preserve reviewed revision2', 'Map explicit sociative to registered comitative tag on marker and clause',
                          'Exact tag-only continuity and focused regression', 'Complete targeted literal/vector controls',
                          'Final fresh disjoint20'],
    'limits': 'Source-stage independent sample only; no full compiled pipeline, full suite or browser refresh.'}
(HERE / 'root-original-review-pass2-20260927422.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passes': 19, 'material_errors': 1, 'report_sha256': hashlib.sha256((HERE / 'root-original-review-pass2-20260927422.json').read_bytes()).hexdigest()}))
