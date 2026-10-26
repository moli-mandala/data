"""Record independent original Kakkala findings on the frozen owner candidate."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
selection = json.loads((HERE / 'root-fresh20-selection-20260927461.json').read_text())
assert selection['csv_sha256'] == '1f2cc08fe786e25c3618aafdbf6164149c18da2854b5cc88bcbdf813c0b8a4bd'
assert selection['audit_sha256'] == 'b92f5954a6878ca6d546388d68c98798e641dd96513e2cce78c34b6bbe5cdd44'
repairs = {
    'p29:negative-markers:w1': 'Literal -aatt agrees. Source explicitly describes negative relative participle; relative tag is omitted.',
    'p30:imperative-plural-examples:w3': 'Literal eṇṇ-uŋko agrees and 2pl is structured. Grammatical (Pl.) label remains in lexical Gloss.',
    'p31:personal-examples:w11': 'Literal toḍa –kk-ir-i and wipes (it) agree. Source table explicitly scopes this example under after present tense markers; pres tag is omitted.',
}
sample = []
for item in selection['sample']:
    row = item['row']
    suffix = row[10].split(':', 1)[1]
    reason = repairs.get(suffix)
    sample.append({'page': item['page'], 'entry_key': row[10], 'form': row[2], 'gloss': row[3], 'tags': row[14],
                   'literal_form_and_gloss_scope': 'pass', 'verdict': 'repair' if reason else 'pass',
                   'original_observation': reason or 'Original printed literal form, semantic gloss scope and scoped tags agree; no material discrepancy in this sampled row.'})
assert len(sample) == 20 and sum(r['verdict'] == 'repair' for r in sample) == 3
report = {'reviewer': 'root', 'status': 'source-stage acceptance withheld pending source-wide metadata repairs and fresh final audit',
    'seed': selection['seed'], 'sampling': selection['sampling'],
    'csv_sha256': selection['csv_sha256'], 'audit_sha256': selection['audit_sha256'],
    'source_pdf_sha256': selection['source_pdf_sha256'], 'image_sha256': selection['image_sha256'],
    'original_pages_visually_reread': [28, 29, 30, 31],
    'sampled_rows': 20, 'literal_form_and_gloss_scope_passes': 20,
    'sample_passes': 17, 'sample_material_errors': 3, 'sample': sample,
    'systematic_classes': ['Explicit relative versus verbal participle class', 'Grammatical Sg./Pl. labels left in glosses',
                           'Explicit source table tense scope missing from personal examples'],
    'targeted_controls': [
        'Original p28 pronoun table/prose disagreement preserved; table gender tags do not silently rewrite prose.',
        'Both p28 female singular slash groups support proposed variant links within their source cells.',
        'Original p29 decomposition/surface arrows remain analyses, not invented variants or ancestry.',
        'Separate acute, ň, macrons, literal en dashes and drawn lowlines are distinct; sampled instances agree.',
        'Original p31 bare plural a is retained and printed -ø remains documentary zero exponent.',
        'Original p31 Tea, water line has no secure attached target; audit-only hold is justified.',
        'Two reported 1980 thesis identities are supported by original p31 and remain reported only.'],
    'required_followup': ['Archive exact reviewed revision', 'Source-wide repair of the three metadata classes',
                          'Focused regressions and exact changed-field continuity', 'Independent targeted reread',
                          'Fresh disjoint final graph-bearing output sample'],
    'limits': 'This balanced actual-output sample and four-page reading do not independently certify all 207 source slots. Whole scope and lowline census remain explicit owner evidence. No full CLDF pipeline, compiled checks, full suite or browser refresh was run.'}
(HERE / 'root-original-review-pass1-20260927461.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'literal_passes': 20, 'overall_passes': 17, 'material_errors': 3,
                  'report_sha256': hashlib.sha256((HERE / 'root-original-review-pass1-20260927461.json').read_bytes()).hexdigest()}))
