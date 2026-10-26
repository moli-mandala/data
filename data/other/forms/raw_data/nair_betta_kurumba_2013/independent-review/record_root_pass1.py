"""Record independent original-page review findings on the frozen first candidate."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
package = HERE.parent
selection = json.loads((HERE / 'fresh20-selection.json').read_text())
assert selection['csv_sha256'] == '111546a78056a47d02c9499be293dc1f84ad17ef0bc5d10c1b21d853c45c4531'
assert selection['audit_sha256'] == '21d90343c567024f37ac1da7fe47d26f008434bb7fcb1e7901fc28806e83a337'
failures = {
    'p12:diagnostic-case-markers:01': 'Printed -ma is correct, but accusative is a grammatical label in Gloss; suffix acc already structures it.',
    'p12:diagnostic-case-markers:07': 'Printed –puli has an en dash, candidate -puli has ASCII hyphen; locative is also a grammatical label in Gloss.',
    'p13:accusative-examples:01': 'Printed piliya-ma agrees; (Acc.) remains in Gloss and noun from the source label is omitted from Tags.',
    'p15:past-examples:07': 'Surface ettica agrees. Printed segmented ett-c-a has a drawn t lowline omitted in Notes/audit first reading.',
    'p15:nonpast-examples:05': 'Printed surface kengavi(yi) agrees; explicit you(sg.) lacks 2sg tagging and sg. remains in Gloss.',
    'p16:relative-participle-markers:01': 'Literal -a agrees, but relative class is absent from Tags and grammatical label remains in Gloss.',
    'p16:relative-participle-examples:01': 'Literal ba-nd-a and clause gloss agree; relative class is omitted despite explicitly scoped relative-participle section.',
    'p16:general-negative-examples:02': 'Literal odici-l-u agrees; source you(Pl.) requires 2pl and past-negative scope requires pret; Pl. remains in Gloss.',
    'p18:person-suffix-table:08:variant:1': 'Printed iyo and 2pl/nonpast cell alignment agree; constructed grammatical descriptor is put in Gloss.',
    'p18:person-suffix-table:07': 'Printed iyi and 2sg/nonpast cell alignment agree; constructed grammatical descriptor is put in Gloss.',
    'p18:person-suffix-table:06:variant:1': 'Printed u and 2pl/past cell alignment agree; constructed grammatical descriptor is put in Gloss.',
}
sample = []
for item in selection['sample']:
    key = item['key']
    suffix = key.split(':', 1)[1]
    row = item['proposed_row']
    reason = failures.get(suffix)
    sample.append({'entry_key': key, 'page': item['page'], 'form': row[2], 'gloss': row[3], 'tags': row[14],
                   'verdict': 'repair' if reason else 'pass',
                   'original_observation': reason or 'Original literal form, source semantic gloss and scoped tags agree; no material discrepancy found in this row.'})
assert len(sample) == 20 and sum(r['verdict'] == 'repair' for r in sample) == 11
report = {'reviewer': 'root', 'status': 'repair required; not accepted for source-stage installation',
    'seed': selection['seed'], 'csv_sha256': selection['csv_sha256'], 'audit_sha256': selection['audit_sha256'],
    'source_pdf_sha256': selection['source_pdf_sha256'],
    'original_pages_visually_reread': list(range(12, 19)),
    'original_image_sha256': {str(p): hashlib.sha256((package / f'original-review/page-{p}.png').read_bytes()).hexdigest() for p in range(12, 19)},
    'sampled_rows': 20, 'sample_passes': 9, 'sample_material_errors': 11, 'sample': sample,
    'systematic_classes': ['Grammatical labels in Gloss instead of structured-only Tags',
                           'Literal en dash normalized to ASCII hyphen',
                           'Drawn lowlines absent from candidate Form, segmented Notes and first reading',
                           'Explicit relative class and person/number/tense omitted from scoped examples'],
    'targeted_controls': [
        {'finding': 'p12 plain diagnostic -ooda versus p13 ooḍa, and p16 prose -lpad- versus -lpaḍ- examples, correctly retain source spelling differences.'},
        {'finding': 'p16 blank purposive speaking target correctly stays audit-only; no invented form.'},
        {'finding': 'p17 ba-nd-um retained despite prose -un rule; appropriate unresolved source contradiction.'},
        {'finding': 'p18 all twelve person/tense cells and three slash groups align with source; three variant edges have support within their cells.'},
        {'finding': 'p17 nooḍ-ku/nooḍuku initial-n lowlines and tardun lowlines; p18 durative/reflexive lowlines require whole geometric census, not PUA-only repair.'}],
    'required_followup': ['Archive exact reviewed revision', 'Whole seven-page dash/space and drawn-lowline census',
                          'Clean grammar from Gloss and complete explicitly scoped tags', 'Regression coverage for each systematic class',
                          'Independent reread of changed literal controls', 'Fresh disjoint20 of final graph-bearing candidate'],
    'limits': 'No full CLDF pipeline, compiled graph/identity checks, full suite or browser refresh was run. This report certifies findings, not completeness or acceptance.'}
(HERE / 'root-original-review-pass1-20260927421.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passes': 9, 'material_errors': 11, 'report_sha256': hashlib.sha256((HERE / 'root-original-review-pass1-20260927421.json').read_bytes()).hexdigest()}))
