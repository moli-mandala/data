"""Record completed independent visual review; never modify owner inputs."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent / 'comparisons'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(x) for x in p.read_text().splitlines()]
selection = json.loads((HERE / 'selection.json').read_text())
for name, digest in selection['pins'].items():
    assert sha(OWNER / name) == digest, name
rows = lines(OWNER / 'comparison-reviewed-candidate.jsonl')
accounting = lines(OWNER / 'comparison-review-accounting.jsonl')
bykey = {r['source_unit_key']: r for r in rows}
assert len(bykey) == len(rows) == 1616
assert len(accounting) == 1628
mapped = [k for r in accounting for k in r.get('candidate_keys', [])]
assert set(mapped) == set(bykey)
results = []
for i, r in enumerate(selection['selected_rows'], 1):
    assert r == bykey[r['source_unit_key']]
    failed = r['item'] == 203
    results.append(dict(sample=i, source_unit_key=r['source_unit_key'], actual_row=r,
        original_image=f"tmp/pdfs/korwa-kodaku-2022/wordlist-200dpi/page-{int(r['historical_audit_record']['PDF_Page']):03d}.png",
        literal_gloss_meaning_and_site='pass',
        status='material metadata failure' if failed else 'pass',
        finding="Parsed '(2nd sg, informal)' remains in lexical Gloss despite explicit pron/2sg/sg/informal tags. Preserve full source prompt in audit; export lexical gloss 'you'." if failed else None))
controls=[]
for r in selection['targeted_rows']:
    assert r == bykey[r['source_unit_key']]
    controls.append(dict(source_unit_key=r['source_unit_key'], actual_row=r,
        status='pass for targeted literal/field-boundary/alternative/layout purpose',
        note='Item 208 left-column placement supported; retain typed conflict with printed similarity-group ranks. Distinct item 210 right-column V occurrence remains attested.' if r['item']==208 else 'Original source supports owner correction or retained fine-mark reading.'))
affected=[dict(source_unit_key=r['source_unit_key'],item=r['item'],old_gloss=r['gloss'],tags=r['tags']) for r in rows if 202<=r['item']<=210]
report=dict(status='failed: systematic parsed-label leakage requires revised output and fresh sample',
    reviewer='rohru_finish', input_hashes=selection['pins'], selection_sha256=sha(HERE/'selection.json'),
    method='All 20 seeded rows inspected against existing original page images, plus 19 targeted candidate rows covering all owner correction classes. Fine glyph crops for PDF67/69/86 and full PDF90 independently inspected. Full candidate/accounting structural checks; not a new exhaustive literal rereading of 1616 rows.',
    sampled_rows=20, sampled_literal_errors=0, sampled_material_metadata_errors=1, sampled_pass=19,
    targeted_rows=len(controls), results=results, targeted_results=controls,
    structural_checks=dict(candidate_rows=len(rows),accounting_units=len(accounting),unique_keys=True,all_candidate_keys_accounted=True,
        accounting_dispositions=dict(collections.Counter(r['disposition'] for r in accounting))),
    systematic_correction=dict(scope='All explicitly parsed person/number/register/gender/inclusivity labels on prompts 202–210',
        affected_rows=len(affected),rows=affected,policy='Remove parsed grammatical label from lexical gloss; preserve exact source_prompt and original audit. Do not replace source-supported tags or erase ambiguous source layout.'),
    identity_gate='All owner language routes remain unaccepted/null; this review does not approve Tanmai or any inferred identity.',
    pending='Review root cleaned integration fields, then draw fresh disjoint 20 from corrected stable output; no canonical installation approval.')
(HERE/'independent-review-pass1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'sampled':20,'errors':1,'targeted':len(controls),'affected_rows':len(affected)},ensure_ascii=False))
