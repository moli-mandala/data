"""Record the independent full visual review; never modify author inputs."""
import collections
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = next(p for p in P.parents if (p / 'data' / 'SOURCE_INGESTION_CHECKLIST.md').exists())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(name):
    return [json.loads(x) for x in (P / name).read_text().splitlines()]

freeze = json.loads((P / 'first-reading-freeze.json').read_text())
checks = {name: sha(P / name) == digest for name, digest in freeze['files'].items()}
assert all(checks.values()), checks
atoms = [x for x in lines('interlinear-first-reading.jsonl') if x['specimen'] == 41]
spans = [x for x in lines('expression-spans-first-reading.jsonl') if x['specimen'] == 41]
assert len(atoms) == 328 and len(spans) == 31
by_key = {x['source_unit_key']: x for x in atoms}
findings = []
def correction(line, word, field, before, after, reason, clip):
    key = f'grierson1906kui:p466:specimen41:line{line}:word{word}'
    assert by_key[key][field] == before, by_key[key]
    findings.append(dict(source_unit_key=key, field=field, before=before, proposed_after=after,
        severity='material_transcription', evidence=reason, original_pixel_clip=clip,
        affected_expressions=[s['source_unit_key'] for s in spans if key in s['atom_keys']]))

correction(3, 7, 'raw_source_form', 'rāāni-kē', 'raāni-kē',
    'Only the second a carries a macron. The first a is plain, independently visible in the full-resolution original crop.', [3030, 930, 4360, 1210])
correction(3, 8, 'raw_gloss', 'staying-become.', 'staying-became.',
    'Printed English is staying-became., with a, not o. Do not regularize the source tense.', [3030, 930, 4360, 1210])
correction(18, 4, 'raw_source_form', 'Dāōkē', 'Dāōke',
    'Final e has no upper quantity stroke. Macron over a and o remains explicit.', [2400, 5040, 2910, 5370])

flat = [k for s in spans for k in s['atom_keys']]
assert flat == list(by_key), 'Expression sequence must preserve every source atom once.'
for s in spans:
    assert s['raw_source_form'] == ' '.join(by_key[k]['raw_source_form'] for k in s['atom_keys'])
    assert s['raw_interlinear_gloss'] == ' '.join(by_key[k]['raw_gloss'] for k in s['atom_keys'])
images = {}
for x in atoms:
    p = ROOT / x['source_image']
    if str(p) not in images:
        images[str(p)] = sha(p)
    assert images[str(p)] == x['source_image_sha256']

report = dict(
    reviewer='/root/asur_finish', date='2026-09-27',
    status='changes_required', scope='Complete specimen 41, printed pages 465–467, District Khondmals',
    method='Independent visual comparison of every source form and aligned English gloss on all three original alternate scan pages, followed by full-resolution narrow crops of suspected discrepancies. No linguistic normalization or paradigm inference.',
    frozen_manifest_sha256=sha(P / 'first-reading-freeze.json'), frozen_file_checks=checks,
    original_images_sha256=images,
    counts=dict(atoms=328, expressions=31, physical_lines=39,
                pages=dict(collections.Counter(str(x['printed_page']) for x in atoms)),
                material_field_corrections=len(findings)),
    findings=findings,
    coverage=[dict(source_unit_key=x['source_unit_key'], raw_source_form=x['raw_source_form'], raw_gloss=x['raw_gloss'],
                   result='correction_proposed' if any(f['source_unit_key']==x['source_unit_key'] for f in findings) else 'agrees_with_original_with_existing_uncertainty_retained') for x in atoms],
    expression_review=[dict(source_unit_key=s['source_unit_key'], first_atom=s['atom_keys'][0], last_atom=s['atom_keys'][-1],
                           boundary_result='agrees', corrections_propagate=any(f['source_unit_key'] in s['atom_keys'] for f in findings)) for s in spans],
    source_distinctions_preserved=[
        'Dotted i in ri and mrikā on p465 is not a macron; no quantity correction proposed.',
        'Plain dana is retained at each inspected occurrence; thick body strokes do not constitute quantity marks.',
        'The unusual printed satēnii is retained; no conventional spelling replacement.',
        'The source distinction ilbātiṅgi versus inbātiṅgi is retained.',
        'The p465–466 and p466–467 expression continuations are source-contiguous and correctly joined.',
        'Nested single/double quotation marks and sentence punctuation remain in atom and expression strings.',
        'Nasal diacritic typography retains the existing typed uncertainties; this review does not assert phonetic values or resolve a chapter notation key by analogy.'
    ],
    existing_typed_uncertainties=[dict(source_unit_key=x['source_unit_key'], uncertainty=x['typography_uncertainty']) for x in atoms if x.get('typography_uncertainty')],
    non_changes='No frozen input, canonical data, profile, metadata registry, or author transcription was modified. Corrections require author reconciliation; this is not an installation approval.'
)
out = P / 'independent-specimen41-full-review-20260927.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report':str(out), 'sha256':sha(out), 'atoms':len(atoms), 'expressions':len(spans), 'corrections':len(findings)}))
