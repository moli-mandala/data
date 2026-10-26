"""Root's independent fresh actual-output sample against original pp45–49."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HERE / 'proposed.csv') == '7e95da15b5e4770da3e2fee7943413dd81b0637f4ea937e8e924b663184ed8b7'
assert sha(HERE / 'proposal-audit.jsonl') == '7c69e497b95665320dc66f75fb2a6f2ce7991051d13472705572544fbd47bb5d'
rows = list(csv.reader((HERE / 'proposed.csv').open()))
selected = []
for page in range(45, 50):
    pool = [r for r in rows if f':p{page}:' in r[10]]
    selected += sorted(pool, key=lambda r: hashlib.sha256(('20260927451:' + r[10]).encode()).hexdigest())[:4]
observations = [
    'Original Mannan pettu, delivered, right-hand column.',
    'Original Malayalam muṛṛam, courtyard; lower dots on both r letters retained.',
    'Original Malayalam vaṛṣam, year; column, lower dots and page locator agree.',
    'Original Mannan eecu, seven; numeral tag agrees.',
    'Original Mannan uraku, wing; dropped initial c preserved.',
    'Original Mannan cakaayam, help; continued p45 comparison heading supports column assignment.',
    'Original Malayalam kapham, phlegm, comparison column.',
    'Original peeṭṭa-laam, brothers-in-law, literal form and plural tag agree; explicit kinship class is missing.',
    'Original Malayalam –nnu in p47 prose has no drawn lowline; correct distinction from p46 nn controls.',
    'Original proximal female ivaḷu; no invented number tag from absent subsection headings.',
    'Original first-person inclusive plural naŋka, table alignment and tags agree.',
    'Original accusative –e expressly occurs in Malayalam also; shared physical marker mapped as scoped comparison, not a fabricated different spelling.',
    'Original ekaram, branch of a tree, left vocabulary column.',
    'Original kaalmontu, feet, right vocabulary column.',
    'Original uḷantaari, young man, lower dot on l retained.',
    'Original meeṭṭa, step, lower dots on both t letters retained.',
    'Original peetti and both elder-relative gloss scopes agree; explicit kinship class missing.',
    'Original maṭayan and both younger-relative gloss scopes agree; lower dot on t retained; explicit kinship class missing.',
    'Original kaṛuttamma, Mother’s younger brother’s wife / Mother-in-law, lower dot on r and both scopes agree; explicit kinship class missing.',
    'Original maaḷa and both younger-relative gloss scopes agree; lower dot on l retained; explicit kinship class missing.',
]
assert len(selected) == len(observations) == 20
sample = []
for row, observation in zip(selected, observations):
    kin = ':kinship:' in row[10] or ':kin-plural:' in row[10]
    assert not kin or 'kinship' not in row[14].split()
    sample.append({'entry_key': row[10], 'form': row[2], 'gloss': row[3], 'tags': row[14],
                   'literal_form_and_gloss': 'pass', 'verdict': 'repair' if kin else 'pass',
                   'original_observation': observation})
assert sum(r['verdict'] == 'repair' for r in sample) == 5
report = {'reviewer': 'root', 'status': 'source-explicit kinship tagging repair required before acceptance',
    'seed': '20260927451', 'sampling': 'SHA256(seed:Entry_Key), first four per p45–49 stratum',
    'csv_sha256': sha(HERE / 'proposed.csv'), 'audit_sha256': sha(HERE / 'proposal-audit.jsonl'),
    'freeze_sha256': sha(HERE / 'review-freeze.json'),
    'source_pdf_sha256': '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af',
    'original_pages_visually_reread': list(range(45, 50)),
    'image_sha256': {str(p): sha(HERE / f'original-review/page-{p}-150dpi.png') for p in range(45, 50)},
    'sampled_rows': 20, 'literal_form_and_gloss_passes': 20, 'sample_passes': 15, 'sample_material_errors': 5,
    'sample': sample,
    'required_followup': ['Preserve exact frozen revision', 'Combine with independent whole-chapter reviewer findings',
                          'Apply kinship tag to all explicitly scoped kinship examples/heads',
                          'Prove lexical field and key continuity for tag-only repair', 'Fresh disjoint actual-output sample'],
    'limits': 'Sample evidence does not certify all unsampled fields. Whole literal/scope review is separately assigned to Norton. Full CLDF pipeline, compiled graph/identity checks, full suite and browser refresh remain deferred.'}
(HERE / 'root-original-review-pass1-20260927451.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'literal_form_and_gloss_passes': 20, 'passes': 15, 'material_errors': 5,
                  'report_sha256': sha(HERE / 'root-original-review-pass1-20260927451.json')}))
