"""Record root's independent manual reread of the frozen graph-bearing candidate."""
import csv
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
package = here.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(package / 'proposed.csv') == '3e52cbdfe68505abb3411eebbda74eb031938d38a5baeb60163811c1311dcc27'
assert sha(package / 'proposal-audit.jsonl') == 'af8a3e1212d7da3bbc2e42c13f4684c5984bc2d8ba09b8e5c433e094975bfd9e'
previous = {r['entry_key'] for r in json.loads((here / 'original-review-pass1-20260927430.json').read_text())['sampled']}
rows = list(csv.reader((package / 'proposed.csv').open()))
selected = []
for page in range(4, 8):
    pool = [r for r in rows if f':p{page}:' in r[10] and r[10] not in previous]
    selected += sorted(pool, key=lambda r: hashlib.sha256(('20260927431:' + r[10]).encode()).hexdigest())[:5]
observations = [
    'Original Adiya second row pakeelu, day-time.',
    'Original Adiya third row ba ṭṭi, basket; visible internal spacing and two lower dots retained.',
    'Original Malayalam second row arakkŭ, lac; control column correct.',
    'Original Malayalam kaṇṇŭ, eye; breve and lower dots retained.',
    'Original Malayalam akalam, distance; table scope correct.',
    'Original feminine plural era (Prox.); blank gloss and feminine/plural/proximal tags agree.',
    'Un-underlined naanu in explicit first-person free-variation prose; source occurrence kept distinct.',
    'Original Adiya peṇiyattiya, Paniya Women; plurality scoped from table.',
    'Original exclusive first-person plural naanka with initial drawn lowline retained as U+0332.',
    'Original non-feminine singular eenu; no fabricated masculine-only tag.',
    'Original atu enṛa appey with father clause gloss and respectful elder-kin scope.',
    'Original future n̲aanu kaaṇuvey, I will see; drawn initial-n lowline retained.',
    'Original parenthetical Malayalam itŭ el̲uttŭ; drawn l lowline, breves, and control identity agree.',
    'Original prose –va, en dash retained; addressee-sex meaning remains in note.',
    'Original parenthetical Malayalam ivan aaṇkuṭṭi, correctly assigned control identity.',
    'Original wife ṛaatti; high-resolution crop explicitly confirms lower dot under r.',
    'Original father\'s elder brother peerappan in slash pair; scope agrees.',
    'Original negative –a, literal en dash and negative suffix tag agree.',
    'Original aakki baruva kaaṇi, elder sister has not come; neg clause scope agrees.',
    'Original baavey niiŋka barii, brother-in-law come; literal ŋ spelling retained.',
]
assert len(selected) == len(observations) == 20
by_key = {r[10]: r for r in rows}
graph = json.loads((package / 'graph-approved.json').read_text())
for edge in graph['proposals']:
    child, parent = by_key[edge['child_key']], by_key[edge['variant_of_key']]
    assert child[11] == parent[10]
    assert child[2] == edge['child_form'] and parent[2] == edge['head_form']
assert sum(bool(r[11]) for r in rows) == len(graph['proposals']) == 12
report = {
    'reviewer': 'root', 'status': 'fresh independent original/output sample passes; source-stage installation approved for pinned inputs',
    'seed': '20260927431', 'sampling': 'SHA256(seed:Entry_Key), five per p4–7 stratum excluding all prior twenty keys',
    'csv_sha256': sha(package / 'proposed.csv'), 'audit_sha256': sha(package / 'proposal-audit.jsonl'),
    'freeze_sha256': sha(package / 'review-freeze.json'),
    'source_pdf_sha256': '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af',
    'render_manifest': json.loads((here / 'render-manifest.json').read_text()),
    'extra_crop_sha256': sha(here / 'p7-kinship-highres.png'),
    'sample_passes': 20, 'sample_material_errors': 0,
    'sampled': [{'entry_key': r[10], 'form': r[2], 'gloss': r[3], 'tags': r[14], 'verdict': 'pass', 'original_observation': o} for r, o in zip(selected, observations)],
    'graph_review': {'accepted_source_explicit_edges': 12, 'actual_csv_endpoints_verified': 12, 'held_non_edge_classes': 5, 'source_evidence': 'Original p5 free-variation statement and slash groups; original p7 slash groups reread. No borrowing or derivation asserted.'},
    'prior_repair': 'Eight marker spellings independently reread in marker-repair-targeted-addendum.json; final graph revision changes only twelve relation fields.',
    'limits': 'Source-stage review only. Full CLDF pipeline, compiled identity/graph checks, full suite and browser refresh remain deferred under user instructions. Unsampled fields are not independently certified by this sample.'
}
(here / 'original-review-final-20260927431.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passes': 20, 'material_errors': 0, 'edges': 12, 'report_sha256': sha(here / 'original-review-final-20260927431.json')}))
