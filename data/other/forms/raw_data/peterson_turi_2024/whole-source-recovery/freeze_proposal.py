"""Pin review artifacts and draw a reproducible fresh stratified independent sample."""
import csv,hashlib,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((HERE/'proposal/summary.json').read_text())
files={}
for p in sorted(HERE.glob('*.jsonl')):files[p.name]=sha(p)
for p in sorted((HERE/'proposal').iterdir()):
 if p.is_file():files[str(p.relative_to(HERE))]=sha(p)
for name in ['page-census.json','source-manifest.json','baseline-installed.json','owner-texts-review.json','owner-grammar-review.json','owner-inline-review.json','auxiliary-reference-review.json','dialect-proposal.json','HANDOFF.md','build_proposal.py','install_proposal.py']:
 files[name]=sha(HERE/name)
manifest=dict(status='frozen_for_independent_audit_not_installed',source='peterson2024turi',original_sha256=json.loads((HERE/'source-manifest.json').read_text())['sha256'],rows=summary['total_rows'],audit_units=summary['audit_units'],baseline_csv_sha256=summary['canonical_unchanged_sha256'],files=files,focused_tests=dict(path='tests/test_peterson_turi_whole_proposal.py',passed=5,failed=0))
(HERE/'frozen-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
audit=[json.loads(x) for x in (HERE/'proposal/whole-audit.jsonl').read_text().splitlines()]
indexed={r[10]:r for p in (HERE/'proposal').glob('*.csv') for r in csv.reader(p.open())}
old=set(json.loads((HERE.parent/'visual-review.json').read_text())['sample_keys'])
strata={'legacy':[],'grammar':[],'texts':[],'historical':[]}
for a in audit:
 if not a['entry_keys']:continue
 if a['source_unit_key'] in old:continue
 u=a['evidence']
 if a['section']=='appendix1':s='legacy'
 elif u.get('lect','').startswith('historical'):s='historical'
 elif 'example' in u and int(''.join(c for c in u['example'].split('-')[0] if c.isdigit()))>=44:s='texts'
 else:s='grammar'
 strata[s].append(a)
rng=random.Random(20260927111);selected=[]
for s,n in [('legacy',5),('grammar',5),('texts',6),('historical',4)]:
 for a in rng.sample(strata[s],n):
  selected.append(dict(stratum=s,source_unit_key=a['source_unit_key'],printed_page=a['evidence']['printed_page'],pdf_page=a['evidence']['printed_page']-260,audit_unit=a,actual_csv_rows=[indexed[k] for k in a['entry_keys']]))
(HERE/'independent-sample-20260927.json').write_text(json.dumps(dict(seed=20260927111,status='selected_not_audited',frozen_manifest_sha256=sha(HERE/'frozen-manifest.json'),prior_legacy_sample_excluded=sorted(old),selected=selected),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(HERE/'frozen-manifest.json'),rows=manifest['rows'],audit_units=manifest['audit_units'],sampled_units=len(selected),sampled_output_rows=sum(len(x['actual_csv_rows']) for x in selected))))
