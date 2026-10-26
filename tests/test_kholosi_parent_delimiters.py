"""Source-parent serialization follows the actual unifier contract."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/kholosi_2021'
def test_only_six_parent_delimiters_change_and_all_compiler_references_resolve():
 old=list(csv.reader((P/'whole-proposal-r3/whole-proposed.csv').open()))
 new=list(csv.reader((P/'source-stage-parent-delimiter-addendum/whole-proposed.csv').open()))
 keys={r[10] for r in new}
 expected={c['entry_key']:c for c in json.loads((P/'source-stage-parent-delimiter-addendum/review.json').read_text())['changes']}
 assert len(new)==len(old)==1135 and len(expected)==6
 changed={}
 for a,b in zip(old,new):
  assert a[:13]+a[14:]==b[:13]+b[14:]
  if a[13]!=b[13]:changed[a[10]]=(a[13],b[13])
  # Same field delimiter as unify_cldf; not whitespace splitting.
  assert all(k in keys and k!=b[10] for k in b[13].split('|') if k)
  assert not b[11] or b[11] in keys
  assert not b[12] or b[12] in keys
 assert changed=={k:(c['old'],c['new']) for k,c in expected.items()}
 assert sum(len(r[13].split('|')) for r in new if r[13])==90
 assert sum(bool(r[11]) for r in new)==35
 assert not any(r[12] for r in new)
 # The previous encoding really failed the consumer's exact-key resolution.
 assert sum(any(k not in keys for k in r[13].split('|') if k) for r in old)==6
