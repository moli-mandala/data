"""Derive the six serializer-only repairs from immutable accepted r3."""
import csv,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
F=P/'whole-proposal-r3/whole-proposed.csv'
O=P/'source-stage-parent-delimiter-addendum'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=True)
 rows=list(csv.reader(F.open())); keys={r[10] for r in rows}; changes=[]
 for r in rows:
  if not r[13]:continue
  parents=r[13].split()
  assert all(p in keys for p in parents),(r[10],parents)
  if len(parents)>1:
   changes.append(dict(entry_key=r[10],old=r[13],new='|'.join(parents),parent_keys=parents))
   r[13]='|'.join(parents)
 assert len(changes)==6
 with (O/'whole-proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 original=list(csv.reader(F.open()))
 assert len(rows)==1135 and all(a[:13]+a[14:]==b[:13]+b[14:] for a,b in zip(original,rows))
 for r in rows:
  assert all(k in keys for k in r[13].split('|') if k)
 result=dict(status='proposed serializer correction; root review pending',rows=1135,affected_rows=6,
  old_csv_sha256=sha(F),new_csv_sha256=sha(O/'whole-proposed.csv'),changes=changes,
  compiler_contract='unify_cldf.py resolves Derivation_Parent_Keys with split("|"); spaces made six compound parent lists unresolved.',
  scope='Only field 13 delimiters change. Every literal, source key, parent identity and source claim is identical.',
  frozen_r3_unchanged=True,canonical_changed=False)
 (O/'review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
