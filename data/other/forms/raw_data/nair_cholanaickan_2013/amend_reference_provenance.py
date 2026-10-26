"""One-time exact citation provenance correction after source-stage installation."""
import hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parent;D=next(x for x in P.parents if (x/'tags.py').exists())
B=D/'cldf/sources.bib';Q=P/'proposed-reference.bib';R=P/'installation-receipt.json'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads(R.read_text());before=B.read_bytes();old=Q.read_bytes()
assert sha(before)==r['outputs']['bib'] and before.count(old)==1
needle=b'Source-local nair_cholanaickan_2013/proposed.csv, proposal-audit.jsonl, physical-units-first-reading.jsonl and pinned original PDF/render evidence; source-stage only, not installed'
replacement=b'data/other/forms/20260927-nair-cholanaickan.csv; data/other/forms/raw_data/nair_cholanaickan_2013/installed-audit.jsonl; physical-units-first-reading.jsonl and pinned original PDF/render evidence. The same 58-page volume is used by the separately cited nair2013aranadan chapter'
assert old.count(needle)==1
new=old.replace(needle,replacement);patched=before.replace(old,new)
assert patched!=before
for target,content in ((Q,new),(B,patched)):
 tmp=target.with_name(target.name+'.cholanaickan-provenance-tmp');assert not tmp.exists();tmp.write_bytes(content);os.replace(tmp,target)
r['outputs']['bib']=sha(patched)
r['reference_provenance_amendment']={'old_bib_sha256':sha(before),'new_bib_sha256':sha(patched),'scope':'replace stale proposal-only provenance with actual canonical CSV/audit and shared-volume note'}
tmp=R.with_suffix('.provenance-tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');os.replace(tmp,R)
(P/'reference-provenance-amendment.json').write_text(json.dumps(r['reference_provenance_amendment'],indent=2)+'\n')
print(r['reference_provenance_amendment'])
