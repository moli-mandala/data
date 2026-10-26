"""Remove repeated shared-volume provenance sentence, preserving all other bytes."""
import hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parent;D=next(x for x in P.parents if (x/'tags.py').exists())
B=D/'cldf/sources.bib';Q=P/'proposed-reference.bib';R=P/'installation-receipt.json'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads(R.read_text());before=B.read_bytes();old=Q.read_bytes();assert sha(before)==r['outputs']['bib'] and before.count(old)==1
needle=b'. The same 58-page volume is used by the separately cited nair2013aranadan chapter. This is the same 58-page volume used by the separately cited nair2013aranadan chapter, not a second edition'
replacement=b'. This is the same 58-page volume used by the separately cited nair2013aranadan chapter, not a second edition'
assert old.count(needle)==1
new=old.replace(needle,replacement);patched=before.replace(old,new)
for target,content in ((Q,new),(B,patched)):
 tmp=target.with_name(target.name+'.cholanaickan-dedup-tmp');assert not tmp.exists();tmp.write_bytes(content);os.replace(tmp,target)
r['outputs']['bib']=sha(patched)
r['reference_provenance_dedup']={'old_bib_sha256':sha(before),'new_bib_sha256':sha(patched),'scope':'remove duplicated shared-volume provenance sentence only'}
tmp=R.with_suffix('.dedup-tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');os.replace(tmp,R)
(P/'reference-provenance-dedup.json').write_text(json.dumps(r['reference_provenance_dedup'],indent=2)+'\n')
print(r['reference_provenance_dedup'])
