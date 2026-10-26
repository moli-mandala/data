"""One-time exact Behera-independent Cholanaickan citation formatting repair."""
import hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parent
D=next(p for p in P.parents if (p/'tags.py').exists())
B=D/'cldf/sources.bib';proposal=P/'proposed-reference.bib';R=P/'installation-receipt.json'
def sha(x):return hashlib.sha256(x).hexdigest()
r=json.loads(R.read_text());before=B.read_bytes();old=proposal.read_bytes()
assert sha(before)==r['outputs']['bib']
replacements={b'title = {Tribal Languages of Kerala}':b'title = {{Tribal Languages of Kerala}}',
              b'title = {A Descriptive Study of Cholanaikka}':b'title = {{A Descriptive Study of Cholanaikka}}'}
new=old
for a,b in replacements.items():
    assert new.count(a)==1 and before.count(a)>=1
    new=new.replace(a,b)
assert before.count(old)==1
patched=before.replace(old,new)
assert sha(patched)!=sha(before)
(P/'pre-title-amendment-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
for target,content in ((proposal,new),(B,patched)):
    temp=target.with_name(target.name+'.cholanaickan-title-tmp');assert not temp.exists()
    temp.write_bytes(content);os.replace(temp,target)
r['outputs']['bib']=sha(patched)
r['reference_title_amendment']={'old_bib_sha256':sha(before),'new_bib_sha256':sha(patched),'scope':'braced two Cholanaickan source titles only; other bibliography bytes identical'}
temp=R.with_suffix('.title-tmp');temp.write_text(json.dumps(r,indent=2)+'\n');os.replace(temp,R)
(P/'reference-title-amendment.json').write_text(json.dumps(r['reference_title_amendment'],indent=2)+'\n')
print(r['reference_title_amendment'])
