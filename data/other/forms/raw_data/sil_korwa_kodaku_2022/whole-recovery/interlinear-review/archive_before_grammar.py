from pathlib import Path
import shutil,json,hashlib
H=Path(__file__).resolve().parent;out=H/'before-grammar-repair';out.mkdir(exist_ok=True)
names=['continuous-semantic-proposal.jsonl','continuous-semantic-coverage.json','prepare_continuous_alignment.py','review-repair-verification.json']
hashes={}
for name in names:
 p=H/name;target=out/name
 if target.exists():assert target.read_bytes()==p.read_bytes()
 else:shutil.copyfile(p,target)
 hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n');print('archived',len(names))
