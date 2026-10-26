from pathlib import Path
import json,hashlib,shutil
H=Path(__file__).resolve().parent;D=H/'revision1';D.mkdir(exist_ok=True)
review=json.loads((H.parent/'interlinear-independent-review/independent-review-pass1.json').read_text())
names=list(review['inputs'])+['continuous-semantic-proposal.jsonl','continuous-semantic-coverage.json','prepare_whole_expressions.py','prepare_c5_columns.py','prepare_continuous_alignment.py','continuous_alignment_specs.py','whole-original-review.json']
hashes={}
for name in names:
 src=H/name;digest=hashlib.sha256(src.read_bytes()).hexdigest()
 if name in review['inputs']:assert digest==review['inputs'][name]
 dst=D/name
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copyfile(src,dst)
 hashes[name]=digest
(D/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n');print('archived',len(hashes))
