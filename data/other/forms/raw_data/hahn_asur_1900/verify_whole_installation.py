"""Read-only scoped check; run after independent approval and canonical installation."""
import csv,hashlib,io,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P.parents[4]
os.chdir(DATA);sys.path.insert(0,str(DATA))
import make_cldf,make_refs,source_meta,profile_policy,tags
from unify_cldf import citation_keys
from segments import Tokenizer
from pybtex.database import parse_file
from pybtex import PybtexEngine
freeze=json.loads((P/'whole-source-freeze-20260926.json').read_text())
for f,h in freeze['hashes'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
canonical=DATA/'data/other/forms/20260925-hahn-asur.csv'
for original,target in [('proposal.csv',canonical),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/hahn-asur-1900.txt')]:
 assert (P/original).read_bytes()==target.read_bytes(),str(target)
rows=list(csv.reader(canonical.open()));by={r[10]:r for r in rows};assert len(rows)==835
old=list(csv.reader((P/'historical-lexical.csv').open()));assert len(old)==621 and {r[10] for r in old}<=set(by)
assert hashlib.sha256((DATA/'data/form-identities.csv').read_bytes()).hexdigest()=='5ec2588f97a569cc7a1de68f59474c03612fa7a4266582be61d70dca29efcfc9'
errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(canonical),errors,name='20260925-hahn-asur');assert not errors.getvalue(),errors.getvalue();assert len(parsed)==stats['converted']==835
for r in parsed:
 raw=by[r.entry_key];assert (r.old_form,r.native,r.ipa,r.source,r.tags)==(raw[2],raw[4],raw[5],raw[7],raw[14]);assert citation_keys(r.source)==set(make_refs.source_ids(r.source))=={'hahn1900asur'}
 assert not any(raw[i] for i in [11,12,13])
t=Tokenizer(str(DATA/'conversion/hahn-asur-1900.txt'));assert all('�' not in t(r[2],column='IPA') for r in rows)
entry=parse_file('cldf/sources.bib').entries['hahn1900asur'];formatted=PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown');assert '1900' in formatted and 'Hahn' in formatted
with (DATA/'cldf/dialects.csv').open() as f:
 lect=next(r for r in csv.reader(f) if r and r[0]=='hahn-1900-asur-dukma')
assert lect[2]=='Asuri' and not any(lect[5:8])
print(json.dumps({'status':'passed','canonical_rows':835,'old_keys_preserved':621,'audit_records':670,'reference':formatted,'canonical_hash':hashlib.sha256(canonical.read_bytes()).hexdigest(),'durable_identity_unchanged':True,'full_database_built':False},ensure_ascii=False,indent=2))
