import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[5];B=Path('/tmp/jambu-selected-before')
def read(p):
 with p.open() as stream:yield from csv.DictReader(stream)
def digest(values):return hashlib.sha256(json.dumps(list(values),ensure_ascii=False).encode()).digest()
def cm(p):
 d=defaultdict(set)
 for r in read(p):d[r['Form_ID']].add(r['Concept_ID'])
 return d
old,new=cm(B/'form_concepts.csv'),cm(R/'cldf/form_concepts.csv')
oldforms={r['ID']:{k:r[k] for k in ['Form','Gloss']} for r in read(B/'forms.csv')};before={r['ID']:r for r in read(B/'concepts.csv')};after={r['ID']:r for r in read(R/'cldf/concepts.csv')}
changes=[{'id':i,'form':oldforms[i]['Form'],'gloss':oldforms[i]['Gloss'],'before':sorted(old.get(i,set())),'after':sorted(new.get(i,set()))} for i in oldforms if old.get(i,set())!=new.get(i,set())]
out={'new_concepts':[r for i,r in after.items() if i not in before],'removed_concepts':[r for i,r in before.items() if i not in after],'changed_concept_labels':[i for i in before.keys()&after.keys() if before[i]!=after[i]],'existing_forms_changed_memberships':changes,'new_form_memberships':sum(len(v) for i,v in new.items() if i not in oldforms)}
del oldforms,old,new
for name,cols in [('form-source-keys.csv',['Legacy_ID','Source_Key']),('form-id-aliases.csv',['Legacy_ID','Form_ID'])]:
 b={digest(r[c] for c in cols) for r in read(B/name)};a={digest(r[c] for c in cols) for r in read(R/'cldf'/name)};out[name]={'removed':len(b-a),'added':len(a-b)};assert not b-a
b={digest(r.values()) for r in read(B/'form-identities.csv')};a={digest(r.values()) for r in read(R/'data/form-identities.csv')};out['form-identities.csv']={'removed':len(b-a),'added':len(a-b)};assert not b-a
p=R/'source_checklists/audits/20260911-selected-surveys-generated-diff.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
