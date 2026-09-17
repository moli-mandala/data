import csv,gzip,io,json,sys
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=root.parent/'tmp/berger-audit-20260914';sys.path.insert(0,str(root))
from segments import Tokenizer
bykey={r[10]:r for p in ['20260726-berger-auto.csv','20220930-berger.csv'] for r in csv.reader((root/'data/other/forms'/p).open())}
with (root/'data/form-identities.csv').open() as f:
 registry=[r for r in csv.DictReader(f) if r['Source_Key'] in bykey and r['Status']=='active']
ids={r['Form_ID']:r['Source_Key'] for r in registry};keyids={v:k for k,v in ids.items()}
with (root/'cldf/edges.csv').open() as f:
 edges=[r for r in csv.DictReader(f) if r['Child_ID'] in ids and r['Rank']=='1']
wanted=set(ids)|{r['Parent_ID'] for r in edges}
with (root/'cldf/forms.csv').open() as f:
 forms={r['ID']:r for r in csv.DictReader(f) if r['ID'] in wanted}
with gzip.open(root/'data/other/forms/raw_data/20260828-berger-audit.csv.gz','rt') as f: audit={r['Installed_Key']:r for r in csv.DictReader(f)}
bad=json.loads((out/'summary.json').read_text())['graph']['missing_endpoints']
bad_details=[{'child_key':k,'kind':kind,'missing_parent_key':parent,'parent_audit':audit.get(parent),'compiled_child':forms.get(keyids.get(k,'')),'compiled_edges':[e for e in edges if e['Child_ID']==keyids.get(k)]} for k,kind,parent in bad]
keyexamples=['berger-entry-1712','berger-entry-2920','berger-entry-7051','berger-entry-7630']
examples=[]
for key in keyexamples:
 fid=keyids.get(key);ee=[e for e in edges if e['Child_ID']==fid]
 examples.append({'key':key,'id':fid,'form':forms.get(fid),'edges':ee,'parents':[forms.get(e['Parent_ID']) for e in ee]})
profile=Tokenizer(str(root/'conversion/berger.txt'))
coverage=[{'key':key,'form':r[2],'converted':profile(r[2],column='IPA')} for key,r in bykey.items() if '�' in profile(r[2],column='IPA')]
result={'active_source_keys':len(ids),'compiled_matching_forms':len(ids.keys()&forms.keys()),'compiled_grammar':dict(Counter(t for fid in ids if fid in forms for t in forms[fid]['Tags'].split())),'compiled_bad_reference_children':len({d['child_key'] for d in bad_details if d['compiled_child']}),'compiled_bad_reference_edges':bad_details,'key_drift_examples':examples,'profile_uncovered':coverage}
(out/'compiled.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Active keys',len(ids),'compiled',len(ids.keys()&forms.keys()),'profile failures',len(coverage))
for ex in examples: print(ex['key'],ex['id'],ex['form']['Original'] if ex['form'] else None,ex['form']['Gloss'] if ex['form'] else None,'parents',[(p['ID'],p['Gloss']) for p in ex['parents'] if p])
print('Missing parent audit status',dict(Counter(d['parent_audit']['Status'] if d['parent_audit'] else 'no audit' for d in bad_details)))
print('Compiled child edge types',dict(Counter(e['Kind'] for d in bad_details for e in d['compiled_edges'])))
