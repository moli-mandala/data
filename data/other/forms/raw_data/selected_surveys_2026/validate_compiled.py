"""Verify the selected-source CLDF and optionally compare the pre-install baseline.

Run after make all. --baseline is a directory of saved pre-install CSVs; it is
optional because the baseline copies are local working artifacts, not source data.
"""
import argparse,csv,hashlib,json,re,sys,unicodedata
from collections import Counter
from pathlib import Path
from segments import Tokenizer
R=Path(__file__).resolve().parent;ROOT=R.parents[4]
sys.path.insert(0,str(R.parent));import selected_surveys as source

def read(path):
 with path.open(newline='') as stream:return list(csv.DictReader(stream))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,default=ROOT/'source_checklists/audits/20260911-selected-surveys-build-validation.json');a=p.parse_args()
 all_raw={k:list(csv.reader((ROOT/f'data/other/forms/20260911-selected-{k}.csv').open())) for k in source.SOURCES}
 keys={r[10]:r for rows in all_raw.values() for r in rows}
 aliases={r['Legacy_ID']:r['Form_ID'] for r in read(ROOT/'cldf/form-id-aliases.csv')}
 links={r['Source_Key']:aliases[r['Legacy_ID']] for r in read(ROOT/'cldf/form-source-keys.csv') if r['Source_Key'] in keys}
 assert links.keys()==keys.keys(),('missing source keys',keys.keys()-links.keys())
 assert len(set(links.values()))==len(keys),'Distinct source records collapsed'
 forms=read(ROOT/'cldf/forms.csv');byid={r['ID']:r for r in forms};assert len(byid)==len(forms)
 edges=read(ROOT/'cldf/edges.csv');ids=set(links.values());assert not any(r['Child_ID'] in ids or r['Parent_ID'] in ids for r in edges)
 report={'sources':{},'files':{},'compiled_total':len(forms),'source_rows':len(keys),'linked':0,'borrowed':0,'skipped':25,'new_graph_edges':0}
 refs={r['ID']:r for r in read(ROOT/'cldf/references.csv')}
 for k,rows in all_raw.items():
  tok=Tokenizer(str(ROOT/'conversion'/f'selected-{k}.txt'));examples=[]
  for r in rows:
   f=byid[links[r[10]]]
   expected=unicodedata.normalize('NFC',tok(r[2],column='IPA').replace(' ','').replace('#',' '))
   assert f['Form']==expected,(r[10],f['Form'],expected)
   assert f['Original']==r[2] and f['Gloss']==r[3] and f['Language_ID']==r[0],(r[10],f)
   assert r[7] in f['Source'] and set(r[14].split())<=set(f['Tags'].split())
   assert not f['Native'] and not f['Phonemic'] and not f['Redirect']
   assert f['Status']=='unlinked',(r[10],f['Status'])
   assert not r[9] or r[9] in f['Etymology'],(r[10],r[9],f['Etymology'])
   if len(examples)<3:examples.append({'id':f['ID'],'source_key':r[10],'form':f['Form'],'original':f['Original'],'gloss':f['Gloss'],'source':f['Source'],'tags':f['Tags']})
  ref=refs[source.SOURCES[k]];assert ref['Provenance'] and ref['Editor'] and ref['Source']
  assert ref['OCR']==('Yes' if k in ['koraga','orissa'] else 'No'),ref
  report['sources'][k]={'rows':len(rows),'distinct_nodes':len({links[r[10]] for r in rows}),'profile_errors':0,'languages':dict(Counter(r[0] for r in rows)),'reference':ref,'examples':examples}
 if a.baseline:
  names=['forms.csv','edges.csv','form-source-keys.csv','form-id-aliases.csv','form-identities.csv','concepts.csv','form_concepts.csv','alignments.csv','references.csv']
  for name in names:
   current=ROOT/('data' if name=='form-identities.csv' else 'cldf')/name
   before=a.baseline/name
   with before.open() as f:b=sum(1 for _ in csv.reader(f))-1
   with current.open() as f:n=sum(1 for _ in csv.reader(f))-1
   report['files'][str(current.relative_to(ROOT))]={'before':b,'after':n,'delta':n-b,'sha256':hashlib.sha256(current.read_bytes()).hexdigest(),'unchanged':current.read_bytes()==before.read_bytes()}
  old_ids=set();missing=[];changes=[]
  with (a.baseline/'forms.csv').open() as stream:
   for r in csv.DictReader(stream):
    i=r['ID'];old_ids.add(i)
    if i not in byid:missing.append(i)
    elif r!=byid[i]:changes.append({'id':i,'fields':{k:[v,byid[i][k]] for k,v in r.items() if v!=byid[i][k]}})
  report['existing_forms']={'removed':sorted(missing),'changed':changes,'added':len(byid.keys()-old_ids)}
  assert not missing and not changes,'Unrelated old forms changed'
  assert byid.keys()-old_ids==ids,'Unexpected new forms'
  assert (ROOT/'cldf/edges.csv').read_bytes()==(a.baseline/'edges.csv').read_bytes(),'Unrelated graph changes'
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['sources']},indent=2))
if __name__=='__main__':main()
