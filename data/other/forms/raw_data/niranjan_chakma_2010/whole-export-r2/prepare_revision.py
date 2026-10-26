"""Write only revision2 preview and prove exact continuity with revision1."""
import csv,io,json
from pathlib import Path
from export_revision import prepare,sha,ROOT
R=Path(__file__).resolve().parent
def main():
 rows,audit,status=prepare();previous=list(csv.reader((ROOT/'whole-source-export-preview.csv').open()))
 # csv.writer serializes None as an empty cell; compare serialized fields.
 rows=[['' if value is None else value for value in row] for row in rows]
 before={r[10]:r for r in previous};after={r[10]:r for r in rows}
 assert all(after[k]==r for k,r in before.items())
 added=sorted(set(after)-set(before));assert len(added)==2 and len(rows)==3662
 for name,values,is_csv in [('whole-source-export-preview.csv',rows,True),('whole-source-export-audit.jsonl',audit,False)]:
  with (R/name).open('w',newline='') as f:
   if is_csv:csv.writer(f).writerows(values)
   else:f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
 status['outputs']={name:sha(R/name) for name in ['whole-source-export-preview.csv','whole-source-export-audit.jsonl']}
 status['revision_scripts']={name:sha(R/name) for name in ['assembly_revision.py','export_revision.py','prepare_revision.py']}
 status['previous_csv_sha256']=sha(ROOT/'whole-source-export-preview.csv')
 status['delta']={'unchanged_rows':len(before),'added_keys':added,'changed_existing_rows':0,'removed_rows':0}
 (R/'whole-source-export-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'rows':len(rows),'delta':status['delta'],'outputs':status['outputs']}))
if __name__=='__main__':main()
