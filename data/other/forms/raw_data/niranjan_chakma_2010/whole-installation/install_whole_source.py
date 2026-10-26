"""Guarded whole-source installer. --check never changes canonical inputs."""
import argparse,csv,hashlib,io,json,os,tempfile,sys
from pathlib import Path
Q=Path(__file__).resolve().parent;P=Q.parent;D=P.parents[4]
R=P/'whole-export-r2'
sys.path.insert(0,str(R))
from export_revision import prepare
CSV=D/'data/other/forms/20260921-niranjan-chakma.csv';YAML=CSV.with_suffix('.yaml')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def checked():
 manifest=json.loads((R/'whole-source-export-status.json').read_text())
 preparation=json.loads((Q/'preparation.json').read_text())
 assert sha(Q/'legacy-742.csv')==preparation['baseline_sha256']
 for name,h in preparation['script_sha256'].items():assert sha(Q/name)==h,name
 for name,h in preparation['source_script_sha256'].items():assert sha(P/name)==h,name
 for name,h in manifest['outputs'].items():assert sha(R/name)==h,name
 for name,h in manifest['revision_scripts'].items():assert sha(R/name)==h,name
 rows,audit,_=prepare();stream=io.StringIO(newline='');csv.writer(stream).writerows(rows)
 csvbytes=stream.getvalue().encode();auditbytes=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit).encode()
 assert csvbytes==(R/'whole-source-export-preview.csv').read_bytes(),'CSV regeneration differs'
 assert auditbytes==(R/'whole-source-export-audit.jsonl').read_bytes(),'audit regeneration differs'
 baseline=list(csv.reader((Q/'legacy-742.csv').open()))
 assert len(rows)==3662 and len(audit)==5232
 assert [r[10] for r in rows[:742]]==[r[10] for r in baseline]
 assert len({r[10] for r in rows})==len(rows)
 assert sha(CSV) in [preparation['baseline_sha256'],digest(csvbytes)],'unexpected canonical CSV revision'
 assert sha(YAML) in [preparation['original_yaml_sha256'],sha(Q/'metadata-proposed.yaml')],'unexpected metadata revision'
 result={'status':'byte-identical regeneration verified; installation not authorized by this check','rows':len(rows),'audit_units':len(audit),'preserved_keys':742,'csv_sha256':digest(csvbytes),'audit_sha256':digest(auditbytes),'metadata_sha256':sha(Q/'metadata-proposed.yaml'),'preparation_sha256':sha(Q/'preparation.json'),'coordinator_sha256':sha(Path(__file__)),'review_manifest_sha256':sha(R/'whole-source-export-status.json')}
 return result,csvbytes,auditbytes
def approval(result):
 p=Q/'root-installation-approval.json'
 if not p.exists():raise RuntimeError('No root installation approval; canonical source remains unchanged.')
 a=json.loads(p.read_text());assert a['status']=='approved' and a['material_errors']==0
 for k in ['csv_sha256','audit_sha256','metadata_sha256','preparation_sha256','coordinator_sha256','review_manifest_sha256']:assert a[k]==result[k],k
 for name,h in a['independent_review_sha256'].items():assert sha(P/name)==h,name
 assert a['independent_review_sha256'],'independent source review required'
def replace(path,data):
 # Staging file lives beside its destination, inside the Jambu workspace.
 with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.chakma-',delete=False) as f:f.write(data);temporary=Path(f.name)
 os.replace(temporary,path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--install',action='store_true');args=parser.parse_args()
 if args.check==args.install:parser.error('Select exactly one of --check or --install')
 result,csvbytes,auditbytes=checked()
 if args.install:
  approval(result)
  replace(CSV,csvbytes);replace(YAML,(Q/'metadata-proposed.yaml').read_bytes());replace(P/'whole-installed-audit.jsonl',auditbytes)
  result['status']='source-stage installed; full build/browser gates deferred'
  result['canonical_sha256']={str(CSV.relative_to(D)):sha(CSV),str(YAML.relative_to(D)):sha(YAML)}
  (Q/'installation-result.json').write_text(json.dumps(result,indent=2)+'\n')
 else:(Q/'regeneration-check.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
