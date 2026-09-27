"""Install approved complete expressions only; no database or full build."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;DATA=P.parents[4]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate():
 r=json.loads((P/'independent-expression-output-audit-20260926-pass1.json').read_text())
 assert r['status']=='passed' and r['material_errors']==0 and r['sample_size']==20
 for name,expected in json.loads((P/'expression-freeze.json').read_text())['hashes'].items():
  assert r['hashes'][name]==expected and sha(P/name)==expected,name
 reconciliation=json.loads((P/'grammar-recovery-reconciliation.json').read_text())
 assert sha(P/'grammar-full-reviewed.tsv')==reconciliation['input_sha256']
 assert sha(P/'independent-full-grammar-recovery-review-20260926.json')==reconciliation['review_sha256']
 assert sha(P/'grammar-final-reviewed.tsv')==reconciliation['output_sha256']
 final={(int(a['page']),int(a['item'])):a for a in csv.DictReader((P/'grammar-final-reviewed.tsv').open(),delimiter='\t')}
 for c in reconciliation['corrections']:assert final[c['page'],c['item']]['forms']==c['after']
 rows=list(csv.reader((P/'expression-proposal.csv').open()));assert len(rows)==1046 and len({a[10]for a in rows})==1046
 audit=[json.loads(x)for x in(P/'expression-proposal-audit.jsonl').read_text().splitlines()]
 assert len(audit)==958 and not any(a['status']=='excluded_sentence'for a in audit)
 return rows,audit

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true');args=parser.parse_args();rows,audit=validate()
 if args.install:
  for src,name in [(P/'grammar_inventory.tsv','grammar-inventory-before-expression-recovery.tsv'),(P/'audit.jsonl','audit-before-expression-recovery.jsonl'),(DATA/'data/other/forms/20260925-cust-norton-korku.csv','canonical-before-expression-recovery.csv')]:
   if not(P/name).exists():shutil.copyfile(src,P/name)
  for source,target in [('expression-proposal.csv',DATA/'data/other/forms/20260925-cust-norton-korku.csv'),('expression-proposal-audit.jsonl',P/'audit.jsonl'),('grammar-final-reviewed.tsv',P/'grammar_inventory.tsv'),('expression-source.yaml',DATA/'data/other/forms/20260925-cust-norton-korku.yaml')]:shutil.copyfile(P/source,target)
 print('1046 source rows /958 audit units validated; no database build')
if __name__=='__main__':main()
