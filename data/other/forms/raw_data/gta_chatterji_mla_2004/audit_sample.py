"""Print every emitted row for a reproducible sample of complete raw records."""
import argparse,json,random
from import_source import prepare

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=2026092610);p.add_argument('--size',type=int,default=20);args=p.parse_args()
 rows,decisions=prepare();by_key={r[10]:r for r in rows}
 for d in sorted(random.Random(args.seed).sample(decisions,args.size),key=lambda d:d['body_line']):
  print(json.dumps({'id':d['source_id'],'body_line':d['body_line'],'raw':d['raw'],'rows':[by_key[x['entry_key']] for x in d['rows']]},ensure_ascii=False))
if __name__=='__main__':main()
