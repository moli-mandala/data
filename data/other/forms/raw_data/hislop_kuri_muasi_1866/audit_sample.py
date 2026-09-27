"""Select full-source records including blanks and unresolved cells."""
import argparse,json,random
from import_source import build
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--count',type=int,default=20);p.add_argument('--exclude-report',action='append',default=[]);a=p.parse_args()
 excluded=set()
 for file in a.exclude_report:
  r=json.load(open(file));excluded.update(x['entry_key'] for x in r.get('entries',[]))
 rows,audit=build();sample=random.Random(a.seed).sample([r for r in audit if r['entry_key'] not in excluded],a.count)
 print(json.dumps(sample,ensure_ascii=False,indent=2))
