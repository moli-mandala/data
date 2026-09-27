"""Stage full expression recovery without touching canonical inputs or existing audit."""
import csv,json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import grammar
import import_source
original=grammar.extend
grammar.extend=lambda rows,audit:original(rows,audit,P/'grammar-final-reviewed.tsv')
rows,audit=import_source.build()
with(P/'expression-proposal.csv').open('w',newline='')as f:csv.writer(f,lineterminator='\n').writerows(rows)
(P/'expression-proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n'for a in audit))
report={'stage':'proposal only; independent review pending','rows':len(rows),'audit_units':len(audit),'all_expression_exclusions_recovered':not any(a['status']=='excluded_sentence'for a in audit),'hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['expression-proposal.csv','expression-proposal-audit.jsonl','grammar-final-reviewed.tsv']}}
(P/'expression-proposal-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
