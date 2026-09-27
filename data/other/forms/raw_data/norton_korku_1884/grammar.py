import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE='cust1884korku'
DIALECT='dialect:ko:norton-1884:Korku%20%28Norton%201884%29'
def extend(rows,audit,inventory_path=None):
 for item in csv.DictReader((inventory_path or ROOT/'grammar_inventory.tsv').open(),delimiter='\t'):
  page,n=int(item['page']),int(item['item'])
  key=f'{SOURCE}:notes:p{page}:item{n:02}'
  forms=item['forms'].split('|') if item['forms'] and item['status']=='selected' else []
  audit.append(dict(entry_key=key,printed_page=page,pdf_page=page+19,column='notes',item=n,section='Grammatical notes',gloss=item['gloss'],printed_response=item.get('source_forms') or item['forms'],selected_forms=forms,status=item['status'],note=item['note']))
  for i,form in enumerate(forms,1):
   parent=f'{SOURCE}:notes:p179:item{n-1:02}' if page==179 and n in (3,5) else ''
   etymology=item['note'] if page==179 and n in (7,8) else ''
   rows.append(['ko','',form.casefold(),item['gloss'],'','','',f'{SOURCE}[p. {page}, Notes, lexical item {n}]','',etymology,key if len(forms)==1 else f'{key}:alt{i}','','',parent,DIALECT+' '+item['tags']])
 return rows,audit
