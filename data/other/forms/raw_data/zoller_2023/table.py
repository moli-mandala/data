import pymupdf,json,re,sys
from pathlib import Path
from extract import decode_page
from languages import PAT
RAW=Path(__file__).parent
p=pymupdf.open(sys.argv[1])
rr=[]
for page in range(864,873):
 pg=p[page+28];delta=0 if page%2==0 else -7.38
 cols=[decode_page(pg,(a+delta,b+delta)) for a,b in [(73,185),(185,252),(252,365),(365,438)]]
 starts=[l for l in cols[0] if l['y']>65 and re.match(r'^[?*√]',l['text'])]
 for i,l in enumerate(starts):
  y=l['y'];end=starts[i+1]['y']-.1 if i+1<len(starts) else 630
  cells=[[x for x in col if y-.2<=x['y']<end and 9<x['size']<9.1] for col in cols]
  tokens=[]
  # The South Asia column is geographical coverage, NOT an attribution of the example to every listed language.
  for n,cell in enumerate([cells[0],cells[2],cells[3]]):
   tokens.append({'style':'r','text':('PIE ' if n==0 else ' — ' if n==1 else ' [Comment: ')})
   if n==1:
    cov=' '.join(x['text'] for x in cells[1]);ex=' '.join(x['text'] for x in cell)
    labs=list(PAT.finditer(cov))
    if len(labs)==1 and cov.strip()==labs[0][1] and not PAT.search(ex):tokens.append({'style':'r','text':labs[0][1]+' '})
   for line in cell:
    tokens+=line['tokens'];tokens.append({'style':'r','text':' '})
  tokens.append({'style':'r','text':']'})
  rr.append({'key':f'zoller2023:18.6:p{page}:row{i+1}','section':'18.6','number':i+1,'page':page,'lines':sum(cells,[]),'tokens':tokens,'text':''.join(t['text'] for t in tokens),'coverage_column':' '.join(x['text'] for x in cells[1])})
(RAW/'table-records.json').write_text(json.dumps(rr,ensure_ascii=False))
print('Table rows:',len(rr))
for r in rr[:3]:print(r['text'])
