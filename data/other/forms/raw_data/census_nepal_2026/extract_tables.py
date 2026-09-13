import pdfplumber,json,re,collections
from pathlib import Path
r=Path(__file__).resolve().parent
conf={'tamil-nadu':(869,913,20,12),'uttar-pradesh':(767,809,19,12),'bihar':(684,705,18,7),'sikkim2':(138,157,1,3),'danuwar':(125,132,10,5)}
for k,(a,b,offset,n) in conf.items():
 p=pdfplumber.open(r/'danuwar-canonical.pdf' if k=='danuwar' else '/tmp/'+k+'-normalized.pdf');out=[]
 for pi in range(a,b):
  page=p.pages[pi]
  for tab in page.find_tables():
   for ri,row in enumerate(tab.extract()):
    pairs=[(v,bb) for v,bb in zip(row,tab.rows[ri].cells) if bb is not None]
    row=[v for v,bb in pairs]
    boxes=[bb for v,bb in pairs]
    if k=='danuwar':
     ni=next((i for i,v in enumerate(row) if re.fullmatch(r'\d+\.',v or '')),None)
     if ni is None:continue
     gi=next(i for i in range(ni+1,len(row)) if row[i] and not re.fullmatch(r'\d+\.',row[i]))
     order=[ni,gi]+list(range(len(row)-5,len(row)))
    elif k in ('tamil-nadu','uttar-pradesh') and (pi-a)%2:order=[6,7,0,1,2,3,4,5]
    else:order=list(range(8 if k in ('tamil-nadu','uttar-pradesh') else n+2))
    if len(row)<len(order):continue
    vals=[row[j] for j in order]
    if not re.fullmatch(r'\d+\.?',vals[0] or '') or re.fullmatch(r'\d+',vals[1] or ''):continue
    num=int(vals[0].strip('.'))
    if not 1<=num<=500:continue
    for ci,v in enumerate(vals[2:]):
     bb=boxes[order[ci+2]]
     if bb is None:raise ValueError((k,pi,num,ci,row))
     ims=[{'name':im['name'],'bbox':[im['x0'],im['top'],im['x1'],im['bottom']]} for im in page.images if bb[0]<(im['x0']+im['x1'])/2<bb[2] and bb[1]<(im['top']+im['bottom'])/2<bb[3]]
     col=ci+(6 if k in ('tamil-nadu','uttar-pradesh') and (pi-a)%2 else 0)
     out.append({'item':num,'column':col,'pdf_page':pi+1,'printed_page':pi+1-offset,'gloss':vals[1],'text':v or '', 'bbox':bb,'images':ims})
 c=collections.Counter(x['column'] for x in out);keys=[(x['item'],x['column']) for x in out];print(k,len(out),dict(c),'dup',len(keys)-len(set(keys)),'images',sum(bool(x['images']) for x in out),flush=True)
 missing=[(i,j) for i in range(1,211 if k=='danuwar' else 501) for j in range(n) if (i,j) not in keys];print('missing',missing[:60],flush=True)
 (r/(k+'-cells.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
