import gzip,json,re,collections
from pathlib import Path
RAW=Path(__file__).parent

def records(pages):
 units=[];cur=None;section='';footnotes=[]
 for page in pages:
  p=page['printed_page']
  if p<519:continue
  for l in page['lines']:
   text=l['text'];size=l['size']
   if re.match(r'^18\.\d',text) and size>9.5 and l['y']>55:
    m=re.match(r'^(18(?:\.\d+)+)',text);section=m[1];cur=None
   if 8<size<8.1:
    footnotes.append({'page':p,**l});continue
   if not (9<size<9.1) or l['y']<55:continue
   m=re.match(r'^(\d+)\. ',text)
   if m and l['x']<95:
    key=f'zoller2023:{section}:p{p}:{m[1]}'
    cur={'key':key,'section':section,'number':int(m[1]),'page':p,'lines':[]}
    units.append(cur)
   if cur:cur['lines'].append({'page':p,**l})
 for u in units:
  ts=[]
  for l in u['lines']:
   if ts:ts.append(dict(style='r',text=' '))
   ts.extend(l['tokens'])
  u['tokens']=ts;u['text']=''.join(t['text'] for t in ts)
 # Footnotes have their own stable printed numbers; retain even notes with no extractable lemmata.
 note=None
 for line in footnotes:
  m=re.match(r'^(\d+)\s+',line['text'])
  if m and line['x']<65:
   note={'key':f"zoller2023:footnote:p{line['page']}:{m[1]}",'section':'footnote','number':int(m[1]),'page':line['page'],'lines':[],'tokens':[]}
   units.append(note)
  if note:
   note['lines'].append(line)
   note['tokens']+=line['tokens']+[{'style':'r','text':' '}]
   note['text']=''.join(t['text'] for t in note['tokens'])
 return units,footnotes

if __name__=='__main__':
 pages=[json.loads(l) for l in gzip.open(RAW/'pages.jsonl.gz','rt')]
 rr,ff=records(pages)
 rr+=json.loads((RAW/'table-records.json').read_text()) if (RAW/'table-records.json').exists() else []
 (RAW/'records.json').write_text(json.dumps(rr,ensure_ascii=False))
 with gzip.GzipFile(filename=str(RAW/'records.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(rr,ensure_ascii=False).encode())
 (RAW/'footnotes.json').write_text(json.dumps(ff,ensure_ascii=False))
 print(len(rr),collections.Counter(r['section'] for r in rr))
 for s in dict.fromkeys(r['section'] for r in rr):
  ns=[r['number'] for r in rr if r['section']==s];missing=set(range(1,max(ns)+1))-set(ns)
  print(s,'missing:', sorted(missing),'duplicates:',[n for n,c in collections.Counter(ns).items() if c>1])
 for r in rr[:4]:
  print(r['key']);print(''.join('<'+t['style']+'>'+t['text']+'</'+t['style']+'>' for t in r['tokens'])[:1800])
