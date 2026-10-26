"""Reproducible typography census, not an approved linguistic parser."""
import fitz,json,pathlib,collections
P=pathlib.Path(__file__).resolve().parent
D=fitz.open(P/'upstream/Kholosi_Dictionary.pdf')
units=[];layout=[];current=None
for pi in range(2,len(D)):
 page=D[pi];spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if 70<s['origin'][1]<775]
 for col in range(2):
  ss=sorted([s for s in spans if (s['bbox'][0]>=300)==bool(col)],key=lambda s:(round(s['origin'][1],1),s['bbox'][0]))
  lines=[]
  for s in ss:
   yy=s['origin'][1]
   if s['size']<8 and s['text'].strip().isdigit():
    nearby=[t['origin'][1] for t in ss if t['size']>8 and 0<t['origin'][1]-yy<6]
    if nearby: yy=min(nearby)
   row=next((r for r in reversed(lines[-3:]) if abs(r['baseline']-yy)<2),None)
   if row is None:row={'baseline':yy,'spans':[]};lines.append(row)
   row['spans'].append(s)
  for row in lines:
   row['spans'].sort(key=lambda s:s['bbox'][0]); ss=row['spans'];text=''
   last=None
   for z in ss:
    if last is not None and z['bbox'][0]-last['bbox'][2]>1 and not text.endswith(' ') and not z['text'].startswith(' '):text+=' '
    text+=z['text'];last=z
   record=dict(pdf_page=pi+1,printed_page=pi-1,column=col+1,baseline=round(row['baseline'],3),raw_text=text,spans=[{k:s[k] for k in ('text','font','size','bbox')} for s in ss]);layout.append(record)
   if any(s['size']>12 for s in ss):continue
   lead=ss[0];is_sub=text.startswith('▷');is_head='Bold' in lead['font'] and lead['size']>9.5 and lead['bbox'][0]<(95 if col==0 else 323)
   if is_head or is_sub:
    kind='head' if is_head else 'subentry';index=sum(x['printed_page']==pi-1 and x['column']==col+1 for x in units)+1
    current=dict(source_unit_key=f'kholosi:pdf:p{pi-1}:c{col+1}:u{index}',kind=kind,printed_page=pi-1,column=col+1,pdf_page=pi+1,lines=[])
    units.append(current)
   if current is None:raise ValueError(record)
   current['lines'].append(record)
for u in units:u['raw_text']='\n'.join(x['raw_text'] for x in u['lines'])
(P/'pdf-unit-census.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in units))
(P/'pdf-layout-census.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in layout))
print(collections.Counter(x['kind'] for x in units))
for p in range(1,11):print(p,collections.Counter(x['kind'] for x in units if x['printed_page']==p))
for u in units:
 if u['kind']=='subentry':print(u['source_unit_key'],u['raw_text'])
