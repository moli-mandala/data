"""Position-aware decoding of Zoller's 2023 TeX/Type3 source PDF (no OCR)."""
import argparse, collections, gzip, hashlib, json, re, unicodedata
from pathlib import Path
import pymupdf
TIPA={**{i:chr(i) for i in range(32,127)},0:'̀',1:'́',2:'̂',3:'̃',4:'̈',6:'̊',7:'̌',8:'̆',9:'̄',10:'̇',12:'̜',16:'̑',17:'̪',21:'̹',25:'ı',26:'ȷ',28:'̞',34:'ˈ',38:'~',48:'ʉ',49:'ɨ',50:'ʌ',51:'ɜ',53:'ɐ',54:'ɒ',55:'ɤ',57:'ɘ',58:'ː',59:'ˑ',60:'͜',62:'͡',64:'ə',65:'ɑ',66:'β',67:'ɕ',68:'ð',69:'ɛ',70:'ɸ',71:'ɣ',72:'ɦ',73:'ɪ',75:'ʁ',78:'ŋ',79:'ɔ',80:'ʔ',81:'ʕ',82:'ɾ',83:'ʃ',84:'θ',85:'ʊ',86:'ʋ',87:'ɯ',88:'χ',90:'ʒ',94:'̚',96:'‘',126:'̰',127:'̩',161:'đ',162:'ɗ',163:'ɗ',170:'ɬ',171:'λ',176:'ω',181:'ʦ',184:'ʓ',187:'ƀ',188:'ʔ',195:'ʣ',200:'ɤ',204:'ɩ',205:'ʄ',217:'ʧ',225:'ɓ',226:'ɗ',227:'ɖ',228:'ɠ',230:'æ',231:'ç',232:'ħ',233:'ʄ',234:'ƒ',235:'ɬ',236:'ɭ',237:'ɭ',238:'ɰ',239:'ɳ',241:'ɲ',243:'ɽ',244:'ɹ',248:'ø',249:'ʂ',250:'ʈ',253:'ʑ'}
EXTRA={6:'̴',32:'a̢',33:'ɑ',42:'e̢',44:'γ',49:'ɿ',51:'ʄ',118:'ɪ'}
ACC={'¯':'̄','´':'́','`':'̀','˙':'̇','˜':'̃','ˇ':'̌','¨':'̈','˘':'̆','˚':'̊'}
RAISED=dict(zip('abdeghiklmnoprstuwxyzəɣ','ᵃᵇᵈᵉᵍʰⁱᵏˡᵐⁿᵒᵖʳˢᵗᵘʷˣʸᶻᵊˠ'))
RAISED['v']='ᵛ'

def decode_page(pg, x_range=None):
 chars=[]
 for s in pg.get_texttrace():
  if s['font']=='Arial':continue
  for u,g,xy,box in s['chars']:
   if x_range and not x_range[0]<=xy[0]<x_range[1]:continue
   c=(EXTRA if s['font'] in ('F237','F248') else TIPA).get(g,'�') if s['font'].startswith('F') else chr(u)
   c=ACC.get(c,c)
   if s['font'].startswith('NewTXMI'):c=unicodedata.normalize('NFKC',c)
   # TeX dot-below is a full stop lowered beneath the baseline.
   chars.append(dict(c=c,x=xy[0],y=xy[1],end=box[2],size=s['size'],font=s['font'],glyph=g,italic=s['font'].startswith('F') or (s['font'].startswith('NewTXMI') and c not in '<>←→=') or bool(re.search('SF(?:SL|TI)',s['font']))))
 # Baselines with multiple ordinary letters; stand-alone accent baselines attach geometrically.
 freq=collections.Counter(round(c['y'],1) for c in chars if c['c'].isalpha())
 baselines=[y for y,n in freq.items() if n>=3]
 for i,c in enumerate(chars):
  nearby=chars[max(0,i-4):i]+chars[i+1:i+5]
  if c['c']=='.':
   near=[b for b in nearby if b['c'] and b['c'][0].isalpha() and abs((b['x']+b['end'])/2-(c['x']+c['end'])/2)<2 and 0.7<c['y']-b['y']<4]
   if near:
    b=min(near,key=lambda b:abs(b['x']-c['x']));c['c']='̣';c['y']=b['y']
  if c['c'] and unicodedata.combining(c['c'][0]):
   near=[b for b in nearby if b['c'] and b['c'][0].isalpha() and -2<c['y']-b['y']<9 and abs((b['x']+b['end'])/2-(c['x']+c['end'])/2)<3]
   if near:
    b=min(near,key=lambda b:abs((b['x']+b['end'])/2-(c['x']+c['end'])/2)+abs(b['y']-c['y'])*.1)
    if c['y']-b['y']>4:c['c']={'̇':'̣','̈':'̤','̊':'̥','̃':'̰','̄':'̱','̆':'̮','̂':'̭','̌':'̬'}.get(c['c'],c['c'])
    b['c']+=c['c'];c['c']=''
 rows=collections.defaultdict(list)
 for c in chars:
  if not c['c']:continue
  y=min(baselines,key=lambda y:abs(y-c['y'])) if baselines else c['y']
  if abs(y-c['y'])>5:y=c['y']
  rows[round(y,1)].append(c)
 output=[]
 for y,cs in sorted(rows.items()):
  cs.sort(key=lambda c:c['x']);tokens=[];prev=None
  for c in cs:
   style='f' if c['italic'] else 'r'
   if c['size']<8 and 1<c['y']*-1+y<5 and (c['italic'] or c['font'].startswith('NewTXMI')):
    base=c['c'].replace('ℎ','h')
    if base in RAISED:c['c']=RAISED[base];style='f'
   # Preserve raised/lowered figures separately (footnotes and homonym indices).
   if c['c'].isdigit() and c['size']<8 and prev and abs(c['y']-y)>1:style='sup' if c['y']<y else 'sub'
   space=' ' if prev and c['x']-prev['end']>max(.9,c['size']*.12) else ''
   if tokens and tokens[-1]['style']==style:tokens[-1]['text']+=space+c['c']
   else:
    if tokens:tokens[-1]['text']+=space
    tokens.append({'style':style,'text':c['c']})
   if prev and c['glyph']==-1:c['end']=max(c['end'],prev['end'])
   prev=c
  for t in tokens:t['text']=unicodedata.normalize('NFC',t['text'].replace('ı','i').replace('ȷ','j'))
  output.append(dict(y=y,x=round(cs[0]['x'],2),size=round(max(c['size'] for c in cs),2),tokens=tokens,text=''.join(t['text'] for t in tokens)))
 return output

def main():
 a=argparse.ArgumentParser();a.add_argument('pdf',type=Path);a.add_argument('--output',type=Path,default=Path('/tmp/zoller-pages.jsonl.gz'));args=a.parse_args()
 p=pymupdf.open(args.pdf);assert len(p)==1115
 assert hashlib.sha256(args.pdf.read_bytes()).hexdigest()=='ac1b832e1a6730247a10ea53b0219f545ad8e9bef764f00850209baec9788b0a', 'Unexpected edition: re-audit page/font assumptions'
 with gzip.open(args.output,'wt',encoding='utf8') as out:
  for i in [*range(13,28),*range(547,1064)]:
   out.write(json.dumps({'pdf_page':i+1,'printed_page':i-28,'lines':decode_page(p[i])},ensure_ascii=False)+'\n')
 print(args.output,hashlib.sha256(args.pdf.read_bytes()).hexdigest())
if __name__=='__main__':main()
