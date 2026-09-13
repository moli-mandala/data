import sys,collections,json,unicodedata,re
import os
import pymupdf as f
from pathlib import Path
R=Path(__file__).parent;d=f.open(os.environ['KEED_PDF'])
mapping={};equations=[]
for page in (18,19,20):
 rows=collections.defaultdict(list)
 for s in d[page].get_texttrace():
  if 'AA_KAN' not in s['font']:continue
  for c in s['chars']:
   if c[0]!=65533:mapping[c[1]]=chr(c[0])
  rows[round(s['chars'][0][2][1],1)].append(s)
 for y,spans in rows.items():
  if len(spans)!=16:continue
  spans.sort(key=lambda s:s['chars'][0][2][0])
  base=chr(spans[0]['chars'][0][0]);assert 0xc80<ord(base)<0xd00
  for s,ending in zip(spans,['','ಾ','ಿ','ೀ','ು','ೂ','ೃ','ೄ','ೆ','ೇ','ೈ','ೊ','ೋ','ೌ','ಂ','ಃ']):
   equations.append(([c[1] for c in s['chars']],unicodedata.normalize('NFD',base+ending)))
for _ in range(5):
 for ids,target in equations:
  unknown=[i for i,x in enumerate(ids) if x not in mapping]
  if len(unknown)!=1:continue
  i=unknown[0];before=''.join(unicodedata.normalize('NFD',mapping[x]) for x in ids[:i]);after=''.join(unicodedata.normalize('NFD',mapping[x]) for x in ids[i+1:])
  if target.startswith(before) and target.endswith(after):
   value=target[len(before):len(target)-len(after) if after else None]
   mapping[ids[i]]=value
for cid in range(65,100):
 if cid in mapping:
  mapping[cid+430]=mapping[cid]+'್'
  mapping[cid+469]='್'+mapping[cid]
mapping[561]='್ೞ';mapping[589]='್ಱ';mapping[591]='ೞ್';mapping[592]='ೞ್';mapping[593]='{REPHA}'
mapping.update({**{i:chr(i+29) for i in range(1,49)},131:'—',135:'“',136:'”',15:',',16:'-',18:'/',611:'್ತ್ರ',606:'್ಟ್ರ',590:'್ಕೃ',581:'ಶ್ರಿ',602:'್ಛ್ರ',596:'್ಕ್ರ',616:'್ಪ್ರ',598:'್ಗ್ರ',608:'್ಡ್ರ',619:'್ಭ್ರ',587:'್ತ್ಯ',588:'್ತೃ'})
def decode(codes):
 t=''.join(mapping.get(cid,'{G'+str(cid)+'}') for cid in codes)
 # The font writes the vowel before the subscript consonant; Unicode stores it after.
 t=re.sub(r'([\u0cbe-\u0ccc\u0cd5\u0cd6]+)((?:್[ಕ-ಹಳಱೞ])+)',r'\2\1',t)
 t=re.sub(r'([ಕ-ಹಳಱೞ](?:್[ಕ-ಹಳಱೞ])*[\u0cbe-\u0ccc\u0cd5\u0cd6]*್?)\{REPHA\}',r'ರ್\1',t)
 return unicodedata.normalize('NFC',t.replace('್್','್'))

