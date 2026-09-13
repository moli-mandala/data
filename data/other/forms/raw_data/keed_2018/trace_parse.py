import gzip,json,re,collections,unicodedata
from pathlib import Path
import os
R=Path(os.environ["KEED_CACHE"])
import keed_glyphmap
def text(s):
 if 'AA_KAN' in s['font']:return keed_glyphmap.decode([c[1] for c in s['chars']])
 return ''.join(chr(c[0]) if c[0]!=65533 else '{G'+str(c[1])+'}' for c in s['chars']).replace('\ue835','r̥').replace('\ue847','r̥̄').replace('\ue898','ɐ̃').replace('\ue8c7','r̥̄')
def spans_to_text(ss, markup=False):
 if markup:
  ss=[s for s in ss if not ('AA_KAN' in s['font'] and 9.9<s['size']<10.1)]
 # Cluster baseline offsets used by reference italics, brackets and register symbols.
 ys=sorted(set(round(s['chars'][0][3],1) for s in ss if s['chars']))
 groups=[]
 for y in ys:
  if groups and y-groups[-1][0]<4.5:groups[-1].append(y)
  else:groups.append([y])
 baselines=[sum(g)/len(g) for g in groups]
 lines=collections.defaultdict(list)
 for s in ss:
  if s['chars']:lines[min(baselines,key=lambda b:abs(b-s['chars'][0][3]))].append(s)
 result=[]
 for y,parts in sorted(lines.items()):
  parts.sort(key=lambda s:s['chars'][0][2]);last=None;line=''
  for s in parts:
   gap=s['bbox'][0]-(last or s['bbox'][0])
   if gap>1:line+=' '
   t=text(s).translate(str.maketrans({'ﬀ':'ff','ﬁ':'fi','ﬂ':'fl','ﬃ':'ffi','ﬄ':'ffl'}))
   if markup and s['font']=='GandhariUnicode' and abs(s['size']-7.5318)<.03:t='{SM}'+t+'{/SM}'
   if markup and s['font']=='GandhariUnicode' and abs(s['size']-7.9701)<.03:t='{R}'+t+'{/R}'
   if markup and s['font']=='GandhariUnicode-Bold' and abs(s['size']-8.3686)<.03 and t.isdigit():t='{S:'+t+'}'
   line+=t;last=s['bbox'][2]
  result.append(line)
 return '\n'.join(result)

out=[];glyphs=collections.Counter()
with gzip.open(R/'keed-traces.jsonl.gz','rt') as stream:
 for l in stream:
  p=json.loads(l);ss=[s for s in p['spans'] if s['chars'] and s['chars'][0][3]>55]
  for col in (0,1):
   spans=[s for s in ss if (s['chars'][0][2]>=260)==bool(col)]
   heads=sorted([s for s in spans if 'AA_KAN' in s['font'] and s['size']>12.5],key=lambda s:s['chars'][0][3])
   hhs=[]
   for s in heads:
    if hhs and abs(hhs[-1]['chars'][0][3]-s['chars'][0][3])<1:
     hhs[-1]['chars']+=s['chars']
    else:hhs.append(dict(s,chars=list(s['chars'])))
   prefix=[s for s in spans if s['chars'][0][3]<(hhs[0]['chars'][0][3]-5 if hhs else 1000)]
   if prefix and out:
    out[-1]['raw']+='\n'+spans_to_text(prefix)
    out[-1]['layout']+=f'\n⟦LOC {p["page"]-29} {col+1}⟧\n'+spans_to_text(prefix,True)
   for i,h in enumerate(hhs):
    y=h['chars'][0][3];end=hhs[i+1]['chars'][0][3]-5 if i+1<len(hhs) else 1000
    selected=[s for s in spans if y-5<=s['chars'][0][3]<end]
    raw=spans_to_text(selected)
    key=f'keed2018:p{p["page"]-29}:c{col+1}:e{i+1}'
    # Roman head precedes the first IPA bracket and uses a distinct font size.
    rs=[s for s in selected if s['font']=='GandhariUnicode' and abs(s['size']-7.9701)<.03]
    roman=spans_to_text(rs).replace('\n','').strip()
    firstipa=next((s for s in selected if s['font']=='GandhariUnicode' and abs(s['size']-8.3686)<.03 and '[' in text(s)),None)
    if firstipa:
     iy=firstipa['chars'][0][3]
     roman=spans_to_text([s for s in rs if s['chars'][0][3]<iy-1 or abs(s['chars'][0][3]-iy)<1 and s['bbox'][0]<firstipa['bbox'][0]]).replace('\n','').strip()
    out.append(dict(key=key,page=p['page']-29,column=col+1,ordinal=i+1,roman=roman,native=keed_glyphmap.decode([c[1] for c in h['chars']]),native_codes=[c[:2] for c in h['chars']],raw=raw,layout=spans_to_text(selected,True)))
with (R/'trace-records.jsonl').open('w') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(len(out))

