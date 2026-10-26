"""Geometric candidates only: review required before any lexical emission."""
from pathlib import Path
import json,sys,re,unicodedata
import fitz
from snapshot_chapters12 import decode
P=Path(__file__).resolve().parent
pdf=fitz.open(P.parents[5]/'tmp/pdfs/darai-survey/dhakal-2011.pdf')
rawpages={}
for n in range(77,125):
 rawpages[n]=[l for b in pdf[n+21].get_text('rawdict')['blocks'] for l in b.get('lines',[])]
def words(li,ipa):
 is_form=any('IPA93' in s['font'] and any(c['c'].isalpha() or 0xe000<=ord(c['c'])<=0xf8ff for c in s['chars']) for s in li['spans'])
 if is_form != ipa:return []
 chars=[(c,s['font']) for s in li['spans'] for c in s['chars']]
 out=[];current=[]
 for c,font in chars:
  if c['c'].isspace():
   if current:out.append(current);current=[]
  else:current.append((c,font))
 if current:out.append(current)
 return [dict(raw=''.join(c['c'] for c,f in w),text=unicodedata.normalize('NFC',''.join(decode(c['c'],f) for c,f in w)),bbox=[min(c['bbox'][0] for c,f in w),min(c['bbox'][1] for c,f in w),max(c['bbox'][2] for c,f in w),max(c['bbox'][3] for c,f in w)]) for w in out]
rows=[]
for line in (P/'chapter4-expressions-first-reading.jsonl').read_text().splitlines():
 expr=json.loads(line);p=expr['printed_page'];native=[];eng=[]
 for li in expr['interlinear_evidence']:
  # Cross-page continuation consists only of headers before translation; no lexical line.
  if li['line']>len(rawpages[p]):continue
  raw=rawpages[p][li['line']-1]
  for i,w in enumerate(words(raw,True),1):
   w.update(source_unit_key=expr['source_unit_key']+f":word:l{li['line']}:w{i}",native_line=li['line'],gloss_groups=[]);native.append(w)
  e=words(raw,False)
  # Normal spaces within a multiword gloss are retained; large column gaps split.
  groups=[]
  for w in e:
   if groups and w['bbox'][0]-groups[-1][-1]['bbox'][2]<6:groups[-1].append(w)
   else:groups.append([w])
  for g in groups:eng.append(dict(text=' '.join(w['text'] for w in g),bbox=[g[0]['bbox'][0],min(w['bbox'][1] for w in g),g[-1]['bbox'][2],max(w['bbox'][3] for w in g)],line=li['line']))
 split_eng=[]
 for g in eng:
  candidates=[w for w in native if 4<g['bbox'][1]-w['bbox'][1]<42]
  if candidates:
   y=max(w['bbox'][1] for w in candidates);near=[w for w in candidates if abs(w['bbox'][1]-y)<2]
   # Reconstruct English word boxes from the original line for exact column starts.
   raw=rawpages[p][g['line']-1]
   ew=[w for w in words(raw,False) if g['bbox'][0]-.5<=w['bbox'][0] and w['bbox'][2]<=g['bbox'][2]+.5]
   parts=[]
   for ewitem in ew:
    anchor=min(near,key=lambda w:abs(w['bbox'][0]-ewitem['bbox'][0]))
    exact=abs(anchor['bbox'][0]-ewitem['bbox'][0])<2.5
    if not parts or (exact and parts[-1][0]['source_unit_key']!=anchor['source_unit_key']):parts.append((anchor,[ewitem]))
    else:parts[-1][1].append(ewitem)
   for best,part in parts:
    result=dict(text=' '.join(w['text'] for w in part),bbox=[part[0]['bbox'][0],g['bbox'][1],part[-1]['bbox'][2],g['bbox'][3]],line=g['line'],x_difference=round(abs(best['bbox'][0]-part[0]['bbox'][0]),2))
    best['gloss_groups'].append(result)
 for w in native:
  rows.append(dict(source_unit_key=w['source_unit_key'],parent_expression_key=expr['source_unit_key'],printed_page=p,pdf_page=p+22,native_line=w['native_line'],form=w['text'],raw_form=w['raw'],form_bbox=w['bbox'],candidate_gloss=' '.join(g['text'] for g in w['gloss_groups']),gloss_evidence=w['gloss_groups'],status='unreviewed geometric alignment candidate',source_example_status=expr['status'],needs_manual_alignment=not w['gloss_groups'] or any(g['x_difference']>7 for g in w['gloss_groups'])))
(P/'chapter4-interlinear-alignment-candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
print(len(rows),'word candidates;',sum(r['needs_manual_alignment'] for r in rows),'explicit geometric warnings; all candidates require review')
