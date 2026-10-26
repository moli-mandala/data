"""Inventory every nonblank IPA93 span before semantic grouping/alignment review.
This is an acquisition census, never a lexical installation or accepted unit count.
"""
import hashlib,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
out=[];census=[]
for n in range(77,125):
 p=P/'chapter4-native'/f'p{n:03}.json';page=json.loads(p.read_text());keys=[]
 for j,line in enumerate(page['lines']):
  for k,s in enumerate(line['spans']):
   if 'SILDoulosIPA93' not in s['font'] or not s['decoded'].strip():continue
   key=f'dhakal2011:p{n}:ch4:l{line["line"]}:span{k+1}'
   out.append({'source_unit_key':key,'printed_page':n,'pdf_page':n+22,'native_line':line['line'],'span':k+1,'raw':s,'line_evidence':line,'context_before':page['lines'][max(0,j-3):j],'context_after':page['lines'][j+1:j+9],'unresolved_private_glyphs':re.findall(r'⟦U\+[0-9A-F]+⟧',s['decoded']),'status':'unreviewed-native-span; not yet a lexical unit or aligned gloss'})
   keys.append(key)
 census.append({'printed_page':n,'pdf_page':n+22,'native_lines':len(page['lines']),'ipa93_nonblank_span_count':len(keys),'candidate_keys':keys,'native_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'whole-page native evidence preserved; semantic grouping, controls and quoted/Times-font morphology require review'})
(P/'chapter4-native-candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
(P/'chapter4-acquisition-census.json').write_text(json.dumps(census,ensure_ascii=False,indent=2)+'\n')
print(len(out),'native IPA93 spans across48pages; NOT accepted lexical count')
