"""Complete surface-example inventory, first reading; tables/prose separate.
Keeps full interlinear evidence; no guessed lemma generation or installation.
"""
import csv,json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
pages={n:json.loads((P/'chapter4-native'/f'p{n:03}.json').read_text()) for n in range(77,125)}
# Printed surface line(s), English translation line(s), and optional explicit source status.
M={
78:[(7,13),(16,40),(43,61),(63,80),(83,90),(93,113)],
79:[(19,30),(33,51),(54,66),(68,81)],80:[(5,15),(23,32),(34,46)],81:[(7,19),(21,42)],82:[(87,98)],83:[(5,18)],
84:[(11,19),(21,30),(36,51),(53,68),(71,82)],
86:[(5,15),(18,30),(33,43),(45,56),(65,77),(79,91)],87:[(5,17),(104,113),(115,124)],
88:[(13,23),(25,33),(41,51),(53,62)],89:[(9,[28,29]),(31,[45,46]),(48,61)],90:[(74,83)],
91:[(4,11),(14,25),(28,41),(44,[53,54]),(62,[79,80])],92:[(4,[20,21]),(63,73)],
93:[(5,14),(16,24),(26,[39,40]),(42,49),(51,60,'source-unacceptable')],
94:[(4,6),(7,9),(10,11),(12,14),(15,17),(18,[19,20]),(21,23),(24,[26,27]),(28,30),(31,33),(34,36),(37,39),(40,42),(45,56),(59,68),(70,82),(84,98)],
95:[(12,22),(24,33),(36,47),(49,60)],96:[(29,39)],97:[(4,22),(30,42),(44,58)],
98:[(11,22),(24,40),(42,55),(57,68),(70,85)],99:[(14,28),(31,45),(47,61),(63,75),(77,89)],100:[(15,25),(27,41)],
101:[(5,19),(22,33),(35,49),(52,65),(68,84)],
102:[(5,16),(19,38),(40,51),(59,74),(76,88),(90,102)],103:[(4,20),(34,52),(54,71)],
104:[(5,[16,17]),(19,31),(37,[48,49]),(54,67)],105:[(5,[25,26]),(28,42),(46,[64,65]),(73,88)],
106:[(4,12),(14,27),(29,43),(55,65)],107:[(5,12),(14,25),(27,36),(38,47),(60,68)],
108:[(5,21),(28,40),(42,50)],109:[(6,19),(21,36),(45,56),(58,69)],
110:[(6,22),(28,[43,44]),(46,[60,61]),(66,78),(85,('cross',111,3))],
111:[(5,18),(25,39),(41,55),(57,78),(80,93)],
113:[(9,19),(22,33)],114:[(5,19),(21,33),(35,46),(55,75),(78,92),(94,100)],
115:[(56,67),(69,80),(83,94)],116:[(4,17),(44,57),(60,76),(78,92)],
117:[(37,48),(50,63),(66,77),(79,87)],118:[(9,25),(27,43),(53,61),(63,75)],
119:[(15,22),(27,33),(37,47),(55,70)],120:[(4,[17,18]),(21,[32,33]),(40,49),(52,58),(61,71)],
121:[(7,15),(48,65),(67,75)],122:[(4,14),(21,35),(37,50),(52,62),(72,82)],
123:[(5,17),(23,33),(35,[53,54]),(65,76),(79,87)]}
def line(n,l):return ''.join(s['decoded'] for s in pages[n]['lines'][l-1]['spans']).strip()
def dec(s):
 # Root confirmed actual embedded acute/x glyphs. Nasal attachment remains a
 # per-record image-review gate; raw sequences are never discarded.
 s=s.replace('⟦U+F040⟧','\u0301').replace('⟦U+F07E⟧','\u033d')
 return unicodedata.normalize('NFC',re.sub(r'\s+',' ',s).strip())
units=[]
for n,entries in M.items():
 for entry in entries:
  fl,gl=entry[:2];status=entry[2] if len(entry)>2 else 'first-reading'
  source_form=line(n,fl);form=dec(source_form)
  if isinstance(gl,tuple):gn,ls=gl[1],[gl[2]]
  else:gn,ls=n,gl if isinstance(gl,list) else [gl]
  quote=' '.join(line(gn,v) for v in ls)
  ids=re.findall(r'\(([A-Za-z][A-Za-z.]+\.\d+)\)',quote)
  gloss=re.sub(r'\s*\([A-Za-z][A-Za-z.]+\.\d+\)','',quote).strip().strip('‘’\'" ')
  analysis=pages[n]['lines'][fl:min(ls)-1] if gn==n else pages[n]['lines'][fl:]+pages[gn]['lines'][:min(ls)-1]
  record={'source_unit_key':f'dhakal2011:p{n}:ch4:surface:l{fl}','printed_page':n,'pdf_page':n+22,'surface_lines':[fl],'translation_page':gn,'translation_lines':ls,'form':form,'source_form_before_scoped_glyph_resolution':source_form,'gloss':gloss,'raw_translation':quote,'text_ids':ids,'interlinear_evidence':analysis,'source_surface_evidence':pages[n]['lines'][fl-1],'source_translation_evidence':[pages[gn]['lines'][v-1] for v in ls],'status':status,'unresolved_glyphs':re.findall(r'⟦U\+[0-9A-F]+⟧',form),'review':'Full native text read; original-image second reading and scoped grammar tagging pending.'}
  if n==78 and fl==43:record['uncertainty']='Literal acute plus x-above from isolated embedded original glyphs; intended phonological analysis unresolved. No nasal substitution.'
  units.append(record)
(P/'chapter4-expressions-first-reading.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in units))
print(len(units),'whole expressions preserved; no canonical installation; tables/prose/interlinear token accounting pending')
