"""Font-scoped visual-glyph recovery; first reading, not installation approval."""
from pathlib import Path
import csv,hashlib,json,re,unicodedata

HERE=Path(__file__).resolve().parent
# These are CID fallback values, NOT Sundanese/Ol Chiki/Georgian text.
# The PDF's ToUnicode maps them to FFFD. Exact original-font glyph evidence
# is retained separately; only this document's ArialUnicodeMS is decoded.
GLYPHS={
 'ᮢ':'त्र','ᮤ':'द्र','ᱠ':'हू','ᱫ':'त्त','ᱭ':'ीं','ᱶ':'ें',
 'ᱹ':'ैं','ᲂ':'ों','Წ':'च्च','Ჯ':'ज्ज','Ჵ':'ट्ट','᲻':'ड्ड',
 '᳎':'द्द','᳖':'न्न','ᳰ':'ि','ᳱ':'ी','ᳲ':'िं',
}

def decode_span(text,font):
 if font!='ArialUnicodeMS': return text
 text=''.join(GLYPHS.get(ch,ch) for ch in text)
 # This font stream places the short-i glyph before its consonant cluster.
 # Other source fonts use logical Unicode ordering and are left alone.
 text=re.sub(r'ि(ं?)([क-हक़-य़](?:्\s*[क-हक़-य़])*)',lambda m:m[2]+'ि'+m[1],text)
 return text

def decode_group(group):
 text=''.join(decode_span(s['text'],s['font']) for s in group['spans'])
 text=re.sub(r'्\s+(?=[क-हक़-य़])','्',text)
 return unicodedata.normalize('NFC',' '.join(text.split()))

def build():
 units=json.loads((HERE/'narrative-geometry-first-extraction.json').read_text())
 tables=json.loads((HERE/'layout-tables-first-extraction.json').read_text())
 free={}
 for page in tables:
  if page['pdf_page']<70:continue
  for t in page['tables']:
   first=next((s for s in t['rows'][0] if s),'')
   match=re.fullmatch(r'(\d+)\.\s*Ku',first)
   if not match:continue
   text=' '.join(s for s in t['rows'][-1] if s)
   assert 'Free' in text
   free[int(match[1])]=' '.join(re.sub(r'\bFree\b','',text).split())
 assert set(free)==set(range(1,36))
 output=[]
 for unit in units:
  words=[];bands=unit['bands'];word=0
  for n in range(0,len(bands),2):
   native,english=bands[n:n+2]
   assert 'Ku' in native['label'] and english['label']=='Eng'
   assert len(native['groups'])==len(english['groups'])
   for form,gloss in zip(native['groups'],english['groups']):
    word+=1
    row={'source_unit_key':unit['source_unit_key']+f':word{word}',
     'printed_page':unit['printed_page'],'pdf_page':unit['pdf_page'],
     'segment':unit['segment'],'physical_band':n//2+1,'word':word,
     'form':decode_group(form),'native':decode_group(form),'phonemic':'',
     'gloss':gloss['raw_text'],'source_native_raw':form['raw_text'],
     'native_bbox':form['bbox'],'english_bbox':gloss['bbox'],
     'lect':'Lohardaga, Jharkhand, India; recorded2018',
     'status':'first reading; full original comparison pending',
     'note':'Printed aligned group; no unprinted lemma or phonemic form inferred.'}
    if unit['segment']==28 and word==1:
     row['note']+=' Two adjacent nasal dots are visibly printed: one in CID1CF2 and one independent0902. Both retained; typographic interpretation uncertain.'
     row['uncertainty']='source-typography:double-nasal-mark'
     assert row['form']=='पिंंटकाय',repr(row['form'])
    words.append(row);output.append(row)
  output.append({'source_unit_key':unit['source_unit_key']+':whole',
   'printed_page':unit['printed_page'],'pdf_page':unit['pdf_page'],'segment':unit['segment'],
   'form':' '.join(r['form'] for r in words),'native':' '.join(r['native'] for r in words),
   'phonemic':'','gloss':free[unit['segment']],
   'aligned_unit_keys':[r['source_unit_key'] for r in words],
   'lect':'Lohardaga, Jharkhand, India; recorded2018',
   'status':'first reading; full original comparison pending',
   'note':'Complete source-numbered segment with its printed free translation; no cross-segment normalization.'})
 assert len(output)==412
 (HERE/'narrative-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
 with (HERE/'narrative-first-reading.tsv').open('w') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['source_unit_key','form','gloss','note'])
  for r in output:w.writerow([r['source_unit_key'],r['form'],r['gloss'],r['note']])
 manifest={'status':'font-scoped first-reading mapping; full narrative review pending',
  'source_font':'NZSUWS+ArialUnicodeMS, PDFfontxref865, CIDToGID Identity; ToUnicode375',
  'mappings':{f'U+{ord(k):04X}':v for k,v in GLYPHS.items()},
  'short_i_policy':'Visual preposed glyph moved after following consonant cluster only within this source font.',
  'normalization':'NFC preserves both U0902 marks; no nasal collapse and no phonological transliteration.',
  'counts':{'aligned_groups':377,'whole_segments':35,'total':412},
  'evidence':['native-glyph-original-chart.png','native-glyph-fire-original.png','native-isolated-i-glyphs.png','native-segment28-double-nasal-original.png','native-segment28-font-trace.json','root-native-glyph-review-20260927.json']}
 (HERE/'native-decoding-first-review.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(manifest['counts'])

if __name__=='__main__':build()
