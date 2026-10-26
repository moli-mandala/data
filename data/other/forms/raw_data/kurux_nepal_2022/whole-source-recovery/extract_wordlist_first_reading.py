"""Full six-column census; legacy extractor is a draft, never an authority."""
from pathlib import Path
import csv, importlib.util, json, re
import pdfplumber

HERE=Path(__file__).resolve().parent
old_path=HERE.parents[1]/'kurux_nepal.py'
spec=importlib.util.spec_from_file_location('kurux_legacy',old_path)
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
LECTS=old.LECTS+((604.,689.,'kurux_pathibari'),(689.,780.,'kurux_ranchi'))
records=[]
with pdfplumber.open(old.PDF) as pdf:
 for page_index in range(51,64):
  page=pdf.pages[page_index]
  words=page.extract_words(x_tolerance=1,y_tolerance=2)
  heads=sorted([(int(w['text']),w['top']) for w in words if 70<=w['x0']<100 and re.fullmatch(r'\d{1,3}',w['text']) and 1<=int(w['text'])<=338],key=lambda x:x[1])
  for position,(number,top) in enumerate(heads):
   bottom=heads[position+1][1] if position+1<len(heads) else 570.
   gloss=old._cell(words,105,193,top,bottom)
   nepali=old._cell(words,193,264,top,bottom)
   for left,right,lect in LECTS:
    selected=[w for w in words if left<=w['x0']<right and top-4<=w['top']<bottom-4]
    raw_lines=[' '.join(w['text'] for w in line) for line in old._lines(selected)]
    draft=old.GLYPH_FIXES.get((number,lect),old._cell(words,left,right,top,bottom))
    records.append({'source_unit_key':f'kurux-nepal:{number}:{lect}', 'printed_page':page_index-7,'pdf_page':page_index+1,'prompt':number,'lect':lect,'gloss':gloss,'nepali_control':nepali,'raw_lines':raw_lines,'draft':draft,'source_bbox':[left,top-4,right,bottom-4],'status':'source-blank' if draft in ('','---') else 'first-reading; original review pending','draft_method':'historical extractor plus its keyed fixes; native glyph ordering not yet certified','raw_words':selected})
assert len(records)==338*6, len(records)
assert len({r['source_unit_key'] for r in records})==len(records)
(HERE/'wordlist-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
with (HERE/'wordlist-first-reading.tsv').open('w') as f:
 writer=csv.writer(f,delimiter='\t');writer.writerow(['source_unit_key','printed_page','gloss','form','status'])
 for r in records: writer.writerow([r['source_unit_key'],r['printed_page'],r['gloss'],r['draft'],r['status']])
for _,_,lect in LECTS:
 rows=[r for r in records if r['lect']==lect]
 print(lect,len(rows),sum(r['status']=='source-blank' for r in rows),'blanks',sum(len(r['draft'].split(',')) for r in rows if r['status']!='source-blank'),'draft alternatives')
