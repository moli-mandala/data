"""Inventory all Appendix C tagged blocks; no accepted lexical extraction."""
import hashlib,json,re
from pathlib import Path
import fitz
W=Path('/Users/aryamanarora/Documents/Code/jambu-all')
HERE=Path(__file__).resolve().parent
PDF=W/'tmp/pdfs/korwa-kodaku-2022/source.pdf'
EXPECTED='a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c'
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(PDF.read_bytes())==EXPECTED
doc=fitz.open(PDF);assert len(doc)==115
blocks=[];current=None;section=None;number=None;question=None
pages=[]
def flush():
 global current
 if current is not None:blocks.append(current);current=None
for pdf_page in range(93,109):
 page=doc[pdf_page-1]
 text=page.get_text('text',sort=True)
 lines=text.splitlines()
 pages.append({'pdf_page':pdf_page,'printed_page':pdf_page-10,'native_text_sha256':sha(text.encode()),
               'lines':lines,'positioned_blocks':page.get_text('blocks')})
 for lineno,line in enumerate(lines,1):
  stripped=line.strip()
  # Retain context separately; a physical page header is not a gloss continuation.
  if lineno<=3 and stripped==str(pdf_page-10):continue
  sec=re.match(r'^C\.([2-5])(?:\s|$)',stripped)
  if sec:
   flush();section='C.'+sec.group(1);number=None;question=None
  match=re.match(r'^\\([npgɡfqa])(?:\s+(.*))?$',stripped)
  # Printed C.4 unit20 omits the backslash on its gloss marker. Restrict
  # recovery to this reviewed source location, never arbitrary target g.
  bare_gloss = (pdf_page==102 and section=='C.4' and number=='20'
                and re.match(r'^g\s+and that time good beating also given$',stripped))
  if bare_gloss:
   match=re.match(r'^([g])\s+(.*)$',stripped)
  if match and section:
   flush()
   if match.group(1)=='n':
    number=(match.group(2) or '').strip();question=None
   current={'physical_key':f'behera2022korwakodaku:pdf{pdf_page}:line{lineno}',
    'section':section,'marker':'g' if match.group(1)=='ɡ' else match.group(1),
    'raw_marker':match.group(1),'pdf_page':pdf_page,'printed_page':pdf_page-10,
    'printed_number':number,'question_number':question,
    'source_lines':[{'pdf_page':pdf_page,'line':lineno,'raw':line}],
    'accepted_transcription':False}
   if bare_gloss:
    current['marker_recovery']='Printed g lacks backslash; source-specific original review'
  elif re.match(r'^QUESTION\s+\d+',stripped):
   flush();question=re.match(r'^QUESTION\s+(\d+)',stripped).group(1)
   current={'physical_key':f'behera2022korwakodaku:pdf{pdf_page}:line{lineno}',
    'section':section,'marker':'question-label','raw_marker':'QUESTION',
    'pdf_page':pdf_page,'printed_page':pdf_page-10,
    'printed_number':number,'question_number':question,
    'source_lines':[{'pdf_page':pdf_page,'line':lineno,'raw':line}],
    'accepted_transcription':False}
  elif sec:
   pass
  elif current is not None:
   if stripped:current['source_lines'].append({'pdf_page':pdf_page,'line':lineno,'raw':line})
flush()
# Exact raw blocks are retained even where a page wraps or marker alignment needs review.
for b in blocks:
 b['raw_text']='\n'.join(x['raw'] for x in b['source_lines'])
 b['review_state']='Unaccepted native-text scaffold; original layout and marker scope require review'
HERE.mkdir(parents=True,exist_ok=True)
assert blocks and {b['section'] for b in blocks}=={'C.2','C.3','C.4','C.5'}
for name,value in [('native-pages.json',pages),('tagged-blocks-v5.json',blocks)]:
 content=json.dumps(value,ensure_ascii=False,indent=2)+'\n';target=HERE/name
 assert not target.exists() or target.read_text()==content
 target.write_text(content)
counts={}
for b in blocks:counts.setdefault(b['section'],{}).setdefault(b['marker'],0);counts[b['section']][b['marker']]+=1
report={'status':'native tagged-block inventory only; not accepted emission', 'source_sha256':EXPECTED,
 'pdf_pages':[93,108],'blocks':len(blocks),'section_marker_counts':counts,
 'pending':['Original marker boundaries and wrapped lines','Whole translations spanning multiple phonetic blocks','Word/gloss alignment','Question/answer language attribution','Final identity and source-specific transcription profile']}
(HERE/'snapshot-status.json').write_text(json.dumps(report,indent=2)+'\n')
images=W/'tmp/behera-interlinear';images.mkdir(exist_ok=True)
for p in (93,102):
 image_path=images/f'original-pdf{p:03d}-150dpi.png'
 if not image_path.exists():doc[p-1].get_pixmap(dpi=150).save(image_path)
print(json.dumps(report,indent=2))
