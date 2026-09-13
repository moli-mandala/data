"""Pinned Census comparative lexicons, Regmi–Thakur Danuwar, and Tharu XLSX.

No network on import. --extract reconstructs coordinate tables from original PDFs.
--output stages a review proposal; --install installs only deterministic outputs.
Image transcriptions are reviewed source evidence, never guessed by the parser.
"""
import argparse, collections, csv, hashlib, io, json, random, re, shutil, sys, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RAW=Path(__file__).with_name('census_nepal_2026')
sys.path.insert(0,str(ROOT))
from dialects import dialect_tag
SOURCES={'tamil-nadu':'census2023tamilnadu','uttar-pradesh':'census2023uttarpradesh','bihar':'census2020bihar','sikkim2':'census2012sikkim2','danuwar':'regmi-thakur2016danuwar','tharu':'mitchell-eichentopf2013tharu'}
PROFILES={k:('census-ascii' if k in ('bihar','sikkim2') else 'census-danuwar' if k=='danuwar' else 'census-ipa') for k in SOURCES}
EXPECTED={'tamil-nadu':5988,'uttar-pradesh':5988,'bihar':3500,'sikkim2':1500,'danuwar':1050,'tharu':2360}
def nfc(s):return unicodedata.normalize('NFC',s)
def load(name):return json.loads((RAW/name).read_text())
def readcsv(path):return list(csv.reader(path.open()))
def writecsv(path,rows):
 with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
def fingerprint(s):
 # PDF places layout spaces before combining marks and sometimes inside words.
 return ''.join(nfc(s).split()).replace(':','ː')
def extract_workbook():
 """Read the pinned XLSX using OOXML; no spreadsheet runtime dependency."""
 import zipfile,xml.etree.ElementTree as ET
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(RAW/'TharuWordlistComparison.xlsx') as z:
  shared=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));out=[];item=0
  for row in root.findall('.//s:sheetData/s:row',ns):
   ri=int(row.attrib['r']);values=[None]*7
   for c in row.findall('s:c',ns):
    col=ord(re.match('[A-Z]+',c.attrib['r'])[0])-65
    v=c.find('s:v',ns)
    if v is not None:values[col]=shared[int(v.text)] if c.attrib.get('t')=='s' else v.text
   if values[0]=='Variety':continue
   if values[0] and not any(values[1:]):
    if ri>4:item+=1;gloss=values[0]
   elif values[0] and ri>6:out.append(dict(item=item,gloss=gloss,row=ri,variety=values[0],transcription=values[1],frame=values[2],aligned=values[3],grouping=values[4],notes=values[5],exclude=values[6]))
 assert item==295 and len(out)==2360
 return out

def extract():
 import logging,runpy
 from pypdf import PdfReader,PdfWriter
 logging.getLogger('pypdf').setLevel(logging.ERROR)
 for k in ['tamil-nadu','uttar-pradesh','bihar','sikkim2']:
  w=PdfWriter();w.append(PdfReader(RAW/(k+'.pdf')))
  with open('/tmp/'+k+'-normalized.pdf','wb') as f:w.write(f)
 runpy.run_path(str(RAW/'extract_tables.py'))
 (RAW/'tharu-cells.json').write_text(json.dumps(extract_workbook(),ensure_ascii=False,indent=2)+'\n')
def lexical_parts(k,r):
 text=r.get('text',r.get('transcription')) or ''
 notes=[]
 if k=='tharu':
  override=load('tharu-readings.json').get(str(r['row']))
  if override is not None:return override
  if not text or text.startswith('same as '):return []
  return [{'form':nfc(t.strip()),'gloss':r['gloss'],'notes':''} for t in re.split(r'\s+or\s+|\s+and\s+|/',text) if t.strip()]
 patch=load('image-transcriptions.json').get(k,{}).get(f"{r['item']}:{r['column']}")
 if r['images']:
  assert patch,(k,r)
  text=patch['text'];notes.append(patch['review'])
 if re.fullmatch(r'[\s_Xx–—-]*',text) or 'to be' in text or '#VALUE!' in text:return []
 if '(cid:' in text or '�' in text:return []
 if '\n' in text:notes.append('transcription: physical line wrapping retained as a space; word/phrase boundary requires review')
 # Join short glyph continuations only; unresolved long wraps remain flagged.
 text=re.sub(r'\n(?=[^ /,]{1,2}/)', '', text)
 if k=='danuwar' and (r['item'],r['column']) in {(87,3),(184,2),(186,2)}:text=text.replace('\n','')
 wrap=load('layout-corrections.json').get(k,{}).get(str(r['item'])+':'+str(r['column']))
 if wrap is not None:text=wrap
 text=' '.join(text.split())
 gloss=' '.join(r['gloss'].split())
 if k=='danuwar':
  gloss={49:'lightning',87:'chicken (young bird)',207:'we (inclusive)',208:'we (exclusive)'}.get(r['item'],gloss)
 # In these comparative tables commas and slashes separate listed alternatives.
 if k in ('tamil-nadu','uttar-pradesh') and '(to ' not in text:
  text=text.replace('(', '/').replace(')', '/')
 parts=[s.strip(' ;') for s in re.split(r'[/,]',text) if s.strip(' ;')]
 result=[]
 annotation=re.compile(r'\((man|woman|m|f|E|Y|younger|elder|small|shirt|Muslim|ploughing|instrumental|stick|head|hand|chase and catch|catch|while walking|fruit|door|eye|mirror|table|cloth|rope|split wood|wood|beat a drum|a drum|as a bird|move tram|tram|sow seed|seed|trams|a person|not good|nominative|to float|to hit|to shoot|I|il|h|hAh|i|s)\)')
 for part in parts:
  matches=list(annotation.finditer(part))
  if matches:
   start=0
   for mt in matches:
    form=part[start:mt.start()].strip();ann=mt.group(1);start=mt.end()
    if not form:
     if result:result[-1]['gloss']+=' ('+ann+')'
     continue
    gram={'m':'m','man':'m','f':'f','woman':'f','nominative':'nom'}
    g=gloss if ann in gram or ann in {'I','il','h','hAh','i','s'} else gloss+' ('+{'E':'elder','Y':'younger'}.get(ann,ann)+')'
    result.append(dict(form=nfc(form),gloss=g,tags=[gram[ann]] if ann in gram else [],notes='Source annotation: '+ann if ann in {'I','il','h','hAh','i','s'} else '',review=list(notes)))
   tail=part[start:].strip()
   if tail:result.append(dict(form=nfc(tail),gloss=gloss,notes='',review=list(notes)))
  elif not re.fullmatch(r'[_Xx–—()\s-]+',part):result.append(dict(form=nfc(part),gloss=gloss,notes='',review=list(notes)))
 return result

def build(out):
 out.mkdir(parents=True,exist_ok=True)
 for name,v in load('snapshot.json')['files'].items():
  path=RAW/name
  if not path.exists() and path.suffix in {'.pdf','.xlsx'}:continue  # hashed cell snapshots support offline installs
  assert hashlib.sha256(path.read_bytes()).hexdigest()==v['sha256'],name
 meta=load('language-map.json');patches=load('image-transcriptions.json')
 langs={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
 for a in meta['languages']:langs.setdefault(a[0],dict(zip(['ID','Name','Glottocode','Latitude','Longitude','Clade','Location','Quality'],a)))
 ds=[]
 for k,lects in meta['lects'].items():
  for m in lects:
   ds.append([m['id'],dialect_tag(m['base'],m['id'],m['name']),m['base'],m['source_label'],m['name'],m.get('glottocode',''),'','',langs[m['base']]['Clade'],m['location'],''])
 tags={r[0]:r[1] for r in ds}
 old=readcsv(ROOT/'data/other/forms/20260813-kochila-tharu.csv')
 # Stable glossary order differs after item 239 between workbook and publication.
 # Match through source gloss and lect; never equate the two item-number systems.
 oi=collections.defaultdict(list)
 for r in old:oi[(r[0],r[3].lower())].append(r)
 rows={};audits={};reuse={};report={}
 for k in SOURCES:
  raw=load(k+'-cells.json');assert len(raw)==EXPECTED[k],(k,len(raw))
  rows[k]=[];audits[k]=[]
  for r in raw:
   col=r.get('column')
   if k=='tharu':col=next(i for i,m in enumerate(meta['lects'][k]) if m['source_label']==r['variety'])
   m=meta['lects'][k][col];parts=lexical_parts(k,r)
   key=f'{k}:i{r["item"]}:c{col}'
   a=dict(entry_key=key,upstream=r,mapping=m,readings=[],status='skipped',reason='Blank, source omission, unresolved corrupt glyph, or nonlexical placeholder' if not parts else '')
   for vi,part in enumerate(parts,1):
    form=part['form'];gloss=part['gloss'];pk=key+f':v{vi}'
    loc=(f'Comparison row {r["row"]}, item {r["item"]}, {m["source_label"]}' if k=='tharu' else f'p. {r["printed_page"]}, item {r["item"]}, {m["source_label"]}')
    citation=SOURCES[k]+'['+loc+']'
    if k=='tharu' and col>=5:
     lect=['kochila_morang_east','kochila_bara_west','kochila_siraha_central'][col-5]
     candidates=oi[(lect,gloss.lower())]
     matches=[o for o in candidates if fingerprint(o[2])==fingerprint(form)]
     if len(matches)==1:
      target=matches[0][10];reuse.setdefault(target,[]).append(citation)
      a['readings'].append(dict(entry_key=pk,status='reused',target=target,form=form,gloss=gloss,citation=citation));continue
    review=part.get('review',[])+([m['review']] if m.get('review') else [])
    if k in ('tamil-nadu','uttar-pradesh','bihar','sikkim2'):
     # The published tables demonstrably contain gloss shifts and mixed transcription.
     review.append('gloss/transcription: published comparative table contains editorial inconsistencies; attested source pairing preserved')
    tt=[tags[m['id']]]+part.get('tags',[])+(['uncertain'] if review else [])
    if k=='danuwar' and r['item'] in (203,204,207,208):tt.append({203:'informal',204:'formal',207:'inclusive',208:'exclusive'}[r['item']])
    row=[m['base'],'',form,gloss,'','',part.get('notes',''),citation,('tharu-group:'+str(r['item'])+':'+str(r['grouping']) if k=='tharu' and r.get('grouping') else ''),'',pk,'','','',' '.join(tt)]
    rows[k].append(row);a['readings'].append(dict(entry_key=pk,status='unlinked',installed=row,review=review))
   if parts:a['status']='reused' if all(x['status']=='reused' for x in a['readings']) else 'unlinked'
   audits[k].append(a)
  if k in ('tamil-nadu','uttar-pradesh'):
   for j in range(12):audits[k].append(dict(entry_key=f'{k}:i94:c{j}',status='source-omitted',reason='Printed tables skip item 94 on both facing pages; no form invented'))
  assert len({r[10] for r in rows[k]})==len(rows[k])
  writecsv(out/f'20260911-census-{k}.csv',rows[k])
  (out/(k+'-audit.jsonl')).write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audits[k]))
  sample=random.Random(20260914).sample([a for a in audits[k] if a['status']=='unlinked'],20)
  (out/(k+'-sample.json')).write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
  report[k]={'raw_cells':len(raw),'installed':len(rows[k]),'statuses':dict(collections.Counter(a['status'] for a in audits[k])),'reused_readings':sum(x['status']=='reused' for a in audits[k] for x in a.get('readings',[])),'uncertain':sum('uncertain' in r[14].split() for r in rows[k])}
 # Append only citations that have exact canonical gloss+lect+transcription support.
 for r in old:
  for c in reuse.get(r[10],[]):
   if c not in r[7].split(';'):r[7]+=';'+c
 writecsv(out/'20260813-kochila-tharu.csv',old)
 (out/'tharu-reuse.json').write_text(json.dumps(reuse,ensure_ascii=False,indent=2)+'\n')
 writecsv(out/'languages.csv',meta['languages']);writecsv(out/'dialects.csv',ds)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def install(out):
 for k in SOURCES:
  name=f'20260911-census-{k}.csv';shutil.copyfile(out/name,ROOT/'data/other/forms'/name)
  for suffix in ['-audit.jsonl','-sample.json']:shutil.copyfile(out/(k+suffix),RAW/(k+suffix))
 for name in ['tharu-reuse.json','report.json']:shutil.copyfile(out/name,RAW/name)
 shutil.copyfile(out/'20260813-kochila-tharu.csv',ROOT/'data/other/forms/20260813-kochila-tharu.csv')
 for name in ['languages.csv','dialects.csv']:
  p=ROOT/'cldf'/name;rr=readcsv(p);ids={r[0] for r in rr}
  rr.extend(r for r in readcsv(out/name) if r[0] not in ids);writecsv(p,rr)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/tmp/census-proposal'));p.add_argument('--extract',action='store_true');p.add_argument('--install',action='store_true');a=p.parse_args()
 if a.extract:extract()
 print(json.dumps(build(a.output),indent=2))
 if a.install:install(a.output)
