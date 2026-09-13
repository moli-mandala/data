"""Five pinned survey tables; three unavailable/misattributed appendices audited separately.

--output stages a proposal; --install installs reproducibly from hashed cell snapshots.
No network, no inferred etymologies. Raw cells retain physical layout and exact locators.
"""
import argparse,collections,csv,hashlib,json,random,re,shutil,sys,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RAW=Path(__file__).with_name('more_surveys_2026')
sys.path.insert(0,str(ROOT))
from dialects import dialect_tag
SOURCES={'jharkhand':'census2023jharkhand','himachal':'census2023himachal','rajasthan':'census2011rajasthan','west-bengal':'census2016westbengal','kisan':'mahato2014kisan'}
EXPECTED={'jharkhand':4000,'himachal':7042,'rajasthan':3500,'west-bengal':1500,'kisan':1050}
def load(n):return json.loads((RAW/n).read_text())
def nfc(s):return unicodedata.normalize('NFC',s)
def space(s):return nfc(' '.join(s.split()))
def writecsv(p,rows):
 with p.open('w',newline='') as f:csv.writer(f).writerows(rows)
def cellkey(k,r):return f"more-{k}:p{r['printed_page']}:i{r['item']}:c{r['column']}:y{r['bbox'][1]:.2f}"
GRAM={'m':'m','m.':'m','male':'m','man':'m','masc':'m','masc.':'m','f':'f','f.':'f','female':'f','woman':'f','lady':'f','fem':'f','fem.':'f','sg':'sg','sg.':'sg','singular':'sg','pl':'pl','pl.':'pl','plural':'pl','hon.':'honorific','n.':'noun','trans':'tr','in trans':'intr','nominative':'nom','verbal adj.':'adj','ordinary':'informal','inferior':'informal','taboo':'vulgar'}
# Explicitly reviewed source-language material in brackets is retained as source notation,
# not mistaken for an English annotation. Optional endings remain exact, not expanded by guesswork.
LEXICAL={'ga:y','sãp','saDhã','maNus','ke','git','ghass nila','chole-chalna','kheL','ciDi','hal','agla','pichla','be','an','jimmi','h','gaDDi','mÒla','tatta','cha:chi','bidhu','sĕra','meratãye','ba? r','jija','daroji','inna','pakhir','miTho','a','no','oa','Dhak','bARo','choTo','de','goli','jansũ marNO','phAsAl','udhar'}
GLOSSANN={'E':'elder','Y':'younger','elder':'elder','younger':'younger','big':'big','small':'small','big variety':'big variety','small variety':'small variety','much cold':'very cold','woodused as fuel':'wood used as fuel','unused wood':'unused wood','younger brother’s wife':'younger brother’s wife','sparrow':'sparrow','dhandameans ‘cattle’':'cattle','concrete building':'concrete building','kacca house':'kacca house','big animal’stail':'big animal’s tail','small animal’s tail':'small animal’s tail','stomach':'stomach','bulged stomach':'bulged stomach','stem':'stem','colour of sky':'colour of sky','weight':'weight','light':'light','maximum white':'intensely white','maximum yellow':'intensely yellow','on head':'on head','in hand':'in hand','inhand':'in hand','on shoulder':'on shoulder','an animal':'an animal','something thrown':'something thrown','sorting':'sorting','self':'self','self burn':'self burn','to clean the middle of the land':'to clean the middle of the land','related to face':'related to face','related to habit':'related to habit','“dirty”':'dirty','left':'left','right':'right','not good':'not good','husband’s sister':'husband’s sister','wife’s brother':'wife’s brother','husband’s brother':'husband’s brother','a day labourer':'a day labourer','stick':'stick','pot':'pot','rope':'rope','chase and catch':'chase and catch','eyes':'eyes','eye':'eye','utensils':'utensils','cloth':'cloth','meat':'meat','rip cloth':'rip cloth','in bad sense':'in a bad sense','hand touch':'hand touch','human beings':'human beings','others':'others','on ground':'on ground','from the':'from the','animal':'animal','tree':'tree','by me':'by me','this side':'this side','white flower':'white flower','stick & pot':'stick and pot','chase and catch an animal':'chase and catch an animal'}
def parse(k,r):
 key=cellkey(k,r);text=r['text'];gloss=space(r['gloss'])
 patches=load('readings.json')
 if key in patches:return patches[key]
 if r['images']:return [] # source's embedded forms have missing-glyph boxes; audited, not guessed
 if not text.strip() or re.fullmatch(r'[\s_–—-]+',text):return []
 if '(cid:' in text or '�' in text:return []
 # Printed physical line wraps are retained as a space unless individually reviewed.
 text=space(text)
 review=[]
 if '\n' in r['text']:review.append('transcription: physical line wrap; boundary retained as space')
 # Every source has printed inconsistencies. Preserve supplied form-gloss pairings.
 review.append('transcription/gloss: source attestation preserved; inconsistent notation or prompt meaning is not silently emended')
 # Protect brackets that are explicitly lexical notation. Arbitrary commentary must be classified.
 annotations={};idx=0
 def bracket(mt):
  nonlocal idx
  a=space(mt.group(1))
  if a in LEXICAL or (k=='himachal' and a=='u'):return mt.group(0)
  tok=f'§{idx}§';annotations[tok]=a;idx+=1;return tok
 text=re.sub(r'\(([^()]*)\)',bracket,text)
 # Hard cases, nested brackets and malformed parentheses require an explicit reading patch.
 if '[' in text or ']' in text or any(a not in GRAM and a not in GLOSSANN and a not in {'I','il','i','sikaripara','dialectal variation','H. also','hAh','hans','in hans','originalword-made of cloth','indicates anaj','caro means ‘fodder’','DhuNDhO - rejected house'} for a in annotations.values()):
  raise ValueError((key,'unclassified annotation',text,annotations))
 parts=re.split(r'\s*[,/~;]\s*',text);out=[]
 for part in parts:
  # A trailing annotation closes a form, even when the source omits a comma.
  chunks=re.split(r'(§\d+§)',part);pending=''
  for chunk in chunks:
   if chunk in annotations:
    a=annotations[chunk]
    if pending.strip():out.append({'form':pending.strip(),'gloss':gloss,'tags':[],'review':list(review)});pending=''
    if not out:raise ValueError((key,'leading annotation',a))
    if a in GRAM:
     out[-1]['tags'].append(GRAM[a])
     if a=='taboo':out[-1].setdefault('notes',[]).append('Source register: taboo')
    elif a in GLOSSANN:out[-1]['gloss']+=' ('+GLOSSANN[a]+')'
    else:out[-1].setdefault('notes',[]).append('Source annotation: '+a)
   else:pending+=chunk
  if pending.strip():out.append({'form':pending.strip(),'gloss':gloss,'tags':[],'review':list(review)})
 return out

def build(out):
 out.mkdir(parents=True,exist_ok=True)
 for name,digest in load('snapshot.json')['extraction_sha256'].items():
  assert hashlib.sha256((RAW/name).read_bytes()).hexdigest()==digest,name
 meta=load('language-map.json');langs={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
 for a in meta['languages']:langs.setdefault(a[0],dict(zip(['ID','Name','Glottocode','Latitude','Longitude','Clade','Location','Quality'],a)))
 ds=[]
 for lects in list(meta['lects'].values())+[meta.get('extra_dialects',[])]:
  for m in lects:ds.append([m['id'],dialect_tag(m['base'],m['id'],m['name']),m['base'],m['source_label'],m['name'],'','','',langs[m['base']]['Clade'],m['location'],''])
 dt={r[0]:r[1] for r in ds};report={}
 for k,source in SOURCES.items():
  raw=load(k+'-cells.json');assert len(raw)==EXPECTED[k]
  audit=[];rows=[]
  for r in raw:
   key=cellkey(k,r);m=meta['lects'][k][r['column']];parts=parse(k,r)
   a={'entry_key':key,'raw':r,'mapping':m,'status':'unlinked' if parts else 'excluded','reason':'' if parts else 'Source blank/placeholder or damaged source glyph (see raw images/text)','readings':[]}
   for j,p in enumerate(parts,1):
    form=space(p['form']);assert form and not any(x in form for x in ['§','�','(cid:']),key
    tags=[dt[m['id']]]+p.get('tags',[])
    if p.get('review') or k!='kisan':tags.append('uncertain')
    citation=f"{source}[p. {r['printed_page']}, item {r['item']}, {m['source_label']}]"
    row=[m['base'],'',form,p.get('gloss',space(r['gloss'])),'','','; '.join(p.get('notes',[])),citation,'','',key+f':v{j}','','','',' '.join(dict.fromkeys(tags))]
    rows.append(row);a['readings'].append({'row':row,'review':p.get('review',[])})
   audit.append(a)
  assert len(set(r[10] for r in rows))==len(rows),k
  writecsv(out/f'20260911-more-{k}.csv',rows)
  (out/(k+'-audit.jsonl')).write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
  report[k]={'raw_cells':len(raw),'installed_rows':len(rows),'excluded_cells':sum(not a['readings'] for a in audit),'uncertain_rows':sum('uncertain' in r[14].split() for r in rows)}
 writecsv(out/'languages.csv',meta['languages']);writecsv(out/'dialects.csv',ds)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def install(out):
 for k in SOURCES:
  shutil.copyfile(out/f'20260911-more-{k}.csv',ROOT/f'data/other/forms/20260911-more-{k}.csv')
  shutil.copyfile(out/(k+'-audit.jsonl'),RAW/(k+'-audit.jsonl'))
 for fn in ['languages.csv','dialects.csv']:
  path=ROOT/'cldf'/fn;old=list(csv.reader(path.open()));ids={r[0] for r in old};old.extend(r for r in csv.reader((out/fn).open()) if r[0] not in ids);writecsv(path,old)
 shutil.copyfile(out/'report.json',RAW/'report.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/tmp/more-surveys-proposal'));p.add_argument('--install',action='store_true');a=p.parse_args();print(json.dumps(build(a.output),indent=2))
 if a.install:install(a.output)
