"""Pinned Bajjika tables, Lindgren 2023 TSV, and DravLex CLDF. No network on import.

Run --output DIR for a review proposal, then --install. Source-local cognacy is
retained in Cognateset and the audit, not promoted to an invented reconstructed
ancestor. Exact republished DravLex observations retain both citations.
"""
import argparse, collections, csv, hashlib, json, random, re, shutil, sys, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RAW=Path(__file__).with_name('ia_dravidian_2026')
sys.path.insert(0,str(ROOT))
from dialects import dialect_tag
SOURCES={'bajjika':'regmi2014bajjika','lindgren':'lindgren2023dravidian','dravlex':'kolipakam2018dravlex'}
def nfc(s):return unicodedata.normalize('NFC',s.strip())
def read(name,delim=','):return list(csv.DictReader((RAW/name).open(),delimiter=delim))
def ipa(segments):return nfc(''.join(segments.split()).replace('+',' '))
def extract_bajjika():
 import pdfplumber
 pdf=pdfplumber.open(RAW/'bajjika.pdf');assert len(pdf.pages)==135
 out=[]
 for pi in range(114,122):
  page=pdf.pages[pi]
  for c in page.chars:
   if c['x0']>240 and c['text']=='h' and c['size']<10:c['text']='ʰ'
  for tab in page.extract_tables():
   for r in tab:
    if re.fullmatch(r'\d+\.',r[0] or ''):out.append(dict(item=int(r[0][:-1]),pdf_page=pi+1,printed_page=pi-11,gloss=r[1],cells=r[3:]))
    elif r[0]=='' and pi==119:
     assert out[-1]['item']==137
     for j,s in enumerate(r[3:]):out[-1]['cells'][j]+=s or ''
 assert [r['item'] for r in out]==list(range(1,211))
 return out
def decode_cell(text,item,col):
 # Line breaks within words are layout; these reviewed locations are real phrase boundaries.
 if item in (49,87) or (item,col) in [(39,2),(184,2),(186,2)]:text=text.replace('\n',' ')
 elif item in (173,174) and col==2:text=text.replace('kəni\nsəb','kəni səb')
 elif (item,col)==(210,4):text=text.replace('kəni\nsəb','kəni səb').replace('u\nsəb','u səb')
 text=text.replace('\n','').replace('(cid:1)','̃').replace('(cid:2)','ŋ').replace('(cid:3)','ɔ').replace('\uf02c','̣').replace('\uf01e','̣')
 assert '(cid:' not in text,text
 return nfc(text)
def concept(label):
 if label.startswith('*'):
  bits=label[1:].split('.')
  tags=[]
  for b in bits:
   if b in ['2SG','3SG','3PL']:tags+=['pron',b.lower()]
   elif b in {'F','M','I','A','H'}:tags.append({'F':'f','M':'m','I':'inanimate','A':'animate','H':'honorific'}[b])
  names={'2SG':'second-person singular','3SG':'third-person singular','3PL':'third-person plural','F':'female','M':'male','I':'inanimate','A':'animate','H':'honorific','P':'proximate','R':'remote'}
  return ', '.join(names.get(b,b.lower()) for b in bits),tags
 return label.lower(),[]
def build(out):
 out.mkdir(parents=True,exist_ok=True)
 for name,v in json.loads((RAW/'snapshot.json').read_text())['files'].items():
  if 'sha256' in v:assert hashlib.sha256((RAW/name).read_bytes()).hexdigest()==v['sha256'],name
 lm=json.loads((RAW/'language-map.json').read_text());base={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
 additions=[['OllariGadaba','Ollari Gadaba','pott1240','18.5968','82.7586','C. Dravidian','Glottolog reference point; not a survey site','C'],['Ravula','Ravula','ravu1237','12.3231','75.6265','S. Dravidian I','Glottolog reference point; not a survey site','C'],['Pattapu','Pattapu','patt1247','','','S. Dravidian I','Coastal southern Andhra Pradesh, India','']]
 fields=list(next(iter(base.values())))
 for a in additions:base.setdefault(a[0],dict(zip(fields,a)))
 dialects=[];existing_d={r['ID']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 def tag_for(d):return existing_d[d]['Tag'] if d in existing_d else next(r[1] for r in dialects if r[0]==d)
 for doc,m in lm.items():
  if m['dialect'] and m['dialect']!='koya':
   label={'F_Bhat':'Madhwa Brahmin Tulu','F_Byari':'Byari'}.get(doc,doc[2:].replace('_',' '));d=m['dialect'];b=m['base']
   dialects.append([d,dialect_tag(b,d,label),b,doc,label,'','','',base[b]['Clade'],'Named variety; no source-locality coordinates supplied',''])
 sites=['Malangawa','Barahathawa','Garuda','Katahariya','Gaur']
 gps=[((26,51,34.9),(85,33,49.4)),((27,0,10.6),(85,27,52.6)),((26,57,3.1),(85,19,8.8)),((26,59,5.4),(85,14,19.1)),((26,45,58.8),(85,16,18.7))]
 for s,(lat,lon) in zip(sites,gps):
  d='bajjika-'+s.lower();dialects.append([d,dialect_tag('Mth',d,'Bajjika '+s),'Mth',d,'Bajjika '+s,'bajj1234',str(lat[0]+lat[1]/60+lat[2]/3600),str(lon[0]+lon[1]/60+lon[2]/3600),'Bihari',s+', Nepal; Regmi et al. 2014, table 2.3','A'])
 audits={k:[] for k in SOURCES};rows={k:[] for k in SOURCES}
 def emit(k,key,lang,form,gloss,phonemic='',tags=None,source_extra='',cog='',raw=None,reason=''):
  source=f'{SOURCES[k]}[{key}]'+(';' +source_extra if source_extra else '')
  r=[lang,'',nfc(form),nfc(gloss),'',phonemic,'',source,cog,'',key,'','','',' '.join(tags or [])]
  assert form and lang in base,(key,lang)
  rows[k].append(r);audits[k].append(dict(entry_key=key,status='unlinked',reason=reason,upstream=raw,installed=r))
 for r in extract_bajjika():
  item=r['item'];gloss=r['gloss'].replace('\n',' ').strip()
  fixes={49:'lightning',69:'wheat (husked)',71:'rice (husked)',74:'groundnut',79:'cauliflower',196:'to run',197:'to go',199:'to speak',200:'to hear; to listen',201:'to look',207:'we (inclusive)',208:'we (exclusive)'}
  gloss=fixes.get(item,gloss)
  for j,cell in enumerate(r['cells']):
   decoded=decode_cell(cell,item,j)
   for vi,form in enumerate(decoded.split('/'),1):
    key=f'bajjika:p{r["printed_page"]}:i{item}:{sites[j].lower()}:v{vi}'
    tags=[tag_for('bajjika-'+sites[j].lower())]
    if item in (203,204,207,208):tags.append({203:'informal',204:'formal',207:'inclusive',208:'exclusive'}[item])
    # Repeated source variants remain distinct records. Source anomalies are preserved.
    reason='Font decoding; slash alternatives; page-wrap join reviewed'
    if item==140 and form.strip()=='dur':tags.append('uncertain');reason+='; gloss:source lists dur under near'
    emit('bajjika',key,'Mth',form,gloss,tags=tags,raw=dict(r,site=sites[j],raw_cell=cell),reason=reason)
 params={x['ID']:x for x in read('dravlex-parameters.csv')};ds=read('dravlex-forms.csv');ls=read('lindgren.tsv','\t')
 # Exact canonical segment + language + concept matching, with special pronoun labels.
 idx=collections.defaultdict(list)
 for l in ls:
  if l['DOCULECT'].startswith('P_'):idx[(l['DOCULECT'][2:],l['CONCEPT'],ipa(l['SEGMENTS']))].append(l)
 matched=set(); targets={}; annotations=collections.defaultdict(list)
 for c in read('dravlex-cognates.csv'):annotations[c['Form_ID']].append(c)
 sourcekeys={'KolipakamFW':'kolipakam2018fieldnotes','Andronov1964':'andronov1964lexicostatistics','BurrowEmeneau1961':'burrow-emeneau1961ded'}
 for d in ds:
  m=lm[d['Language_ID']];par=params[d['Parameter_ID']];cg=par['Concepticon_Gloss']
  cg={'BURN':'BURN (SOMETHING)','FAT (FROM ANIMALS)':'FAT (ORGANIC SUBSTANCE)','KNOW':'KNOW (SOMETHING)','HEAD LOUSE':'LOUSE','MEAT':'FLESH OR MEAT','SKIN':'LEATHER OR HIDE','STONE':'STONE OR ROCK','WE':'WE (INCLUSIVE)','ROAD':'PATH OR ROAD'}.get(cg,cg)
  if par['Name'].lower()=='this':cg='*3SG.I.P'
  if par['Name'].lower()=='that':cg='*3SG.I.R'
  matches=idx.get((d['Language_ID'],cg,ipa(d['Segments'])),[])
  extra=';'.join(sourcekeys[x] for x in d['Source'].split(';'))
  for l in matches:
   matched.add(l['ID']);targets.setdefault(l['ID'],[]).append('dravlex:'+d['ID']);extra+=f';{SOURCES["lindgren"]}[row {l["ID"]}, {l["DOCULECT"]}]'
  tags=[tag_for(m['dialect'])] if m['dialect'] else []
  if d['Loan']=='true':tags+=['loanword']
  emit('dravlex','dravlex:'+d['ID'],m['base'],d['Value'],par['Name'],ipa(d['Segments']),tags,extra,'dravlex:'+d['Cognacy'],raw=dict(d,cognate_annotations=annotations[d['ID']],republished_lindgren_ids=[l['ID'] for l in matches]),reason='Source cognacy retained; no reconstructed ancestor asserted; donor unspecified for loans')
 for l in ls:
  if l['ID'] in matched:
   audits['lindgren'].append(dict(entry_key='lindgren:'+l['ID'],status='reused-dravlex',target_entry_keys=targets[l['ID']],upstream=l,reason='Exact language, concept and segmented form match; citation retained on DravLex record'));continue
  doc=l['DOCULECT'];m=lm[doc if doc.startswith('F_') else doc[2:]];gloss,tags=concept(l['CONCEPT'])
  if m['dialect']:tags.insert(0,tag_for(m['dialect']))
  emit('lindgren','lindgren:'+l['ID'],m['base'],l['IPA'],gloss,ipa(l['SEGMENTS']),tags,cog='lindgren:'+l['COGSET'],raw=l,reason='Source cognate-set annotation preserved; not an ancestry assertion')
 report={}
 for k in SOURCES:
  stem='20260911-'+k
  with (out/(stem+'.csv')).open('w') as f:csv.writer(f).writerows(rows[k])
  with (out/(stem+'-audit.jsonl')).open('w') as f:
   for a in audits[k]:f.write(json.dumps(a,ensure_ascii=False)+'\n')
  (out/(stem+'-sample.json')).write_text(json.dumps(random.Random(20260912).sample([a for a in audits[k] if a['status']=='unlinked'],20),ensure_ascii=False,indent=2)+'\n')
  report[k]={'installed':len(rows[k]),'audit_records':len(audits[k]),'statuses':dict(collections.Counter(a['status'] for a in audits[k]))}
  assert len({r[10] for r in rows[k]})==len(rows[k])
 with (out/'languages.csv').open('w') as f:csv.writer(f).writerows(additions)
 with (out/'dialects.csv').open('w') as f:csv.writer(f).writerows(dialects)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def install(out):
 for k in SOURCES:
  stem='20260911-'+k;shutil.copyfile(out/(stem+'.csv'),ROOT/'data/other/forms'/(stem+'.csv'))
  for s in ['-audit.jsonl','-sample.json']:shutil.copyfile(out/(stem+s),RAW/(stem+s))
 for name in ['languages.csv','dialects.csv']:
  p=ROOT/'cldf'/name;rows=list(csv.reader(p.open()));ids={r[0] for r in rows}
  for r in csv.reader((out/name).open()):
   if r[0] not in ids:rows.append(r);ids.add(r[0])
  with p.open('w') as f:csv.writer(f).writerows(rows)
 shutil.copyfile(out/'report.json',RAW/'report.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/tmp/ia-dravidian-proposal'));p.add_argument('--install',action='store_true');a=p.parse_args();print(json.dumps(build(a.output),indent=2))
 if a.install:install(a.output)
