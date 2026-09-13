"""Build proposed rich rows and explicit decisions from the typographic extraction."""
import argparse,collections,csv,gzip,json,re,unicodedata
from pathlib import Path
from parse import candidates,flatten,PAT,GRAM
from languages import MAP
from cache import load_records
from west_pahari import PROMOTIONS, resolve_language
RAW=Path(__file__).parent;ROOT=RAW.parents[4];SOURCE='zoller2023'

def split_forms(s):
 out=[];start=0;depth=0
 for i,c in enumerate(s):
  if c in '([':depth+=1
  elif c in ')]':depth-=1
  elif c in ',/' and depth==0:out.append(s[start:i].strip());start=i+1
 out.append(s[start:].strip())
 return [x for x in out if x]

def direct_claim(rec,cc,valid):
 """Resolve only a single explicitly numbered main OIA derivation, never comparisons or components."""
 text,spans=flatten(rec)
 rel=re.search(r'<\s*(?:OIA\s*)?(?:lex\.\s*)?\*?[^‘<>]{1,65}‘[^’]+’\s*\((\d+[a-z]?(?:\.\d+)?)\)',text)
 if not rel:return None
 pid=rel[1].replace('.','-')
 if pid not in valid:return None
 # This conservative route leaves all qualified/compound/multiple claims in prose and audit.
 prefix=text[:rel.end()+2]
 if re.search(r'\?|\b(?:perhaps|probably|possibly|may|might|could|doubtful|unclear|unlikely|not|second|first component|plus|suggest|seems|connected|compared|comparison)\b',prefix,re.I):return None
 if text[:rel.start()].count('<') or text[:rel.start()].count('>'):return None
 if re.search(r'\b(?:wrong|incorrect|reject|rejected|untenable|doubtful|unlikely|unconvincing)\b|not (?:convincing|correct|related)',text,re.I):return None
 # A numbered derivation must govern a plain initial list, not an earlier comparison,
 # embedded example, or another clause. Citation parentheses are harmless here.
 front=text[:rel.start()]
 for a,b,_ in reversed(spans):
  if b<=rel.start():front=front[:a]+' '*(b-a)+front[b:]
 front=re.sub(r'‘[^’]*’',' ',front)
 front=re.sub(r'\([^()]*\)|\[[^][]*\]',' ',front)
 front=PAT.sub(' ',front);front=re.sub(GRAM,' ',front)
 front=re.sub(r'\b(?:and|or|all|both|derives|derive|is|a|borrowing|borrowed|from|directly)\b',' ',front)
 if re.search(r'[^\s\d.,;:()/*-]',front):return None
 return {'target':pid,'position':rel.start(),'printed':rel[1],'kind':'borrowed' if re.search(r'\bborrow(?:ed|ing)\b',prefix,re.I) else 'reflex'}

def build():
 records=load_records();mapping=json.loads((RAW/'language-map-proposed.json').read_text())
 registry={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())};registry.update(json.loads((RAW/'new-languages-proposed.json').read_text()))
 registry.update({r['ID']:r for r in PROMOTIONS.values()})
 dialects=list(csv.DictReader((ROOT/'cldf/dialects.csv').open()));dialect_by_name=collections.defaultdict(list)
 def norm(s):return ''.join(c for c in unicodedata.normalize('NFD',s).lower() if c.isascii() and c.isalnum())
 for d in dialects:dialect_by_name[(d['Language_ID'],norm(d['Name']))].append(d)
 proposed_dialects={}
 def dialect_tag(lang,name):
  if not name:return ''
  existing=dialect_by_name[(lang,norm(name))]
  if len(existing)==1:
   if existing[0]['ID'].startswith('zoller-'):proposed_dialects[existing[0]['ID']]=existing[0]
   return existing[0]['Tag']
  if len(existing)>1:return sorted(existing,key=lambda x:len(x['ID']))[0]['Tag']
  key='zoller-'+norm(lang)+'-'+norm(name)
  from urllib.parse import quote
  tag=f'dialect:{lang}:{key}:{quote(name)}'
  proposed_dialects[key]={'ID':key,'Tag':tag,'Language_ID':lang,'Source_Language_ID':f'Zoller2023:{name}','Name':name,'Glottocode':'bang1335' if name=='Bangani' else '', 'Latitude':'','Longitude':'','Clade':registry[lang]['Clade'],'Location':f"{name}; source variety identified by Zoller 2023, pp. XIII–XIX; exact field locality coordinates not supplied",'Quality':''}
  return tag
 valid={r['ID'] for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['Status']=='entry'}
 rows=[];audits=[];recordaudit=[]
 for rec in records:
  cc=candidates(rec);claim=direct_claim(rec,cc,valid)
  raw,_=flatten(rec);endpage=max([x.get('page',rec['page']) for x in rec['lines']] or [rec['page']]);start=len(rows)
  for c in cc:
   decision=dict(c);decision.pop('raw');decision['end_page']=endpage;decision['rows']=[]
   if c['status']!='candidate':audits.append(decision);continue
   labels=c['labels'] or [c['label']]
   if c['language']=='Bur' and c['dialect'] in ('Hunza','Nagar','Yasin'):labels=[c['label']]
   for li,label in enumerate(labels):
    lm=mapping.get(label)
    if not lm:
     decision['status']='unresolved-language';continue
    lang=lm['language'];dialect=lm['dialect']
    if label==c['label'] and c['dialect']:dialect=c['dialect']
    if c['language']=='Bur' and c['dialect'] in ('Hunza','Nagar','Yasin'):dialect=c['dialect']
    lang,dialect=resolve_language(lang,dialect)
    if lang in ('Gondi','Kui'):decision['status']='excluded-prior-user-scope';continue
    if lang not in registry:decision['status']='unregistered-language';continue
    for vi,form in enumerate(split_forms(c['form'])):
     form=unicodedata.normalize('NFC',re.sub(r'\s+',' ',form.strip(' ,;“”')))
     # Reject fragments, meta-symbols and damaged extractions, while retaining their evidence.
     if not form or '�' in form or any(c in form for c in '<>←→') or unicodedata.combining(form[0]) or any(ord(ch)<32 for ch in form) or len(form.split())>14:
      decision['status']='unresolved-transcription';continue
     if not c['gloss'] or c['gloss'].strip() in ('ditto','id.','idem'):
      decision['status']='unresolved-gloss';continue
     tags=list(c['tags']);dt=dialect_tag(lang,dialect)
     if vi and form.startswith('-'):
      tags.append('uncertain');decision.setdefault('review',[]).append('transcription: source elliptical replacement form; omitted stem not silently expanded')
     if dt:tags.append(dt)
     gloss=c['gloss'];gloss=re.sub(r'(\w)-\s+(\w)',r'\1-\2',gloss)
     param='';cog='';link_status='unlinked-source-analysis'
     if claim and c['span'][1]<=claim['position']:
      # IA targets cannot silently turn cross-family parallels into inheritance.
      ia_clades={'W. Pahari','C. Pahari','E. Pahari','Eastern','Bihari','Rajasthanic','Migratory','Kohistani','Shinaic','Sindhic','Lahndic','W. Hindi','Kunar','Kashmiric','Gujaratic','E. Hindi','Marathi-Konkani','Insular','Punjabic','Halbic','Chitrali','Pashai','Bhil','MIA','Early NIA'}
      clade=registry[lang]['Clade']
      if (clade in ia_clades or claim['kind']=='borrowed') and lang not in ('Sk','IA','Indo-Aryan'):
       param=claim['target'];cog='0:Indo-Aryan →' if claim['kind']=='borrowed' else '';link_status=claim['kind']
     pages=f'p. {rec["page"]}' if endpage==rec['page'] else f'pp. {rec["page"]}–{endpage}'
     unit='footnote' if rec['section']=='footnote' else 'table row' if rec['section']=='18.6' else 'entry'
     cite=f'{SOURCE}[{pages}, {rec["section"]}, {unit} {rec["number"]}]'
     if param:cite+=f';CDIAL[{claim["printed"]}]'
     for dedr in dict.fromkeys(re.findall(r'DEDR\s*(?:no\.\s*)?([0-9]+)',raw)):
      cite+=f';dedr[{dedr}]'
     key=c['key']+f':lect{li+1}:form{vi+1}'
     row=[lang,param,form,gloss,'','','',cite,cog,raw,key,'','','',' '.join(dict.fromkeys(tags))]
     letters=[c for c in form if c.isalpha()]
     if lang=='Gk' and letters and all('GREEK' in unicodedata.name(c,'') for c in letters):row[4]=form
     if '�' in raw:
      decision.setdefault('review',[]).append('transcription: undecoded glyph in surrounding analysis; raw evidence preserved in record audit')
      row[9]=raw.replace('�','[undecoded source glyph]')
     row=[unicodedata.normalize('NFC',v) for v in row]
     rows.append(row);decision['rows'].append(key);decision['status']='ingested';decision['link_status']=link_status;decision['target']=param
     decision.setdefault('resolved_rows',[]).append({'key':key,'language':lang,'dialect':dialect})
   audits.append(decision)
  # Only explicit same-lect "or" alternates receive a variant edge.
  local_rows={r[10]:r for r in rows[start:]}
  local_audits=[c for c in audits if c['record']==rec['key'] and c['rows']]
  for prev,cur in zip(local_audits,local_audits[1:]):
   gap=raw[prev['span'][1]:cur['span'][0]]
   gap=re.sub(r'‘[^’]*’|\([^()]*\)',' ',gap).strip(' ,;')
   if gap=='or' and prev['language']==cur['language'] and prev['gloss']==cur['gloss'] and len(prev['rows'])==len(cur['rows'])==1:
    r=local_rows[cur['rows'][0]];r[11]=prev['rows'][0];r[1]=r[8]='';cur['link_status']='variant';cur['target_key']=prev['rows'][0]
  recordaudit.append({'key':rec['key'],'page':rec['page'],'end_page':endpage,'section':rec['section'],'number':rec['number'],'text':raw,'rows':len(rows)-start,'status':'ingested' if len(rows)>start else 'no-resolved-attestation','claim':claim,'unresolved_relations':'Other comparisons, components and qualified alternatives retained as source analysis; no inferred ancestry','unresolved_auxiliary_citations':re.findall(r'[A-Z][\w-]+\s*\[?\(?\d{4}[a-z]?[^)\]]{0,45}',raw)})
 return rows,audits,recordaudit,list(proposed_dialects.values())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');args=p.parse_args()
 rows,audit,recs,dialects=build()
 with (RAW/'proposed.csv').open('w') as f:csv.writer(f).writerows(rows)
 for name,data in [('audit',audit),('record-audit',recs)]:
  with gzip.open(RAW/(name+'.jsonl.gz'),'wt',encoding='utf8') as f:
   for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 (RAW/'new-dialects-proposed.json').write_text(json.dumps(dialects,ensure_ascii=False,indent=2)+'\n')
 summary={'records':len(recs),'candidates':len(audit),'rows':len(rows),'linked':sum(bool(r[1]) for r in rows),'languages':len(set(r[0] for r in rows)),'new_dialects':len(dialects),'decisions':dict(collections.Counter(c['status'] for c in audit))}
 (RAW/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
 if args.install:
  import shutil
  shutil.copyfile(RAW/'proposed.csv',RAW.parents[1]/'20260913-zoller-linguistic-data.csv')
  used={r[0] for r in rows}
  for filename,new in [('languages.csv',list(json.loads((RAW/'new-languages-proposed.json').read_text()).values())+list(PROMOTIONS.values())),('dialects.csv',dialects)]:
   path=ROOT/'cldf'/filename
   with path.open() as f:
    reader=csv.DictReader(f);fields=reader.fieldnames;existing={r['ID'] for r in reader}
   with path.open('a',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    for r in new:
     if r['ID'] not in existing and (filename=='dialects.csv' or r['ID'] in used):writer.writerow(r)
  chars=sorted(set(''.join(r[2] for r in rows))-{' '})
  with (ROOT/'conversion/zoller-2023.txt').open('w') as f:
   f.write('Grapheme\tIPA\n')
   for c in chars:f.write(c+'\t'+c+'\n')
