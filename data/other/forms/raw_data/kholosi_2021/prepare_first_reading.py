"""Extract a complete review draft; never install. All source fields remain in audits."""
from pathlib import Path
import csv,json,re,hashlib,collections,unicodedata
P=Path(__file__).resolve().parent
units=[json.loads(l) for l in (P/'pdf-unit-census.jsonl').open()]
def clean(s):
 s=s.replace("kh-\njust", "kh- just").replace("vaxt-\ne.", "vaxt-e.")
 # Only compositor line-end word breaks; source-line evidence is never changed.
 return re.sub(r'\s+',' ',re.sub(r'-\s*\n\s*(?=\w)','',s)).strip()
records=[]
parent_key=None
for u in units:
 r={k:u[k] for k in ('source_unit_key','kind','printed_page','pdf_page','column','raw_text')};r['review_status']='pending independent source and segmentation review';t=clean(u['raw_text']).removeprefix('▷').strip();r['joined_text']=t
 if u['kind']=='head':parent_key=u['source_unit_key']
 else:r['parent_head_key']=parent_key
 r['layout_evidence']=[{'baseline':l['baseline'],'raw_text':l['raw_text']} for l in u['lines']]
 if '[' in t and ']' in t:
  form,tail=t.split('[',1);ipa,tail=tail.split(']',1);qualifiers=re.findall(r'\b[AN]\.',form)
  form=re.sub(r'(?:\s+[AN]\.)+$','',form.strip());hom=re.search(r'(\d+)$',form)
  if hom:r['homonym_number']=hom.group(1);form=form[:hom.start()]
  r.update(form=form.strip(),phonetic=ipa.strip(),source_qualifiers=qualifiers)
  analysis,sep,etym=tail.strip().partition('Etym:');r['etymology']=etym.strip() if sep else '';r['source_analysis']=analysis.strip()
  if analysis.startswith('=') or analysis.startswith('see '):r['cross_reference']=analysis.strip();r['gloss']='';r['tags']=[]
  else:
   m=re.match(r'\s*(n|v|adj|a|adv|pron|num|post|conj|intj|part|suf|det|prep)\s*\.\s*(.*)',analysis)
   if m:r['source_pos']=m.group(1);r['gloss']=m.group(2).strip().removesuffix('.');r['tags']=[dict(n='noun',v='verb',adj='adj',a='adj',adv='adv',pron='pron',num='num',post='postp',conj='conj',intj='interj',part='part',suf='suffix',det='determiner',prep='prep')[m.group(1)]]
   else:r['gloss']=analysis.strip();r['tags']=[];r['unparsed_analysis']=True
 elif t.startswith('Plural:'):
  r.update(form=t.split(':',1)[1].strip(),gloss='',tags=['pl'],parent_scope='preceding head, including page boundary')
 elif '‘' in t and t.endswith('’'):
  form,gloss=t.split('‘',1);r.update(form=form.strip(),gloss=gloss[:-1],tags=[],expression_classification_pending=True)
 else:r.update(form='',gloss='',tags=[],unparsed_record=True)
 records.append(r)
(P/'pdf-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
raw_csv=[]
for fn in ('lexicon.csv','sentences.csv'):
 rows=list(csv.reader((P/'upstream'/fn).open()))
 for i,row in enumerate(rows[1:],2):
  raw_csv.append(dict(source_unit_key=f'kholosi:github:18b43f982efa:{fn}:row{i}',source_file=fn,physical_csv_row=i,header=rows[0],raw_fields=row,status='candidate' if row[0] else 'blank-source-form',review_status='pending cross-representation reconciliation'))
(P/'structured-source-census.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw_csv))
print('PDF',len(records),'heads',sum(r['kind']=='head' for r in records),'subentries',sum(r['kind']=='subentry' for r in records))
print('Unparsed',[(r['source_unit_key'],r['joined_text']) for r in records if r.get('unparsed_record') or r.get('unparsed_analysis')])
print('POS',collections.Counter(r.get('source_pos','other') for r in records))
