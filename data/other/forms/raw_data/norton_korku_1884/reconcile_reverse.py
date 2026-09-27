"""Integrate complete manually read reverse heads; reuse only exact form + lexical gloss."""
import csv
import re
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
SOURCE='cust1884korku'
DIALECT='dialect:ko:norton-1884:Korku%20%28Norton%201884%29'

def inventory():
 out=[]
 for name in ('reverse_middle_pages_review.tsv','reverse_p173_independent_review.tsv','reverse_p174_independent_review.tsv','reverse_final_pages_review.tsv'):
  for r in csv.DictReader((ROOT/name).open(),delimiter='\t'):
   out.append(dict(page=int(r.get('printed_page') or r.get('page')),column=r['column'],ordinal=int(r.get('ordinal') or r.get('printed_ordinal')),form=(r.get('reviewed_source_form') or r.get('form')).replace(' / ','|'),gloss=r.get('reviewed_english_gloss',r.get('gloss')),review_note=r['review_note']))
 assert Counter(x['page'] for x in out)=={172:56,173:75,174:78,175:80,176:75,177:63}
 return sorted(out,key=lambda x:(x['page'],x['column'],x['ordinal']))

def normalize_gloss(s):
 return re.sub(r'[^a-z0-9]','',s.casefold())

def extend(rows,audit):
 lookup={}
 for r in rows: lookup.setdefault((r[2],normalize_gloss(r[3])),[]).append(r)
 for item in inventory():
  page,col,n=item['page'],item['column'],item['ordinal']
  key=f'{SOURCE}:kor-english:p{page}:{col}:item{n:02}'
  entry=dict(entry_key=key,printed_page=page,pdf_page=page+19,column=col,item=n,section='Kor–English',printed_response=item['form'],gloss=item['gloss'],selected_forms=[],reused_keys=[],status='selected',note=item['review_note'])
  hold={(174,'right',24):'No printed gloss; meaning unresolved.',(174,'right',35):'Printed he conflicts with forward head; probable compositor truncation.',(175,'left',3):'Printed grain conflicts with forward gram; unresolved source gloss.',(173,'left',38):'Segmentation of negative expression remains uncertain.'}.get((page,col,n))
  if hold:
   entry.update(status='held',note=hold);audit.append(entry);continue
  forms=item['form'].split('|')
  for variant,form in enumerate(forms,1):
   form=form.strip()
   tag=DIALECT
   explicit_pos={(175,'left',34):'noun',(175,'left',37):'verb',(176,'right',1):'verb',(176,'right',3):'noun',(177,'right',11):'noun'}.get((page,col,n))
   if explicit_pos: tag+=' '+explicit_pos
   note=''
   if (page,col,n)==(177,'right',30): note='Source explicitly marks final n as nasal; transcribed ñ.'
   if (page,col,n)==(177,'right',2):
    tag+=' uncertain';entry['review_reason']='gloss: reverse meek versus forward neck; source sense retained without emendation'
   matches=lookup.get((form.casefold(),normalize_gloss(item['gloss'])),[])
   citation=f'{SOURCE}[p. {page}, Kor–English, {col} column, item {n}]'
   if len(matches)==1:
    matches[0][7]+=';'+citation
    if explicit_pos and explicit_pos not in matches[0][14].split(): matches[0][14]+=' '+explicit_pos
    entry['reused_keys'].append(matches[0][10])
   else:
    fkey=key if len(forms)==1 else f'{key}:alt{variant}'
    rows.append(['ko','',form.casefold(),item['gloss'],'','',note,citation,'','',fkey,'','','',tag])
    entry['selected_forms'].append(form)
  if not entry['selected_forms']: entry['status']='reused'
  elif entry['reused_keys']: entry['status']='selected_and_reused'
  audit.append(entry)
 assert len({r[10] for r in rows})==len(rows)
 return rows,audit
