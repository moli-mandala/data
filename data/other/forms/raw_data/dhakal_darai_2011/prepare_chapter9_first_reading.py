"""Whole Chapter 9 native-text inventory; no canonical installation."""
import json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
SURFACES={
236:[(29,37),(40,61)],237:[(4,[21,22]),(27,36),(38,47),(50,59),(62,68),(71,84)],
239:[(11,17),(20,32),(35,45),(48,54),(57,70),(73,80)],
240:[(5,11),(14,20),(27,[41,42]),(45,54),(60,65)],
241:[(10,16),(19,27),(29,41),(44,51),(54,59)],
242:[(10,16),(19,29),(32,44),(47,55),(58,72)],
243:[(5,12),(15,23),(26,33),(49,61)],
244:[(8,[28,29]),(35,43),(46,55),(58,63)],
245:[(29,46),(48,61),(64,75)],246:[(4,19),(32,46),(49,55)],
247:[(6,13),(16,24),(26,34),(36,43),(45,50),(52,60),(63,70)],248:[(15,31)],
}
LISTS={}
PROSE=[
(241,[61],'macʰəkə','of fish','','Source genitive phrase modifies farming.'),
(241,[63],'-kə','','suffix gen','Source genitive marker used with nouns; explicitly contrasted with pronoun marker -rə.'),
(241,[63],'-rə','','suffix gen','Source genitive marker used with pronouns; explicitly contrasted with noun marker -kə.'),
(246,[20],'ijə','this','','Source substitution commentary names ijə, while the cited example prints i; preserve both attestations without harmonization.'),
]
def build():
 pages={p['printed_page']:p for p in map(json.loads,(P/'whole-native-pages.jsonl').open()) if p['printed_page'] and 236<=p['printed_page']<=248}
 def txt(n,i):return ''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip()
 def norm(t):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',t).strip())
 def ev(n,ls):return [pages[n]['lines'][i-1] for i in ls]
 def row(n,key,kind,form,gloss,ls,**extra):return dict(source_unit_key=f'dhakal2011:p{n}:ch9:{key}',printed_page=n,pdf_page=n+22,kind=kind,form=norm(form),gloss=norm(gloss),source_lines=ls,evidence=ev(n,ls),status='Native-text first reading; original-page review pending',**extra)
 out=[]
 for n,pairs in SURFACES.items():
  for f,g in pairs:
   fs=f if isinstance(f,list) else [f];gs=g if isinstance(g,list) else [g];raw=' '.join(txt(n,i) for i in gs)
   ids=re.findall(r'\(([A-Za-z][A-Za-z.]+\.\d+)\)',raw)
   gloss=re.sub(r'\s*\([A-Za-z][A-Za-z.]+\.\d+\)','',raw).strip('‘’\' ')
   out.append(row(n,f'surface:l{fs[0]}','whole-expression',' '.join(txt(n,i) for i in fs),gloss,fs+gs,surface_lines=fs,translation_lines=gs,raw_translation=raw,text_ids=ids,source_surface_evidence=ev(n,fs),source_translation_evidence=ev(n,gs),interlinear_evidence=pages[n]['lines'][max(fs):min(gs)-1]))
 for n,pairs in LISTS.items():
  for f,g in pairs:
   head=txt(n,f);raw=txt(n,g)
   if f==g:head=re.split('[‘\']',head,maxsplit=1)[0].strip()
   m=re.search('[‘\'](.*)',raw);gloss=m.group(1).strip('’\' ') if m else raw
   category='interjection' if n==231 or n==232 and f<10 else 'onomatopoeia' if n==232 or f<27 else 'non-sound manner word'
   out.append(row(n,f'list:l{f}','listed-head-group',head,gloss,sorted(set([f,g])),source_category=category))
 seen={}
 for n,ls,form,gloss,tags,note in PROSE:
  k=(n,ls[0]);seen[k]=seen.get(k,0)+1
  extra={'tags':tags.split(),'note':note}
  if n==225 and ls==[85]:extra['disposition']='foreign-comparison-control';extra['source_language']='Nepali'
  out.append(row(n,f'prose:l{ls[0]}:a{seen[k]}','prose-attestation',form,gloss,ls,**extra))
 assert len(out)==len({r['source_unit_key'] for r in out})
 return out
if __name__=='__main__':
 rows=build();(P/'chapter9-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print(len(rows),'groups; all pages 236–248 read; original review and integration pending')
