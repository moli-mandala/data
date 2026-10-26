"""Whole Chapter 8 native-text inventory; no canonical installation."""
import json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
SURFACES={
222:[(36,49)],223:[(4,10),(13,20),(23,33),(35,40),(43,49),(51,60)],
224:[(14,22),(24,33),(41,51)],225:[(4,16),(25,[43,44]),(47,58),(67,81)],
226:[(4,18),(28,39),(42,[56,57]),(66,75)],227:[(4,15),(26,35),(45,55),(57,67)],
228:[(10,21),(24,39),(48,[64,65])],229:[(4,16),(26,32),(34,40),(42,56)],
230:[(8,15),(17,25),(35,45),(47,[63,64]),(66,77)],
231:[(4,10),(16,27),(30,38),(41,49)],232:[(12,20),(22,[32,33]),(35,45),(47,57)],
234:[(7,14),(16,26),(29,35),(37,42),(44,52)],235:[(9,18),(20,30),(32,43)],
}
LISTS={
231:[(58,60),(61,63),(64,65),(66,68),(70,72)],
232:[(3,5),(6,8),(66,67),(68,69),(70,70),(71,72)],
233:[(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(33,34),(35,36),(37,38),(39,40),(41,42),(43,44),(45,46),(47,48),(49,49),(50,51),(52,53),(54,55)],
}
PROSE=[
(222,[29,30,31],'-ei','','clitic emph','Emphatic/focus clitic marks intensity and attaches to numerals, adverbs, nouns and demonstratives.'),
(224,[6,7,8,9,10,11],'re','','particle voc masc','Used with male individual addressees, optional, with overtones of emergency and a pause after the expression.'),
(224,[6,7,8,9,10,11],'ri','','particle voc fem','Used with female individual addressees, optional, with overtones of emergency and a pause after the expression.'),
(224,[35,36,37,38],'e','hey','particle voc','Attention-calling, clause-initial before kin terms, pronouns or proper names.'),
(224,[59],'e','','particle','Footnote explicitly identifies a homophonous surprise particle; do not merge its function with vocative e.'),
(225,[18,20,21,22],'pe','','particle','Contradicts something already said or implies unlikelihood; follows its referent.'),
(225,[20],'pei','','particle','Explicitly described as another realization of pe.'),
(225,[60,62,63,64],'se','you think?','particle','Polite question anticipates negative response, or negative question anticipates affirmative response. Attributed functional explanation cites Schmidt.'),
(225,[85],'rə','','particle','Nepali comparison only; not a Darai attestation.'),
(226,[21,23,24,25],'dze','you must realize that…','particle','Confirmation particle normally follows the verb; information should be known to the speaker according to the source.'),
(226,[59,61,62,63],'tou','okay','particle','Normally clause-final with questions, softening the question.'),
(226,[61],'toũ','okay','particle','Explicitly described as alternative pronunciation of tou.'),
(227,[18,20,21,22,23],'təre','','particle','Imperative addressee equal or lower status: appropriate for sons, brothers or wife, inappropriate for sister-in-law or honorific addressee.'),
(227,[39,40,41,42],'haĩ','okay','particle','Softens requests or turns statements into requests; often clause-final with requests or hortatives. Within the source-designated Nepali-origin particle group.'),
(228,[3,5,6,7],'tə','as for…','particle','Contrastive referent marking; within the source-designated Nepali-origin particle group.'),
(228,[41,43,44,45],'cahĩ','','particle emph','Focuses competing nominal referents; within the source-designated Nepali-origin particle group.'),
(229,[17,18],'cahĩ','as for…','particle emph','Its absence is described as straightforward meaning without focus.'),
(229,[20,22,23],'kʰoi',"I don't know",'particle','Doubt or hesitancy; within the source-designated Nepali-origin particle group.'),
(230,[3,4,5],'kere','I suppose; I guess','particle','Uncertainty or ignorance, always clause-final according to the source; Nepali-origin particle group.'),
(230,[27,29,30,31,32],'lə','okay','particle','Initiates something or suggests moving on, with hortative and other clauses; Nepali-origin particle group.'),
(231,[11,12],'lu','okay','particle','Author explicitly identifies same function and a variant of lə.'),
(231,[12],'lə','','particle','Explicit comparator for lu with same initiation function.'),
(231,[13],'lure','','particle','Related initiator; no automatically inferred equivalence to bare lə.'),
(231,[13],'luri','','particle','Related initiator; no automatically inferred equivalence to bare lə.'),
(233,[27,28],'ɡuruŋ ɡuruŋ','the beatings of drums','onomatopoeia','Source explicitly contrasts sound imitation with non-sound manner words.'),
(233,[29,30],'curlummə','','','Describes the manner a person dives into water without producing sound, according to the source.'),
(234,[62],'s-','','prefix','Identifiable echo-forming initial element; source says echoed part generally begins with s-.'),
(234,[63],'kam','work','',''),(234,[63],'kam sam','work and so on','','Source gives imprecise meaning work etc.; echoed part is meaningless in isolation (continued on page 235).'),
]
def build():
 pages={p['printed_page']:p for p in map(json.loads,(P/'whole-native-pages.jsonl').open()) if p['printed_page'] and 222<=p['printed_page']<=235}
 def txt(n,i):return ''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip()
 def norm(t):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',t).strip())
 def ev(n,ls):return [pages[n]['lines'][i-1] for i in ls]
 def row(n,key,kind,form,gloss,ls,**extra):return dict(source_unit_key=f'dhakal2011:p{n}:ch8:{key}',printed_page=n,pdf_page=n+22,kind=kind,form=norm(form),gloss=norm(gloss),source_lines=ls,evidence=ev(n,ls),status='Native-text first reading; original-page review pending',**extra)
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
 rows=build();(P/'chapter8-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print(len(rows),'groups; all pages 222–235 read; original review and integration pending')
