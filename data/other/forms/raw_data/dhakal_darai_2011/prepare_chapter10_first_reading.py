"""Whole Chapter 10 native-text inventory; no canonical installation."""
import json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
SURFACES={
249:[(20,28),(31,38),(44,53)],250:[(5,12),(17,28),(31,41),(48,58),(64,71),(74,82)],
251:[(12,19),(22,32),(39,49),(52,59)],252:[(29,37),(39,49),(54,62)],
253:[(5,14),(19,26),(28,38),(43,54),(57,65),(68,79),(83,93)],
254:[(5,12),(17,28),(30,39),(44,56)],255:[(9,20),(40,52),(54,66)],
256:[(13,25),(28,35),(37,51),(61,70)],257:[(4,13),(16,28),(40,52),(55,65),(67,76)],
258:[(35,44),(46,55),(58,70)],259:[(4,12),(26,36),(38,50),(52,65),(68,77)],
260:[(4,14),(17,23),(25,32),(47,55)],
261:[(5,[19,20]),(23,30),(32,38),(41,[60,61])],262:[(14,29),(31,45),(48,57),(59,70),(77,87)],
263:[(5,16),(18,28),(30,41),(43,53),(59,[73,74]),(76,87)],
264:[(4,12),(14,23),(27,36),(38,46),(54,66),(68,83)],
265:[(5,24),(26,41),(53,65),(68,81),(83,97)],
266:[(8,18),(20,32),(35,47),(49,61),(68,83),(85,100)],
267:[(4,25),(34,46),(48,53),(56,67),(69,80),(83,92)],
268:[(5,18),(20,28),(44,48),(50,54)],269:[(5,15),(17,27),(34,42),(44,52),(58,66),(68,76)],
270:[(5,12),(14,21),(24,31),(33,40),(46,55),(57,62),(64,70)],
271:[(5,11),(16,22),(24,30),(35,48),(50,62)],
272:[(5,14),(16,27),(29,41),(46,[61,62]),(64,73),(76,85),(87,95)],
273:[(4,10),(16,27),(29,[38,39]),(42,49),(51,58)],
274:[(5,12),(14,22),(25,32),(34,42),(44,51),(53,58)],275:[(42,48),(50,57)],
276:[(4,11),(14,24),(34,51),(53,60),(64,73)],277:[(5,14)],
278:[(28,35),(37,45),(48,54),(56,63),(66,72)],
279:[(4,13),(16,22),(24,29),(46,54)],280:[(4,16),(19,30),(32,37),(39,52)],
281:[(5,11),(13,18),(21,27),(29,34),(37,43),(45,48),(51,57),(59,62)],
282:[(5,10),(12,17),(31,39)],283:[(5,[17,18]),(20,[37,38]),(40,47),(50,58),(60,[72,73])],
285:[(7,20),(22,36)],
}
LISTS={
277:[(33,34),(35,36),(39,40),(41,42),(43,44),(45,46),(47,47),(48,49),(51,52),(54,55),(57,58),(59,60),(61,62),(63,64)],
278:[(4,5),(7,8),(9,10),(11,12),(14,15),(16,17),(18,19),(20,21)],
}
PROSE=[
(250,[42],'radzake','king-DAT','','Source labels this the indirect object.'),
(250,[43],'roṭi','bread','','Source labels this the direct object.'),
(255,[3,4],'həkʰə-','','cop','Attributive and nominal copula; inflects for agreement and tense.'),
(255,[3,4],'aṭə-','','cop','Locational and possessive copula.'),
(255,[4],'rəhə-','','cop pret','Source calls it the suppletive past form of the two copulas.'),
(255,[27,28,29],'baṭə-','','cop','Source explicitly identifies free variation with aṭə- in locative and possessive clauses.'),
(255,[28],'aṭə-','','cop','Source explicitly identifies free variation with baṭə-.'),
(255,[29],'rəhə','','cop pret','Printed without hyphen in the suppletive-form discussion; retain literal spelling.'),
(255,[36],'hame','we','','Copular subject in source example.'),
(255,[36],'həkʰəi','','cop','Copula in the source example; no supplied person-number gloss in this prose.'),
(255,[37],'car bʰai','four brothers','','Copular complement in source example.'),
(256,[52],'pʰirli','','proper-noun','Source identifies Phirli as copular subject.'),
(256,[52],'həkʰəi','','cop','Source identifies this as copula.'),
(256,[52],'əlɡə','tall','adj','Source identifies this as adjective.'),
(257,[29],'dzoɡi','hermit','','Source identifies this as copular subject.'),
(257,[29],'baṭəi','','cop','Source identifies this as copula.'),
(257,[30],'ucʰin','right there','adv','Source identifies this as adverb.'),
(257,[36],'baṭə-','','cop','Source states possessive-clause use, with aṭə- alternative.'),
(257,[36],'aṭə-','','cop','Source states possessive-clause use, with baṭə- alternative.'),
(258,[6],'həkʰə-','','cop','Table 10.2 marks predicate nominal/adjective use, but not possessive/locative use.'),
(258,[7],'aṭə-','','cop','Table 10.2 marks possessive/locative use, but not predicate nominal/adjective use.'),
(258,[7],'baṭə-','','cop','Table 10.2 marks possessive/locative use, but not predicate nominal/adjective use.'),
(264,[47,48,49],'kiti kiti','what what','interr','Reduplicated question presupposes plural answer or a list of items; source extends process to person, place and manner.'),
(265,[47,48],'tou','','particle interr','Optional clause-final interrogative particle with rising intonation; constituent order unchanged.'),
(266,[4],'hə̃','yes','','Positive response alternative.'),
(266,[4],'aṭə-','yes','cop','Explicit positive-response alternative in disjunctive-question discussion; preserve bound notation.'),
(266,[4],'nidzə','no','neg','Negative response alternative.'),
(266,[63,64],'ki','or','conj','Disjunctive coordinator placed to the right of the statement in alternative questions.'),
(267,[28,30],'hə̃','yes','','Positive particle agrees with proposition.'),
(267,[29,31],'nidzə','no','neg','Negative particle used with negated sentences.'),
(268,[32,33,58,59],'nidzə','','neg','Independent negative particle; nihidzə is the slow, careful pronunciation. Preverbal across tenses/aspects/moods according to introduction.'),
(268,[33],'nihidzə','','neg','Explicit slow, careful pronunciation of nidzə.'),
(268,[32,38,39,40,41],'nai-','','prefix neg','Confined to non-past in this discussion; prefixation deletes tense marker, leaving person marker.'),
(268,[38],'na-','','prefix neg','Explicit alternative realization of nai-.'),
(268,[32],'dzʰun-','','prefix neg','Named prohibitive marker; later restricted to imperative and optative.'),
(269,[31,53],'nidzə','not','neg','Source calls it a prefix here, used in past tense and all aspects, contrasting its independent-particle analysis elsewhere.'),
(270,[41,42,43],'dzʰun-','','prefix neg','Prohibitive confined to imperative and optative sentences.'),
(271,[12],'nidzə','','neg','Desiderative mood negation.'),
(271,[31],'nai-','','prefix neg','Source states real conditional negation uses nai-.'),
(271,[31],'nidzə','','neg','Source states hypothetical conditional negation uses nidzə.'),
(271,[63,64,65],'nidzə','not','neg','All copulas use this independent particle, with copula optionally absent.'),
(271,[69],'dzin-','','','Bote comparison attributed to Poudel 1999:78; not Darai.'),
(271,[70],'dzun-','','','Majhi comparison attributed to Dhakal 2008:114; not Darai.'),
(271,[70],'dzun-','','','Danuwar comparison attributed to Timilsina 2009:83; not Darai.'),
(273,[11],'nai-','','prefix neg','Negates sequential converbs and infinitives.'),
(273,[12],'nidzə','','neg','Negates prospective forms.'),
(273,[59,60,61,62,63,64],'-be','','suffix emph neg','Emphatic negative construction suffix; collocates with following negative markers and occurs in prohibitive emphatic sentences.'),
(273,[62],'nai-','','prefix neg','Negative marker in emphatic -be construction.'),
(273,[62],'nidzə','','neg','Negative marker in emphatic -be construction.'),
(277,[26,27,28,29,30],'-a','','suffix caus','Productive morphological causative increases valence; some verbs show vowel harmony.'),
(277,[26,30],'-wa','','suffix caus','Source causative suffix alternative; detailed glide insertion analysis follows.'),
(278,[23],'-a','','suffix caus','Source states glide w is inserted when causative -a follows a verb stem ending in a; final stem vowel and glide are metalinguistic segments, not extra lexical attestations.'),
(279,[30],'cər','graze','verb intr','Source explicitly contrasts this intransitive verb root with causative car.'),
(279,[31,32],'car','cause to graze','verb caus','Explicit causative root, no inferred form from paradigm.'),
(279,[32,33],'ɡokʰra','cause to bathe','verb caus','Explicit source causative root.'),
(279,[40,41,42,43],'laɡaike','to apply/attach','verb','Source calls this a vector verb used in an attested but infrequent periphrastic causative construction.'),
(280,[59,60,61],'-inuhuk','','suffix pass','Source morphological passive restricted to past tense, not usable for non-past passive.'),
(282,[21,22,23],'-inuhuk','','suffix pass','Suffix can passivize even an intransitive verb in past tense; non-past example 83b is explicitly unacceptable.'),
(285,[44,45,46],'həkʰə-','','cop','Summary: nominal and adjectival predicates.'),
(285,[44,46],'aṭə-','','cop','Summary: locative and possessive predicates.'),
(285,[45],'rəhə-','','cop pret','Summary: suppletive past copula.'),
]
def build():
 pages={p['printed_page']:p for p in map(json.loads,(P/'whole-native-pages.jsonl').open()) if p['printed_page'] and 249<=p['printed_page']<=285}
 def txt(n,i):return ''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip()
 def norm(t):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',t).strip())
 def ev(n,ls):return [pages[n]['lines'][i-1] for i in ls]
 def row(n,key,kind,form,gloss,ls,**extra):return dict(source_unit_key=f'dhakal2011:p{n}:ch10:{key}',printed_page=n,pdf_page=n+22,kind=kind,form=norm(form),gloss=norm(gloss),source_lines=ls,evidence=ev(n,ls),status='Native-text first reading; original-page review pending',**extra)
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
   category='verb base or causative counterpart'
   out.append(row(n,f'list:l{f}','listed-head-group',head,gloss,sorted(set([f,g])),source_category=category))
 seen={}
 for n,ls,form,gloss,tags,note in PROSE:
  k=(n,ls[0]);seen[k]=seen.get(k,0)+1
  extra={'tags':tags.split(),'note':note}
  if n==271 and ls in ([69],[70]):
   extra['disposition']='foreign-comparison-control';extra['source_language']='Bote' if ls==[69] else ('Majhi' if 'Majhi' in note else 'Danuwar')
  out.append(row(n,f'prose:l{ls[0]}:a{seen[k]}','prose-attestation',form,gloss,ls,**extra))
 for record in out:
  if record['source_unit_key']=='dhakal2011:p282:ch10:surface:l12':
   record['disposition']='unacceptable-source-control'
   record['note']='Source marks example 83b with an asterisk as unacceptable non-past passive.'
 assert len(out)==len({r['source_unit_key'] for r in out})
 return out
if __name__=='__main__':
 rows=build();(P/'chapter10-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print(len(rows),'groups; all pages 249–285 read; original review and integration pending')
