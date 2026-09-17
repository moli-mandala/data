from research_helpers import Batch
b=Batch(21)
def a(l,g,w,p,e,c,u):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e+' The existing Hindi record anchors the proposed immediate donor form; it does not establish a particular donor village or exclude other regional mediation.',tier='qualified',kind='borrowed',citation=c,source_url=u)
for l,w in [('Malvi','asmān|āsmān|asman|āśmān'),('Nimadi','āsmān'),('Bagheli','asman')]:
 a(l,'sky',w,'f_alir5ifyzfzvq','Platts āsmān/asmān explicitly means sky and is Persian in origin. The Hindi control asman (Kannauji survey p.64) supports the Hindustani stage; source s/ś variation is retained.','platts1884[s.v. āsmān];kannauji[p. 64]','https://www.rekhta.org/urdudictionary?keyword=%D8%A2%D8%B3%D9%85%D8%A7%D9%86')
for l,w in [('Malvi','hava|havā'),('Nimadi','hava|hāvā|havā|havai'),('Bagheli','heva')]:
 a(l,'wind',w,'f_ra6uvjvpsrlbe','Platts hawā includes air/wind and identifies the Persian-mediated Arabic word. Hindi hava (Kannauji survey p.66) is the immediate comparison; Bagheli e and Nimadi final i need local review.','platts1884[s.v. hawā];kannauji[p. 66]','https://www.rekhta.org/urdudictionary?keyword=havaa')
for l,w in [('Malvi','rasto|rasta|rastā'),('Nimadi','rasto|rāsto'),('Bagheli','rasta')]:
 a(l,'path',w,'f_44kuz3f4pxo5e','Platts explicitly distinguishes Hindustani rāstā ‘road, path’ from its Persian rāsta source. The -o survey forms may reflect regional gender adaptation; they are not direct Sanskrit reflexes.','platts1884[s.v. rāstā];kannauji[p. 67]','https://www.rekhta.org/urdudictionary?keyword=raasta')
a('Bagheli','onion','piyaj|piyaju|pyaj','f_22utxbhjyd5lu','Platts piyāz means onion/leek; the Hindi survey control pyaj (p.73) documents the affricate form. The Bagheli expanded vowel and final-u form are compatible adaptations, with exact local morphology open.','platts1884[s.v. piyāz];kannauji[p. 73]','https://www.rekhta.org/urdudictionary?keyword=pyaaz')
for l,w in [('Malvi','sal|sāl'),('Nimadi','sāl'),('Bagheli','sal')]:
 a(l,'year',w,'f_thgcjhj7vr5i2','The full Platts Persian sāl entry explicitly means year, distinct from its thorn, house and tree homonyms. Hindi sal (Kannauji survey p.85) provides the immediate Hindustani comparison.','platts1884[s.v. sāl, Persian year];kannauji[p. 85]','https://urdu.hawramani.com/%D8%B3%D8%A7%D9%84-4/')
for l,w in [('Malvi','barābar|bārābar'),('Nimadi','barabar'),('Bagheli','beraber')]:
 a(l,'same',w,'f_xitegx7mbquyi','Platts barābar expressly includes alike/the same. The installed Hindi donor baraabar is also cited by Liljegren (LX000263); the borrowing is proposed as a whole word, rather than inventing local bar+bar reduplication.','platts1884[s.v. barābar];liljegren[entry LX000263]','https://www.rekhta.org/urdudictionary?keyword=baraabar')
a('Nimadi','few','kam','f_ymjgts524mota','Platts kam is the Persian-origin little/less adjective, distinct from kām ‘work’. The installed Hindi kam donor is supported by Liljegren’s donor comparison, while the count-sense few is compatible.','platts1884[s.v. kam];liljegren[entry kam]','https://www.rekhta.org/urdudictionary?keyword=kam')
b.save()
