from research_helpers import Batch
b=Batch(26)
def a(l,g,w,p,e,senses=None):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Form'] in w.split('|') and (set(r['Gloss'].split('; '))<=set(senses) if senses else r['Gloss']==g)]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier='qualified',source_glosses=list(dict.fromkeys(r['Gloss'] for r in rs)))
for l,g,w in [('Malvi','husband','gharvālo|gharāḷo|gharvaḷa|gharvala|gharvara'),('Malvi','wife','ghervāli|gharāḷi|gharvari|gharvali'),('Nimadi','husband','gharvāḷo'),('Nimadi','wife','gharvaḷi|gharvāḷai'),('Bagheli','husband','gheruala|ghervala'),('Bagheli','wife','gheruali')]:
 a(l,g,w,'4435','CDIAL 4435 *gharapāla compares Sindhi gharavāro husband and Hindi gharwālā householder/husband, with feminine gharwālī wife. Both gender forms fit this historical compound; local l/ḷ/r, loss of v and vowel contraction require review, and productive -vālā formation or regional diffusion may have reinforced it.')
a('Bagheli','woman; wife','meheṛiya|meheriye|meheṛiye|meheriya','9962','CDIAL 9962 mahilā gives Maithili mehar and Hindi mahar/maharī/mehar/mihariyā woman/wife. Turner discusses competing l/ḍ histories and mahī/mahiṣī connections; the -iyā forms fit the cited family, but e-vocalism and rhotic variants do not establish a unique deep derivation.',senses=['woman','wife'])
a('Bagheli','woman; wife','meheraru|meheṛaru|meheṛaṛu|mehəṛaru','9963','CDIAL 9963 *mahilārūpa specifically gives Bihari mehrārū and Maithili/Bhojpuri meharārū woman/wife, unlike the shorter mahilā family. The full -rāru extension selects this compound; Bagheli retroflex and vowel variants remain qualified, together with the uncertainty in the first member’s history.',senses=['woman','wife'])
b.save()
