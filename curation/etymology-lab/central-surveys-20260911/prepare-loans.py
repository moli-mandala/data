import json
from research_helpers import Batch,P
b=Batch(11)
def loan(l,g,w,parent,key,url,evidence,senses=None,extra_cite=''):
 allowed=[x for x in dict.fromkeys(r['Gloss'] for r in b.inv[l]) if set(x.split('; '))<=set(senses or [g])]
 b.add(l,g,w,parent,evidence,tier='qualified',kind='borrowed',source_glosses=allowed,citation=f'platts1884[s.v. {key}]'+extra_cite,source_url=url)
for l,w in [('Malvi','badan'),('Nimadi','bādān'),('Bagheli','beḍen')]:
 loan(l,'body',w,'f_7tuydfvqt7ici','badan','https://www.rekhta.org/urdudictionary?keyword=%D8%A8%D8%AF%D9%86','Platts distinguishes Arabic-origin badan ‘body’ in Hindustani from its Sanskrit-derived mouth/face homonym. Propose the existing Hindustani body head as immediate donor; the survey vowel/retroflex adaptations and precise contact route remain for review.')
for l,w in [('Malvi','ādmi|admi|adimi'),('Nimadi','ādmi|admi|ādəmi'),('Bagheli','aḍmi')]:
 loan(l,'man/husband',w,'f_gsynajbspnc2y','admi','https://www.rekhta.org/urdudictionary?keyword=aadmii','Platts’s Hindustani ādmī entry includes both a human/man and a spouse, so the grouped elicited senses are compatible. Propose borrowing of that whole Hindustani lexeme; the source does not independently date or identify the exact transmission route.',senses=['man','husband'])
for l,w in [('Malvi','orat'),('Nimadi','orāt'),('Bagheli','eureṭ|aureṭ')]:
 loan(l,'woman/wife',w,'f_vhi3eao4uvyjm','aurat','https://www.rekhta.org/urdudictionary?keyword=aurat','Platts explicitly places the woman/wife senses in Urdu usage of ʻaurat, separate from the earlier Arabic sense. This supports the Hindustani lexeme as proposed immediate donor; contraction, vowels and final ṭ are local adaptations still needing review.',senses=['woman','wife'])
for l in ['Nimadi','Bagheli']:
 loan(l,'fingernail','nakhun','f_bgpkwqir4pwla','nakhun','https://www.rekhta.org/urdudictionary?keyword=naakhun','Platts attests Hindustani nāḵẖun for finger/toe nails and claws, marking Persian origin. The complete -un form supports this donor rather than a bare nakha link; the x-to-kh adaptation is plausible, while the precise intermediate contact history remains open.')
for l,w in [('Malvi','jaban|jabān'),('Nimadi','jābān')]:
 loan(l,'tongue',w,'f_zqbwnodbqwsho','zaban','https://www.rekhta.org/urdudictionary?keyword=zabaan','Platts’s zabān/zubān includes the physical tongue as well as language; the existing Hindi-Urdu head’s gloss ‘language’ is therefore compatible. Initial j for z is a proposed local adaptation. The unrelated Arabic jabān ‘coward’ is excluded.')
for l,w in [('Malvi','arvājā'),('Nimadi','darvājā|darvājo'),('Bagheli','ḍevaja')]:
 loan(l,'door',w,'f_prmeiidvi4gto','darwaza','https://www.rekhta.org/urdudictionary?keyword=darvaaza','Platts attests the complete Hindustani darwāza ‘door’, marked Persian in origin. The w/v and z/j adaptation fits this family; missing initial d or medial r in reduced responses needs source/dialect confirmation. No compound decomposition into remote Persian roots is proposed.')
loan('Bagheli','whole','sebuṭ|sabuṭ','f_dbuiao2qkdgv4','sabut','https://www.rekhta.org/urdudictionary?keyword=saabit','Platts explicitly gives an adjectival ‘entire’ sense for s̤ubūt, also s̤abūt. Use the existing whole/entire Hindustani adjective, not the homophonous proof noun; the source’s retroflex final stop remains qualified.')
for l in ['Malvi','Nimadi','Bagheli']:
 loan(l,'heart','dil','f_snjuyk7aztjyu','dil','https://www.rekhta.org/urdudictionary?keyword=dil','Platts identifies the heart noun dil as Persian-origin Hindustani, distinct from local dil ‘village mound’ and ḍīl ‘body/build’. The existing Hindi control attestation anchors a proposed Hindustani donor; it does not establish that control locality as the historical source.',extra_cite=';kannauji[p. 59]')
for l,w in [('Malvi','khun|kun'),('Nimadi','khun|khuṇ'),('Bagheli','khūn|khun')]:
 loan(l,'blood',w,'f_nomhmgwrljcl4','khun','https://www.rekhta.org/urdudictionary?keyword=khuun','Platts’s Persian-origin ḵẖūn denotes blood in Hindustani; the existing Hindi khun control supplies the adapted form. Propose borrowing through Hindustani, retaining aspiration/nasal-place variation and the exact intermediate route as uncertainties.',extra_cite=';kannauji[p. 60]')
for l,w in [('Malvi','maino|maina|mino|miniya'),('Nimadi','maino|mahino|mainu|maina|mayino|māino|māhino'),('Bagheli','mehina|mehinna|mehiṇa|mehīna')]:
 loan(l,'month',w,'f_n4lkgn2ws3cvw','mahina','https://www.rekhta.org/urdudictionary?keyword=mahiina','Platts attests mahīnā as a month noun in Hindustani; the existing Hindi control provides the proposed donor anchor. The n-bearing extension distinguishes this family from a bare māh/māsa comparison, but h-loss, contraction and the survey endings need local review.',extra_cite=';kannauji[p. 85]')
for l,w in [('Malvi','garam'),('Nimadi','garam'),('Bagheli','gerem|geṛəm')]:
 loan(l,'hot',w,'f_cdko5w5xxg5qg','garm','https://www.rekhta.org/urdudictionary?keyword=garm','Platts explicitly gives garam as a Hindustani variant of Persian-origin garm ‘hot’; the inserted vowel thus already exists in the proposed immediate donor. The Hindi control attests this hot adjective, while the precise borrowing route and Bagheli vowel/r variation remain qualified.',extra_cite=';kannauji[p. 88]')
for l,w in [('Malvi','haphed'),('Bagheli','səfeḍ|səpəḍ|səpəṭ|sapeth|supeḍ')]:
 loan(l,'white',w,'f_lwa4hsrbk5gee','safed','https://www.rekhta.org/urdudictionary?keyword=safed','Platts records safed and sufed variants as a Persian-origin Hindustani white adjective. The existing Hindi control anchors the donor; the survey’s ph/p for f, h for s, vowels and final stop adaptations still need local confirmation.',extra_cite=';kannauji[p. 91]')
b.save()
# Reopened held records are now pending; retain their former reasons in the manifest history.
h=json.loads((P/'holds.json').read_text())
for pp in b.proposals.values():
 for x in pp:
  reopened={i:h.pop(i) for i in x['formIds'] if i in h}
  if reopened:x['reopenedHolds']=reopened
(P/'holds.json').write_text(json.dumps(h,ensure_ascii=False,indent=2))
# Save only the newly added provenance notes in this still-pending batch.
for lang,lid in {'Malvi':'mewari_basad','Nimadi':'Nimadi','Bagheli':'bagheli_lakshman'}.items():
 f=P.parent/lid/'batch-011.json';d=json.loads(f.read_text());d['proposals']=b.proposals[lang];f.write_text(json.dumps(d,ensure_ascii=False,indent=2))
