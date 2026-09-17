import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0]);qs=[]
def add(h,p,g,w,entry,query,file,ev):
 primary=next(x for x in json.loads((P/file).read_text()) if x['word']==query)
 qs.append(dict(parent=h,persian=p,gloss=g.split('|'),words=w.split('|'),citation='platts1884[s.v. '+entry+']',evidence=ev,primary=primary))
add('f_skjvetixg4pnq','f_pa2wkl7fezmhg','red','lal','lāl','lāl','platts-additional-research.json','Platts p. 946 explicitly distinguishes Persian/Hindi lāl “red” from the unrelated affectionate/nursery and saliva homonyms. Only the color response belongs to this loan family.')
add('f_xqxbqhxafwpsu','f_f5yhi6kl6jazu','knife','caku|cakku|cako|cakua|cakhu|cokku','ćāqū','ćāqū','platts-research.json','Platts p. 418 identifies ćāqū “clasp knife, penknife” as Persian. Hindi caku and survey caku/cakku show q adapted as k, with vowel and gemination variation; uncertain intra-IA transfer is retained.')
add('f_ra6uvjvpsrlbe','f_rlqlstrekfzvc','wind|air','hava|hawa|havo|havao','hawā','hawā','platts-final-research.json','Platts p. 1240 gives hawā “air, wind” through Persian from Arabic. The wind/air sense selects this loan family, distinct from the separately listed Arabic desire homonym.')
add('f_44kuz3f4pxo5e','f_74vcu23lscgak','path|road','rasta|rasto|rastha|rast a','rāstā','rāstā','platts-final-research.json','Platts p. 581 expressly labels rāstā “road, way, path” as the Hindi adaptation of Persian rāsta. Final a/o variation is ordinary regional noun adaptation.')
add('f_cdzupphnjmodm','f_edny5jsgvz5fk','fat','carbi|corbi|cerbi|caṛbi|carbhi','ćarbī','ćarbī','platts-additional-research.json','Platts p. 429 gives Persian ćarbī “fat, grease, suet, tallow”. The survey carbi/cerbi/corbi material-fat responses preserve its distinctive consonant skeleton; this does not cover unrelated adjectives meaning stout.')
add('f_n4lkgn2ws3cvw','f_vbtodh3gxxoq2','month','mahina|mahino|mayino|maino|maina|mainah|mehino','mahīnā','mahīnā','platts-additional-research.json','Platts p. 1103 labels mahīnā “month” Persian/Hindi and analyzes mah/māh plus -īna. The explicit h-bearing and transparent h-loss variants are kept together; highly contracted mina/mena need separate local evidence.')
add('f_tybgs7a4lnet2','f_4txbqir4uazlw','bad','kharab|xarab|kharob|kharap','ḵẖarāb','ḵẖarāb','platts-additional-research.json','Platts p. 488 documents Arabic-derived kharāb “ruined, spoiled, bad” in Urdu/Hindi. The survey kharab family preserves its consonants, with voiced/voiceless final adaptation; the remote Arabic origin does not by itself identify each local donor.')
add('f_cdko5w5xxg5qg','f_khzburdg3bkvo','hot|hot (water)','garm|garam|garma|garom|gorom','garm','گرم','platts-final-research.json','Platts p. 905 explicitly lists Persian garm and vernacular garam “hot, warm”. This supports the regional hot-water responses and vowel-inserted forms; it does not infer direct Sanskrit gharma descent merely from cognacy.')
add('f_snjuyk7aztjyu','f_jl7b6dgwhptmm','heart','dil|dili','dil','dil','platts-additional-research.json','Platts p. 523 identifies Persian dil “heart”. Dental d is retained here; retroflex ḍil may belong to the different expressive body/lump family and is excluded.')
add('f_thgcjhj7vr5i2','f_7a3ccguwm7lam','year','sal|salo','sāl','sāl','platts-additional-research.json','Platts p. 626 distinguishes Persian sāl “year” from Indic sal-tree, thorn and house homonyms. The plains sāl year forms support this loan family; northern kāla-derived sāl possibilities are excluded from this pass.')
add('f_nomhmgwrljcl4','f_tusbqtxogmrni','blood','khun|xun|khūn','ḵẖūn','ḵẖūn','platts-additional-research.json','Platts p. 497 identifies Persian khūn “blood”. The distinctive kh/x plus ūn family is retained, separate from inherited lohi/lahu blood words.')
needed={q['parent'] for q in qs};parents={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['ID'] in needed};current={r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1'}
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};acc=[];held=[];rules=[];ix={}
for rr in csv.DictReader((P/'unresearched-records.csv').open()):
 if rr['ID'] in current:continue
 r=raw[rr['ID']];gs={g.strip().lower().rstrip('?!.') for g in r['Gloss'].split(';')};w=norm(r['Form'])
 for j,q in enumerate(qs):
  if w not in {norm(v) for v in q['words']} or not gs<=set(q['gloss']):continue
  if rr['clade'] in {'Kohistani','Chitrali','Shinaic','Kunar'}:continue
  # Arabic kharab is not necessarily borrowed directly into Hindi from Arabic.
  if r['Language_ID']=='H' and j==6:continue
  target=q['persian'] if r['Language_ID']=='H' else q['parent']
  if (j,target) not in ix:ix[j,target]=len(rules);rules.append(dict(parent=target,citation=q['citation'],evidence=q['evidence']))
  i=ix[j,target]
  if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':held.append(dict(record=r,families=[i],reason='Source uncertainty needs review before accepting the loan-family analysis.',passNumber=18));continue
  if r['Language_ID']=='H':ev=q['evidence']+' Platts explicitly identifies the Persian loan in Urdu/Hindi; the target is another Hindi survey attestation.';cite=q['citation']
  else:
   donor=parents[target];ev=q['evidence']+' Existing Hindi '+donor['Form']+' “'+donor['Gloss']+'” ('+donor['Source']+') supplies an attested provisional donor. Its locality is the attestation site; the historical lending locality and possible intervening Indo-Aryan language remain unknown. The borrowed edge is saved under the user’s explicit preference to link supported families despite unresolved cross-IA transmission.';cite=q['citation']+';'+donor['Source']
  if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty concerns locality attribution only, not the lexical reading, as established by the source audit.'
  acc.append(dict(record=r,family=i,parent=target,kind='borrowed',citation=cite,evidence=ev))
(P/'loan-second-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1));(P/'loan-second-primary-articles.json').write_text(json.dumps({k:[dict(url=q['primary']['url'],text=q['primary']['text'])] for q in qs for k in [q['parent'],q['persian']]},ensure_ascii=False,indent=1));(P/'loan-second-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'loan_second_save.py').write_text((P/'sixteenth_save.py').read_text().replace('sixteenth','loan-second'));print('accepted',len(acc),'held',len(held))
for q in qs:
 xs=[x for x in acc if x['parent'] in {q['parent'],q['persian']}];print(q['gloss'],len(xs),sorted({x['record']['Form'] for x in xs}))
