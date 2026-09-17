import csv,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
qs=[]
def add(parent,gloss,words,citation,ev):qs.append(dict(parent=parent,gloss=gloss.split('|'),words=words.split('|'),citation=citation,evidence=ev))
add('f_gtfibpl3lt2qo','near','najik|nazdik|nasdik|najhik|najdik|nazik|nazdika','kannauji[p. 89];platts1884[s.v. nazdīk]','Platts p. 1136 identifies Persian nazdīk “near”; the existing Hindi nazdik attestation supports a regional donor. Cluster reduction to najik and z/j adaptation preserve the whole near adjective; the particular intra-IA route remains provisional.')
add('f_dcgjc3liwg53s','morning','subah|suba|subeh|sube|subha','census2020bihar[p. 685, item 435, Hindi];platts1884[s.v. ṣubḥ]','Platts p. 742 explicitly gives Arabic-derived ṣubḥ/ṣubaḥ “dawn, morning” in Urdu/Hindi. The existing Hindi subah attestation supports the whole regional loan, including h-loss or metathesized subha; no direct Arabic contact is inferred.')
add('f_lwa4hsrbk5gee','white','safed|saphed|sufed|safet|safedu|saphet','kannauji[p. 91];platts1884[s.v. safed]','Platts p. 662 gives Persian safed/safaid/sufed “white”. The existing Hindi safed attestation supports the regional loan, with f/ph and final d/t adaptation; its deeper Iranian origin is distinct from the provisional immediate donor.')
add('f_5azv73u3n6upa','chicken|hen','murgi|murghi|murgija|murgiya|muragi|mugi|murəgi','kannauji[p. 75];platts1884[s.v. murg̠ī]','Platts p. 1024 explicitly gives Hindi murgī “hen”, derived from murg plus -ikā, while p. 1023 gives the Persian bird/cock base. The feminine survey forms select the complete Hindi murgi donor rather than dropping the gender suffix.')
add('f_yykcfnb5xlywa','chicken|cock','murga|muruga','census2023uttarpradesh[p. 753, item 68, Hindi];platts1884[s.v. murg̠]','Platts p. 1023 states that murg “bird, cock” is commonly murgā in India. The existing Hindi murga cock attestation supplies the whole masculine regional form; it is kept distinct from the murgī hen donor.')
add('f_prmeiidvi4gto','door','darvaza|darvaja|darbaja|darwaja|darwaza|darvazo|dorja','liljegren[entry LX000624];platts1884[s.v. darwāza]','Platts p. 514 gives Persian darwāza “door”. The existing Hindi-Urdu darwaaza head supports the whole regional loan, including v/w/b and z/j substitutions and contracted dorja. The local transmission route remains provisional.')
parents={r['ID']:r for r in csv.DictReader(open(P.parents[2]/'cldf/forms.csv')) if r['ID'] in {q['parent'] for q in qs}};raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv')) if r['Status']=='accepted'};acc=[];held=[]
for rr in csv.DictReader((P/'unresearched-records.csv').open()):
 r=raw[rr['ID']]
 if r['ID'] in current or r['Language_ID']=='H' or rr['clade'] in {'Kohistani','Chitrali','Shinaic','Kunar'}:continue
 w=norm(r['Form']);gs={x.strip().lower() for x in r['Gloss'].split(';')}
 for i,q in enumerate(qs):
  if w not in {norm(z) for z in q['words']} or not gs<=set(q['gloss']):continue
  reason=None
  if q['parent'] in {'f_dcgjc3liwg53s','f_yykcfnb5xlywa'}:reason='The lexical loan is supported, but its proposed Hindi donor currently lacks an accepted ancestry path. A supporting donor analysis is required before the overlay can validate this chain; the form itself is not etymologically unexplained.'
  if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason='Source uncertainty needs a lexical-reading check.'
  if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=28));continue
  ev=q['evidence']+' Hindi '+parents[q['parent']]['Form']+' “'+parents[q['parent']]['Gloss']+'” is used as an attested provisional donor under the user’s preference to link supported intra-IA borrowing families. The recorded donor locality is not asserted to be the historical source locality. An intermediate Indo-Aryan language remains possible.'
  if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty concerns locality attribution only, according to the source audit.'
  acc.append(dict(record=r,family=i,parent=q['parent'],kind='borrowed',citation=q['citation'],evidence=ev))
for donor,parent,entry in [('f_gtfibpl3lt2qo','f_phhnd6c5nukmk','nazdīk'),('f_lwa4hsrbk5gee','f_tfolnhjem4qz6','safed')]:
 assert donor in raw and donor not in current
 i=len(qs);ev='Platts explicitly identifies Persian '+entry+' as the source of this Hindi lexical item. This supporting Hindi survey assignment makes the regional borrowing chain explicit instead of skipping the immediate donor.';cite='platts1884[s.v. '+entry+']'
 qs.append(dict(parent=parent,gloss=[raw[donor]['Gloss']],words=[raw[donor]['Form']],citation=cite,evidence=ev));acc.append(dict(record=raw[donor],family=i,parent=parent,kind='borrowed',citation=cite,evidence=ev))
(P/'loan-fourth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'loan-fourth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
s=(P/'sixteenth_save.py').read_text().replace('sixteenth','loan-fourth')
s=s.replace("assert selected[x['parent']]['Status']!='unlinked',x['parent']", "assert selected[x['parent']]['Status']!='unlinked' or x['parent'] in {'f_gtfibpl3lt2qo','f_dcgjc3liwg53s','f_lwa4hsrbk5gee','f_yykcfnb5xlywa'},x['parent'] # Verified existing lexical donors need not already have their own etymology.")
(P/'loan_fourth_save.py').write_text(s)
print('accepted',len(acc),'held',len(held))
for i,q in enumerate(qs):print(q['gloss'],len([x for x in acc if x['family']==i]),sorted({x['record']['Form'] for x in acc if x['family']==i}))
