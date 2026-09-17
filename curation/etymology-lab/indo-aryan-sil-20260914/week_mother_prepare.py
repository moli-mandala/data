import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
ledger=json.loads((P/'pass-ledger.json').read_text());accprev={x['record']['ID'] for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']};eligible={r['ID'] for r in csv.DictReader(open(P/'unresearched-records.csv'))};raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())}
# Revisit the 13 specifically examined rain holds after following Platts's cross-reference.
for x in json.loads((P/'global-eighth-decisions.json').read_text())['held']:
 if x['record']['Gloss']=='rain':eligible.add(x['record']['ID'])
qs=[
 dict(parent='f_s4nms4eshmvdm',citation='Platts[p. 1230];liljegren[entry LX000908]',evidence='Platts p. 1230 explicitly gives Urdu/Hindi hafta week as Persian hafta, from haft seven. The existing Hindi hafta donor is independently recorded in Liljegren LX000908. These haft-/hapt- and contracted āft- week forms are linked to that whole Hindi word as a provisional regional donor; actual intermediate Indo-Aryan or Persian transmission remains open.'),
 dict(parent='11396',citation='CDIAL[11396];Platts[pp. 147, 148]',evidence='Platts p. 148 equates barkhā with barshā; p. 147 derives barshā rain from feminine Sanskrit varṣā. The exact existing parent is therefore CDIAL 11396 varṣā, not generic varṣa 11392. This selection follows the explicit lexical cross-reference; learned or regional transmission remains open.'),
 dict(parent='1351',citation='CDIAL[1351, Addenda]',evidence='CDIAL 1351 āryikā lists WPah. ijjī/ij mother and its addendum expressly gives Jaunsari iji and poetic ije. The surveyed iji/ijā forms fit that regional mother series, distinct from mā́tṛ and dāī.'),
 dict(parent='10016',citation='rensch-hallberg-oleary1992[hindko, items 106–108]',evidence='The cached Hindko source explicitly places mother, older brother, younger brother in adjacent columns, followed by mā, vada pirā, nika pirā. The compiled mother response appears to combine adjacent cells; it must not receive a single-word etymology.'),
 dict(parent='6774',citation='CDIAL[6774];Platts[pp. 504, 555]',evidence='Dāī mother needs a dedicated lexical history. Turner 6774 has aspirated dhāī nurse from dhātrī, while Platts derives unaspirated dāī/daiyā from dātrikā and gives daiyā mother. Neither mā́tṛ nor āryikā nor elderly-relative *dādda is established by the near-match candidate; the needed dātrikā parent has not been located.')]
acc=[];held=[]
for fid in sorted(eligible-accprev):
 r=raw[fid];w=norm(r['Form']);i=None;kind='reflex';reason=None
 if r['Gloss']=='week' and re.fullmatch(r'h?[aāəʌeo]+[fpɸ](?:h)?t+h?[aāəʌeoiu]*(?:h)?',w):i=0;kind='borrowed'
 elif r['Gloss']=='rain' and w in {norm(z) for z in ['barkhā','bʌɾkʰa','berkha','bərkʰa','barsa','barsā','baras','varṣā','varsā','varṣa','borsa','varkha']}:
  i=1
  if w=='baras':reason='Unextended baras rain still competes with varṣa; Platts feminine barkhā/barshā cross-reference does not settle this form.'
 elif r['Gloss']=='mother':
  if r['Language_ID'] in {'jaun','Dotyali','kul'} and w in {norm(z) for z in ['iji','īji','ižā','ijā','iːdʒə','eːdʒə']}:i=2
  elif r['Source'].startswith('rensch-hallberg-oleary1992') and (' ' in w or '/' in w):i=3;reason=qs[3]['evidence']+' Exact record is retained for source-layout audit; no reparsing was performed.'
  elif w in {norm(z) for z in ['dai','dāī','dāyī','ḍai','ḍaⁱ','daⁱ','ḍayi']}:i=4;reason=qs[4]['evidence']
 if i is None:continue
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs lexical-reading verification.'
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=43));continue
 q=qs[i];acc.append(dict(record=r,family=i,parent=q['parent'],kind=kind,citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
(P/'week-mother-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'week-mother-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));s=(P/'loan_fourth_save.py').read_text().replace('loan-fourth','week-mother');s=s.replace(" or x['parent'] in {'f_gtfibpl3lt2qo','f_dcgjc3liwg53s','f_lwa4hsrbk5gee','f_yykcfnb5xlywa'}",'');(P/'week_mother_save.py').write_text(s)
prim=json.loads((P/'near-seventh-primary-articles.json').read_text());prim['f_s4nms4eshmvdm']=[x for x in json.loads((P/'platts-week-mother-research.json').read_text()) if x['word']=='hafta'];(P/'week-mother-primary-articles.json').write_text(json.dumps(prim,ensure_ascii=False,indent=1))
print('accepted',len(acc),'held',len(held))
for i,q in enumerate(qs):print(i,sorted({x['record']['Form'] for x in acc if x['family']==i}))
