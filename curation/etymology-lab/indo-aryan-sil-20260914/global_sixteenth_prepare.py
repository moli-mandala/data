import json
from pathlib import Path
P=Path(__file__).resolve().parent
ledger=json.loads((P/'pass-ledger.json').read_text());done=set()
for f in ledger['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec=[
('7012-2','CDIAL[7012.2]','CDIAL 7012.2 *navalla explicitly gives Gujarati navlũ new. Bhil navlu selects this geminate-l branch; regional transmission remains open.'),
('12613','CDIAL[12613]','CDIAL 12613 *śaitala gives Kumaoni selo cold and, in the addendum, Western Pahari śeḷo cold. The e-vowel and l-bearing forms match this branch rather than *śaitalya coldness.'),
('11012','CDIAL[11012]','CDIAL 11012 *lāḍa gives Kashmiri lāra husband/lörī wife, Kangri lāṛī wife and Western Pahari laṛɔ bridegroom/laṛi bride. These regional marital terms match that family, retaining each survey sense.'),
('9749','CDIAL[9749]','CDIAL 9749 matkoṭaka gives Punjabi/Hindi makoṛā and Gujarati makoṛī black ant. The survey retroflex flap forms and ant gloss match that insect family.'),
('11504a','CDIAL[11504a]','CDIAL 11504a *vātodgūra explicitly gives Western Pahari bagur/bagər wind, air and Jaunsari bāgur. The whole survey word matches this compound etymon.'),
('10840','CDIAL[10840]','CDIAL 10840 rodati gives Prakrit roaï/royaï and Hindi ronā, Gujarati rovũ weep. The survey ro cry stem selects the o-vowel branch, not section 2 rudati.'),
('5523-3','CDIAL[5523.3]','CDIAL 5523.3 *dag gives Lahnda dagg, Punjabi dagaṛ and Hindi dagṛā road. The dental-initial survey forms select this branch rather than retroflex *ḍag; regional transmission remains open.'),
('5934','CDIAL[5934]','CDIAL 5934 tṛpra explicitly gives Khowar trup salt. The direct lexical comparison supports this link; Turner also compares *tṛpu, leaving the deeper reconstruction qualified.'),
('6590','CDIAL[6590]','CDIAL 6590 doṣā night explicitly includes Palula dohōṛ/dhōṛ/doṛ yesterday. Turner marks the added material (+?); the lexical family is linked while that unexplained extension remains an audit qualification.'),
('12532','CDIAL[12532]','CDIAL 12532 śubha gives Palula šuwo/šūo, feminine šūī good, and Shina šo/ši good. Survey śo/śī fits the contracted regional forms; inheritance versus intra-Indo-Aryan transmission remains open.'),
('6368','CDIAL[6368]','CDIAL 6368 dīrgha explicitly gives Kalasha drhīga/drīga long, tall. Survey drīgā preserves the diagnostic dr-g sequence and the same meaning.'),
('6547','CDIAL[6547]','CDIAL 6547 deśa explicitly gives Kalasha dēša far, distant beside dēš country. This semantic shift is directly documented, not inferred from the country gloss alone.'),
('2540','CDIAL[2540]','CDIAL 2540 *occha explicitly gives Palula učo a little. The survey ūco few matches the same quantity adjective; source vowel length is retained.'),
('9882','CDIAL[9882]','CDIAL 9882 markaṭa gives Bashkarik makīr monkey beside regional makeṛ/makäṛ and Palula mākaṛ. Survey makār matches this consonantal family; the article questions whether Bashkarik continues feminine markaṭī, so the deeper gender formation remains qualified.'),
('8201-6','CDIAL[8201.6]','CDIAL 8201.6 *pilīla explicitly gives Khowar pilili and Bashkarik pilil ant. These l-l forms select section 6 rather than the pipīla head; Turner notes contamination and uncertain deeper origins throughout this insect family.'),
('8201-5','CDIAL[8201.5]','CDIAL 8201.5 *pippīḍa explicitly gives Bengali pipiṛā/pĩp(i)ṛā ant. Survey pipɽa selects this retroflex branch, retaining the unnasalized spelling documented in the article.'),
('10680','CDIAL[10680]','CDIAL 10680 rājana explicitly gives Bashkarik rān, feminine rēn good. The unusual royal-to-good semantic relation is directly documented; the Prakrit rāṇa formation is itself qualified against rājñ in the article.'),
('1962','CDIAL[1962]','CDIAL 1962 *udguru explicitly gives Bashkarik ugūr and Torwali ūgū heavy. Both survey shapes directly match the named regional comparanda.'),
('4939','CDIAL[4939]','CDIAL 4939 cyavate explicitly gives Bashkarik čō in the walk/go series. Survey co walk!/to walk matches that stem; the slash joins glosses, not separate source forms.')]
rules=[dict(parent=p,citation=c,evidence=e) for p,c,e in spec];index={r['parent']:i for i,r in enumerate(rules)}
keys=set('7012 6722 12613 11012 9749 11504a 10840 5523 6298 5934 6590 12532 6368 6547 3364 11168 2540 9882 8201 10680 1962 9202 4939'.split())
acc=[];held=[]
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 r=x['record'];ps=x['parents']
 if r['ID'] in done or not any(p.split('-')[0] in keys for p in ps):continue
 parent=ps[0];reason=None
 if parent=='5523':parent='5523-3'
 if parent=='8201' and r['Language_ID']=='Kho':parent='8201-6'
 if parent=='6722':reason='CDIAL 6722 explicitly questions whether husband continues dhanika under dhanikā rather than dhanin. Resolve the competing formation before choosing an exact root.'
 if parent=='6298':reason='CDIAL 6298 distinguishes dāru from dāruka. Its Khowar comparator is dar, whereas these survey forms retain final u; the exact formation or learned transmission needs additional evidence.'
 if parent=='3364':reason='CDIAL 3364 supports maimed/defective, not left. A similarity-only link for Palula kuśī left would assume an undocumented semantic development.'
 if parent=='11168':reason='CDIAL 11168 has Palula lohilu and Savi lohĩlo, not contracted lo; Mewari lilo red also needs a regional color comparison, since the cited lilo red is Shina. Do not transfer that language-specific semantic evidence.'
 if parent=='9202':reason='CDIAL 9202 lists Bashkarik bār many but explicitly cites a competing derivation from vaḍra. Resolve this root competition and the source glottal notation before assigning bāʔr.'
 if parent=='9882' and r['Language_ID']=='Chil':reason='CDIAL 9882 supports markaṭa-family monkey forms but does not document Chiliss mokū or explain its loss of the final liquid. Seek direct Chiliss evidence before choosing this family.'
 if parent=='8201':reason='CDIAL 8201 warns that not all ant forms can be assigned exactly to its six branches. Torwali pīval and Chiliss pībīlī need a specific subsection analysis beyond similarity to pipīla/pivīliā.'
 if reason:held.append(dict(record=r,families=[],reason=reason,passNumber=61));continue
 i=index[parent];rule=rules[i];acc.append(dict(record=r,parent=parent,family=i,kind='reflex',citation=rule['citation'],evidence=rule['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
(P/'global-sixteenth-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1)+'\n')
(P/'global-sixteenth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1)+'\n')
s=(P/'global_fifteenth_save.py').read_text().replace('global-fifteenth','global-sixteenth');(P/'global_sixteenth_save.py').write_text(s)
print({'accepted':len(acc),'held':len(held),'parents':len(set(x['parent'] for x in acc))})
