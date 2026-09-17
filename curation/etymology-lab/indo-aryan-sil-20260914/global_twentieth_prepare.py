import json
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'3916':'CDIAL 3916 kheṭa village gives Marathi kheḍẽ hamlet and Gujarati kheṛũ village, matching Khandesi kheḍa. Turner leaves the deeper connection with cultivation qualified.',
'6225':'CDIAL 6225 davara gives Gujarati and Marathi dor/dorā string and rope, alongside regional ḍuri. Bhilali durā thread is compared in this family with vowel raising retained; intra-Indo-Aryan transmission is open.',
'5086':'CDIAL 5086 jaṭā section 1 gives Marathi jaḍ and Gujarati jaṛ root. Bhili jeḍ fits that root family with its fronted vowel retained as a comparative qualification; the deeper substrate analysis remains uncertain.',
'3277':'CDIAL 3277 *kuttira explicitly gives Gujarati and Marathi kutrī female dog. Pauri kutri dog matches this feminine form.',
'5268':'CDIAL 5268 jemana gives Prakrit jemaṇaya right (eating) hand and Old Gujarati jimaṇaüṁ to the right side, alongside Marathi jevaṇ eating. Nimadi jevṇu right fits the eating-hand semantic formation; the exact regional transmission is open.',
'12064-3':'CDIAL 12064.3 bukka gives Oriya buka breast and buku heart/chest, with Assamese buk/buku breast. These eastern breast forms belong to the bukka branch, not the vṛkka kidney head; Turner discusses a non-Aryan deeper origin and semantic contamination.',
'7692':'CDIAL 7692 paṭa cloth includes paṭī and paṭikā, and Pali paṭi cloth/garment. Adivasi Oriya paṭi cloth is compared with this feminine cloth family; its precise regional or learned transmission remains open.',
'5300':'CDIAL 5300 jyotis explicitly gives Oriya joe/joi/jui fire. Adivasi Oriya joi fire is a direct match.',
'3797':'CDIAL 3797 khaṇḍita explicitly gives Oriya khaṇḍiā maimed and khāṇḍiā broken. The Adivasi Oriya survey form matches that participial adjective.',
'5564-2':'CDIAL 5564.2 *dera explicitly gives Hindi derā tent/house, distinguished from retroflex *ḍera in section 1. Goj derā house selects the dental branch; intra-Indo-Aryan transmission remains open.',
'4354':'CDIAL 4354 granthi gives Lahnda gaṇḍhā bulb and Shahpur gaḍḍh onion, alongside Sindhi gaṇḍhi bulb of onion. Goj gaṇḍā onion is compared with this regional bulb family with loss of aspiration retained as a qualification.',
'8330':'CDIAL 8330 pūra gives Western Pahari puro full/complete and Nepali puro complete. Goj puro whole (unbroken) matches the complete sense.',
'5656':'CDIAL 5656 tanū explicitly gives Hindi tan body. Kaithal tan body matches that noun.',
'635':'CDIAL 635 arcis gives Hindi ā̃c flame/heat and Garhwali ā̃c blaze. Kaithal ac fire is compared with that flame family; the broader fire gloss and absence of a nasalization mark are retained.',
'3950':'CDIAL 3950 gagana explicitly gives Jaunsari gaiṇ sky and regional gɔiṇ, with additional gaiṇ in the addenda. Survey gəiṇ sky matches this regional series.',
'7756':'CDIAL 7756 *padara gives Punjabi/Hindi pair foot and Western Pahari pɛ̃r foot. Kullui peːr foot fits this r-extension family; Turner notes alternative *padaḍa for other language forms.',
'3735':'CDIAL 3735 kṣetra gives Punjabi and several neighboring languages khet field. Kullui khet matches that field noun; regional intra-Indo-Aryan transmission remains open.',
'7733':'CDIAL 7733 pattra addenda explicitly give Western Pahari pāc leaf. Kullui paːtʃ leaf matches the affricated regional reflex, rather than requiring an unsupported t-to-c correspondence.',
'9312':'CDIAL 9312 *bokka gives Hindi bokrā goat and Maithili bokṛā he-goat. Kullui bokṛi goat is compared as a feminine member of this regional goat family; the exact transmission remains open.',
'14108-2':'CDIAL 14108.2 *hīyas gives Western Pahari hīj and Prakrit hijjo yesterday; the addenda give Kullui-area hīj and hizz. Kullui hiːdʒ yesterday selects the long-vowel second branch.',
'12812':'CDIAL 12812 ṣoḍaśa section 1 and addenda explicitly give Western Pahari soḷa sixteen. Kullui soˈḷə matches this branch.',
'8652-2':'CDIAL 8652.2 *prathilla gives Western Pahari regional pɔila/pɔ̈la and pɛlo first, with Punjabi paihla. The Kullui paila/poila series selects section 2, not the *prathila heading in section 1.'}
hold={
'6110':'CDIAL 6110 supports the mosquito meaning, but the regional comparanda predominantly have retroflex ḍ whereas Pauri das has dental d. Verify this source spelling independently before using the retroflex regional series; the Noira error is not a universal correction.',
'5244':'Goj jhān body is not the Kalasha jhan attestation quoted by CDIAL 5244. Distinguish jīvanta from a possible jān life/body loan before using the distant Kalasha match.',
'11491':'CDIAL 11491 explicitly warns that vāta and vāyu are not always distinguishable in New Indo-Aryan. Goj bā alone does not select one root; retain the competing family for audit.',
'3761':'CDIAL 3761 khakkhati means laughs, with loud laughter/noise comparanda, not mouth. The Kullui mouth candidate requires a different lexical analysis.',
'274':'CDIAL 274 adhyadhi gives Shina aže upon/upwards, but no Western Pahari counterpart. Kullui adʒe above requires regional evidence to distinguish this reconstruction from another adverbial formation.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0]
 if r['ID'] in done:continue
 if ps==['12064','12064-3']:p='12064-3'
 elif len(ps)>1:continue
 if p=='8652':p='8652-2'
 if p=='14108':p='14108-2'
 if p in hold:held.append(dict(record=r,families=[],reason=hold[p],passNumber=70))
 elif p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('global-twentieth-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'global_twentieth_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','global-twentieth'))
print({'accepted':len(acc),'held':len(held)})
