import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
for r in rs:
 if r['ID'] in done or r['Language_ID']!='Kho':continue
 f,g=r['Form'],r['Gloss'];x=None;reason=None
 if f in ['kūmoro','kūmorū'] and g=='girl':x=('3303','CDIAL[3303];CDIAL[13448];bashir2023[s.v. kumóru]','Turner explicitly derives Kho kumóru girl, servant girl from *kumarakā within the kumāra family. The mention under sukumāra 13448 is a vowel comparison, not a derivation from the prefixed sukumāra; both full entries have been checked.')
 if f in ['kīmirī','kīmerī','kīmɛrī'] and g=='woman':x=('3303','CDIAL[3303];bashir2023[s.v. kiméri]','Turner explicitly derives Kho kiméri woman from kumārikā within the kumāra family. Bashir confirms the form and etymological reference; the survey’s vowel variants preserve the same consonants and sense.')
 if f in ['bṓk','bōk'] and g=='wife':x=('11250','CDIAL[11250 and addenda];bashir2023[s.v. boók]','Turner explicitly identifies Kho bok wife with vadhū and corrects its extension in the addendum to *vadhukkā rather than vadhukā. Bashir independently cites T11250. The saved family link preserves this specific extension in its evidence.')
 if f=='źāo' and g=='son':x=('10446','CDIAL[10446];bashir2023[s.v. žaʋ]','Turner yahu young, son explicitly gives Kho žau son; Bashir žaʋ son, fetus cites the same etymon. The survey źāo is the corresponding diphthongal form; the isolated āo response is excluded because its missing onset needs checking.')
 if f in ['gi-','gī-'] and g=='come':x=('4121','CDIAL[4121 and addenda]','Turner gāti explicitly includes Kho gīk come, goyan, gīru and absol. giti. It explains the ī-vocalism by influence from aiti in other Dardic languages. The survey gi-/gī- stems match this documented paradigm.')
 if f in ['ūlī-','ūlūī-'] and g=='he flew':x=('2061-5','CDIAL[2061,5];bashir2023[s.v. uluík]','Turner section 5 gives Kho uléik throw away/pour out and its intransitive ulúik fly away/fall under udrecayati. Bashir explicitly lists uluík and Yarkhun uliík. The exact fifth branch is used, rather than undifferentiated udricyate or an unrelated flying verb.')
 if f=='grɛniś' and g=='noon':x=('4530','CDIAL[4530];bashir2023[s.v. graníš]','Turner *ghraṇiṣya sun’s heat explicitly cites Kho graniš noon. Bashir gives graníš and greníš for noontime/early afternoon. The differently glossed grānis morning response is excluded pending source-context review.')
 if f=='hanūn' and g=='today':x=('7576','CDIAL[7576];bashir2023[s.v. hanún]','Turner nūnam now explicitly gives Kho hanún today and provisionally identifies ha- as an emphatic particle. Bashir independently gives hanún and its base nun today.')
 if f in ['mūlā','mūḷā','mūḷa','mūḷī / mūlā'] and g=='below':x=('10250-2','CDIAL[10250,2];bashir2023[s.v. muɫ]','Turner explicitly places Kho muḷ root/bottom and muḷa below in the *mūḍa second branch. Bashir documents the directional muɫ forms. These single-stem responses, including the slash-separated inflectional variants, retain the specific second branch rather than unsplit mūla.')
 if f in ['por-','pār-'] and g=='lay down':x=('7722','CDIAL[7722,1];bashir2023[s.v. porík]','Turner patati explicitly includes Kho porik lie down/sleep, and Bashir confirms the intransitive porík beside causative poréik. The bare survey por-/pār- stems are matched with this family; no additional causative construction is assumed from the English elicitation gloss alone.')
 if f=='hanūnt' and g=='today':reason='The otherwise matching hanun today form has an unexplained final t in this survey response; retain for original-source verification.'
 if f in ['cūī','cʰūī','cʰī'] and g=='night':reason='Turner 3652 originally compared Kho čhui with kṣapā, but the addendum explicitly questions this and says čh- scarcely derives from kṣāp-. Hold rather than present the superseded suggestion as firm.'
 if f in ['istān','stān'] and g=='roof':reason='Turner 13686 compares istàn with staraṇa through *starn and early syncope, marked with a question. Bashir repeats the reference; the detailed phonological bridge still needs review.'
 if f in ['bī-','bā-','bo-'] and g=='go':reason='Turner 497, 12223 and 12225 offer competing apaiti, vyeti and vrajati analyses for Kho bik/be, with a suppletive past from apagāt. Hold the elicited short stems until the paradigm and preferred historical analysis are established.'
 if f in ['nīś-','niś-'] and g=='sit down':reason='Turner 7467 explicitly treats Kho nišik as a Nuristani borrowing, while Bashir cites Sanskrit ancestry without resolving the immediate route. No specific donor language/node is yet established.'
 if f in ['miṛū','mirū','mīṛū'] and g=='urine':reason='Bashir mirú urine cites T10239 *mūtriya, but the full CDIAL entry contains no Khowar comparandum or explanation of the vowel development. The retroflex-flap survey variants need a fuller historical argument; not assigned solely from the cross-reference.'
 if x:a.append(dict(record=r,parent=x[0],kind='reflex',citation=x[1],evidence=x[2]))
 if reason:h.append(dict(record=r,reason=reason))
(P/'pass112-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print('accepted',len(a),'held',len(h))
p=P/'lexical-reference-links.json';links=json.loads(p.read_text())
for x in a:
 for c in x['citation'].split(';'):
  if c.startswith('bashir2023'):links[c]='https://escholarship.org/uc/item/955239w9'
p.write_text(json.dumps(links,ensure_ascii=False,indent=1)+'\n')
