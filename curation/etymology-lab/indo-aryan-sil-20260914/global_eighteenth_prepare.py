import json
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec=[
('7310','7310','CDIAL 7310 *nirguru explicitly gives Punjabi/Hindi niggar and Hindi nigar heavy, solid. This is its intensive heavy sense, not the homonymous not-heavy sense.'),
('9962','9962','CDIAL 9962 mahilā gives Hindi mihariyā and related mehar/mehrī woman, wife, with Gujarati merī. The survey meriya fits this contracted regional family; the article’s alternative prehistoric l/ḍ formations remain qualified.'),
('11374','11374','CDIAL 11374 vardhaka explicitly gives Khowar bardoγ/bardox axe. The exact survey form matches this noun; the article’s borrowing arrow concerns its early transmission into Kalasha, not a loan into Khowar.'),
('135-2x','135.2','CDIAL 135.2 *aṅgūḍi explicitly gives Palula aṅguṛi finger. Survey aŋgūṛī selects this retroflex branch rather than aṅguli.'),
('4855','4855','CDIAL 4855 *cuccu explicitly gives Kalasha čūču breast. The survey cūcū/cū̃cū̃ preserves the expressive reduplicated stem; nasalization does not identify a different etymon in this sound-symbolic family.'),
('3213','3213','CDIAL 3213 kukṣi explicitly gives Kalasha kuč belly. The survey kūc matches that comparator, with vowel length retained.'),
('8399','8399','CDIAL 8399 pota explicitly gives Bashkarik pō/pɔ̈̄ son, boy and Torwali pō child. The survey po/pō child belongs to section 1, not one of the aspirated or extended branches.'),
('10875','10875','CDIAL 10875 lakuṭa includes Nepali lauro stick and Oriya laüṛi stick. The Danuwar lauri and Dotyali lāuro match this k-less branch; intra-Indo-Aryan transmission remains open.'),
('10875-2','10875.2','CDIAL 10875.2 *lakkuṭa includes Western Pahari lakṛɔ log and Hindi lakṛī firewood. Jaunsari lakəda retains the k and matches this branch, retaining its survey dental spelling of the medial stop.'),
('1341','1341','CDIAL 1341 ārdraka gives Kumaoni ādo, Bihari ādī and Oriya adā ginger. Dotyali ādo and Kochila adi/adə match these named regional variants.'),
('4911','4911','CDIAL 4911 *cella explicitly gives Kumaoni celo son. Dotyali celo son matches this kinship sense, beyond the more familiar disciple meaning.'),
('5535','5535','CDIAL 5535 ḍayana explicitly gives Oriya ḍeṇā wing, fin, arm and Maithili ḍen wing, arm. Both the Bhatri arm and Kochila wing senses are directly documented.'),
('11513','11513','CDIAL 11513 vādyate explicitly gives Maithili bājab to speak. The Magahi bājab infinitive matches the eastern speech sense rather than requiring a change from sound based on gloss alone.'),
('11418-3','11418.3','CDIAL 11418.3 *valkhala/*volkala explicitly gives Nepali bokro and Hindi boklā bark. The o-vowel survey forms select section 3 rather than the valkala head.'),
('9183','9183','CDIAL 9183 *bahira explicitly gives Prakrit bahira outside. These Majhi/Bote bahira forms preserve that full adverbial stem; intra-Indo-Aryan transmission remains open.'),
('12159-2','12159.2','CDIAL 12159.2 *viyaṅga explicitly gives Bengali beṅ/byāṅ frog. The Bengali and Bishnupriya bæŋ forms select this branch, not section 1 vyanga marks on skin.'),
('4616','4616','CDIAL 4616 caturmāsa gives Gujarati comāsũ and Hindi caumāsā the Rains. Survey comāsā/comāso rain is this season term used in the elicited rain slot, with the source sense retained.'),
('6251','6251','CDIAL 6251 dākṣiṇa explicitly gives Prakrit dāhiṇa/ḍāhiṇa and eastern dāhin/dāhina right. The selected retroflex-initial Tharu and Oriya dahano forms fit this documented family.'),
('12732','12732','CDIAL 12732 ślakṣṇa gives Oriya sāna small, youngest and Gujarati nānũ small. Oriya sanõ and Bhili nāno younger brother fit the same small/young adjective, with nominal use in the kinship response retained.'),
('9757','9757','CDIAL 9757 matsara gives macchar mosquito across Nepali/Hindi and Gujarati machrũ gnat. Jaunsari mətchər fits that family; the article leaves the deeper reconstruction qualified.'),
('11384','11384','CDIAL 11384 vardhita gives Nepali baṛiyā, Bengali bāṛiyā and Oriya baṛiā excellent. Kaithal baḍiya good fits the non-aspirated regional variant; transmission within Indo-Aryan remains open.'),
('5466-2','5466.2','CDIAL 5466.2 *ṭukka gives Gujarati ṭũk/ṭũkũ small, brief. Palya ṭuku short selects the piece/short adjective branch rather than the cutting verb; the expressive family’s deeper origin is qualified.'),
('55','55','CDIAL 55 agni gives Prakrit agiṇi/agaṇi fire. Kochila agin matches the expanded vowel shape with loss of the final vowel; the n-bearing form is retained.'),
('1388','1388','CDIAL 1388 ālu explicitly gives Nepali ālu and Bengali ālu potato. Chitwan alo and Rathwi ālo preserve this tuber name with final vowel variation; regional transmission remains open.'),
('7727','7727','CDIAL 7727 pati is Sanskrit husband. These patī forms retain the intervocalic t of the learned word, unlike the regular pai/poi forms in the article; the specific path of learned circulation within Indo-Aryan remains open.'),
('13574-2','13574.2','CDIAL 13574.2 sūrya is Sanskrit sun. These sūryā/suryə forms retain the learned ry cluster rather than the regular sujja descendants; the precise path of learned circulation remains open.')]
rules=[dict(parent=p,citation='CDIAL['+c+']',evidence=e) for p,c,e in spec];ix={x['parent']:i for i,x in enumerate(rules)}
keys=set(json.loads((P/'global-eighteenth-primary-articles.json').read_text()));acc=[];held=[]
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0];why=None
 if r['ID'] in done or not any(z.split('-')[0] in keys for z in ps):continue
 if len(ps)>1 and ps!=['13544','13544-2']:continue
 if p=='11384' and r['Form']=='bane':why='CDIAL 11384 vardhita gives baṛiyā-type forms, not bane. Resolve the unrelated-looking nasal form rather than propagating this exact-match candidate.'
 if p=='5466':why='CDIAL 5466 gives piece/cloth and an -ll- extended girl word, but does not account for Bashkarik ṭīkīr daughter or Maiyan ṭūkāī cloth as whole formations.'
 if p=='13290':why='CDIAL 13290 gives Gujarati savār morning. Bhilali həvār needs a local s-to-h correspondence, while Bashkarik sār requires a regional account of the missing medial material; no direct Dardic comparator appears here.'
 if p=='13544':why='Both Niya sugara under sūkara and nasalized Nepali sũgar under *sūṅkara occur in CDIAL 13544. These unnasalized sugar/sugər forms do not select a unique branch without regional evidence.'
 if p=='6251' and r['Form']=='dāvo':why='Mewari dāvo is glossed right, but resembles the left-hand family. Resolve source orientation or a regional right form before choosing dākṣiṇa.'
 if p=='12732' and r['Form']=='nānko':why='The base nānũ small is documented under ślakṣṇa, but the additional k of nānko needs a supported formation or a whole-word donor.'
 if p=='12159':p='12159-2'
 if p=='4701-2':why='CDIAL 4701 documents cāmḍũ skin, but the survey cāmbḍo additionally has b. Establish that cluster development or its local source interpretation.'
 if p=='9757' and r['Language_ID']!='jaun':why='Nimadi macəri and Bhilali michiriyā have additional feminine or extended morphology not given in the cited macchar family; resolve the whole formation.'
 if p=='10223':why='Mewari muḷ-/muḍ- pestles require choosing musala versus *muṣala and accounting for the contraction; Adivasi Oriya mvsul suspiciously matches the Khowar spelling in CDIAL and warrants a source check.'
 if p=='1388' and r['Language_ID']=='KochilaTharu':why='Kochila əlūi potato has final -ui, potentially the ālukī branch normally used for a different aroid. Resolve the formation and species sense before assigning.'
 if p=='9369':why='The primary distinguishes bhaṇṭākī from bhṛṇṭikā. Majhi bhenṭa has an e-vowel without matching the cited nasal bhẽṛā form, while Nimadi bhāṭṭe has a geminate retroflex. Resolve the specific branch/formation.'
 if p=='55' and r['Language_ID']=='B':why='CDIAL 55 lists Bengali āg, not āgun. The agni family is plausible, but obtain a direct Bengali account of the retained n and u-vowel rather than using the short form as sufficient evidence.'
 if p=='11418':p='11418-3'
 if why:held.append(dict(record=r,families=[],reason=why,passNumber=64));continue
 i=ix[p];q=rules[i];kind='borrowed' if p in ['7727','13574-2'] else 'reflex';acc.append(dict(record=r,parent=p,family=i,kind=kind,citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is retained.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('global-eighteenth-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'global_eighteenth_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','global-eighteenth'))
from collections import Counter
print(len(acc),len(held),Counter(x['parent'] for x in acc))
