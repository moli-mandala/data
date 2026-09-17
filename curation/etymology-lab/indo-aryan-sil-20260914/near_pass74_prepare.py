import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='near-pass74';assert not (P/(stem+'-decisions.json')).exists();done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'10757':'CDIAL 10757 *rukṣa gives Nepali rukh and neighboring rukkh/rūkh tree. Sunha rukhːa matches this tree noun with source gemination retained.',
'3329':'CDIAL 3329 kurkura/kukkura explicitly means dog and gives Nepali kukur and neighboring kukar/kukur forms. The survey kukara/kukuru/kurkur shapes select this k-medial family, not kuttira with medial t.',
'10951':'CDIAL 10951 lamba gives Maithili namā long alongside regional lambā. Dang namba long is compared with this n-initial development, retaining mb rather than claiming an independently established local sound law.',
'12803-3':'CDIAL 12803.3 *kṣaṭ/*kṣvaṭ gives Maithili chao and Jaunsari chau six. Dang chɔᶸ fits this affricated branch and rounded ending, not the unextended ṣaṣ heading.',
'7089-2':'CDIAL 7089.2 nāsikā explicitly gives Gawri nāsī nose. This exact whole form supersedes the near-search suggestion nasta 7031.',
'7785':'CDIAL 7785 panthā gives Gawri phont road and phant Milky Way. Survey fānth path is compared with this regional path family, retaining frication of the initial aspirate and final aspiration as phonetic qualifications.',
'9758-2':'CDIAL 9758.2 *matsiya explicitly gives Kalasha macī fish, including Urtsoon ū-macī. Survey mats hī/mūtsī forms are compared with that extended fish family; aspiration and the rounded vowel are retained as regional phonetic qualifications.',
'7641':'CDIAL 7641 *pakṣya explicitly gives Khowar peṣ-affricate hot (pec̣) and related heat/heating forms. Survey piʦ̣ hot matches the retroflex-affricate family with its raised vowel retained.',
'13952':'CDIAL 13952 haḍḍa gives Awan haḍ and regional haḍḍī bone, with h-less aḍa in Kashmiri. Awan aḍī is compared with this feminine bone family, retaining h loss; the deeper connection to asthi is explicitly doubtful in Turner.',
'8179':'CDIAL 8179 pitṛ explicitly gives Lahnda/Punjabi peo and Awan pio father. Survey peo is a direct regional match.',
'3848':'CDIAL 3848 khalla gives Lahnda/Awan khal hide and Punjabi khall skin, alongside h-less kal in Romani. Goj kal skin is compared with this regional skin family with deaspiration qualified; the deeper relation to challi is doubtful.',
'1268':'CDIAL 1268 āmra gives Punjabi ambī and Hindi ambī mango in the regional borrowed amb series. Goj āmbī fits this feminine mango noun; precise intra-Indo-Aryan transmission remains open.',
'14029':'CDIAL 14029 hastatala gives Punjabi hathelī/thelī and Hindi hathelī palm. Bhatri hātel retains the hand-palm compound with its ending and aspiration differences preserved.',
'6663':'CDIAL 6663 dvāra gives regional dār door. Danuwar der is compared with this contracted door family, with fronting of the vowel retained rather than silently treating it as an exact match.',
'13561':'CDIAL 13561 sūtra gives Nepali sut thread and MIA sutta. Danuwar śutta fits the thread family; sibilant and gemination are retained in the record.',
'10016':'CDIAL 10016 mātṛ gives Maithili/Bhojpuri/Hindi māī mother. Danuwar mai̤ fits this regional mother noun with breathy phonation preserved.',
'1670':'CDIAL 1670 ujjvala gives Maithili ujar/ujjar white and Awadhi ujar white. Danuwar ujura fits that bright/white adjective with medial vowel and final ending retained.',
'11580':'CDIAL 11580 vālukā gives neighboring bālu/bāluwā sand. Hajong bulu is compared with this u-final sand family with raising/rounding of the first vowel qualified; it does not select the i-final vālikā branch.'}
holds={
'13271':'Dang sapuwa snake plausibly expands sāp, but the -uwa material needs local morphological or whole-form evidence; the bare sarpa article does not document it.',
'12452':'Khowar sor head requires comparison with an Iranian sar loan as well as śiras. CDIAL 12452 does not explicitly document this Khowar form; the IA-only borrowing preference does not settle that donor question.',
'700':'Awan ālādā different lacks the g of alagna comparanda and may instead involve a separate loan such as alahida. Establish the immediate lexical donor.',
'9190':'Awan bhõ many lacks the t characteristic of the bahutva comparanda; distinguish bahu from bahutva before selecting an extension.',
'9092':'Goj phuṛ flower has a retroflex rhotic absent from the cited local phul forms. Distinguish phulla from related phura or another regional development.',
'7025-2':'Danuwar laya new requires independent evidence for initial n/l variation before selecting naviya.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e.replace('mats hī','matshī')) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0]
 if r['ID'] in done:continue
 if ps in [['3277','3329'],['3219','3329']]:p='3329'
 elif ps==['9758','9758-2'] and r['Language_ID']=='Kal':p='9758-2'
 elif len(ps)>1:continue
 if p=='7031' and r['Language_ID']=='Gaw':p='7089-2'
 if p in holds:held.append(dict(record=r,families=[],reason=holds[p],passNumber=74))
 elif p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'near_pass74_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
