import json
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec=[('138-2x','138.2','CDIAL 138.2 *aṅguṣṭhiya includes Assamese āṅgaṭhi/āṅuṭhi, Bengali āṅṭi and Marathi ãgṭhī finger-ring. The eastern aŋtʰi/aŋthi forms fit this contracted i-bearing branch, not the aṅguṣṭhya head.'),('5070','5070','CDIAL 5070 *chokara gives Gujarati choro/chori and Hindi chora/chori boy/girl, with Assamese sora. The Bhil s-/ts- forms preserve the k-less stem and gender endings; child and daughter are compatible survey kinship senses.'),('5070-3','5070.3','CDIAL 5070.3 *chokkara gives Hindi chokri girl, with the medial k retained. The Kaithal form belongs to section 3 rather than the k-less section 1.'),('4661','4661','CDIAL 4661 candra includes Gujarati cā̃d/cā̃do moon, Assamese sā̃d and Sindhi caṇḍru. The western survey s-initial and retroflex-cluster variants fit this documented moon family; regional transmission remains open.'),('43','43','CDIAL 43 akṣi includes Awankari akh, Maiyan āc̣hi/ãc̣hi and Western Pahari aċh eye. These survey velar and affricate forms match the regional eye family; phonetic x represents the spirant counterpart of kh.'),('10049','10049','CDIAL 10049 mānuṣa gives mānus man, Awadhi mansawā husband, Sindhi māṇhũ and Western Pahari maṇu/māṇu man. The survey forms and husband extension fit this documented family.'),('2485','2485','CDIAL 2485 ekādaśa explicitly gives Marathi akrā eleven. Survey akra matches that western form; its alternative discovery node redirects to 2485, so it is not a competing root. Regional transmission remains open.'),('2462-2','2462.2','CDIAL 2462.2 *ekka explicitly gives Gawarbati yak, Bashkarik ak/akh and Maiyan ak one. The retained velar selects this branch; aykh/æk retain the same stem with survey vowel variation.'),('5994-2','5994.2','CDIAL 5994.2 *trāyaḥ explicitly gives Bashkarik λā three. Survey ɬā is the lateral-fricative transcription of that comparator, selecting the ā-vowel branch.'),('5110','5110','CDIAL 5110 jantu documents the creature-to-snake sense in Niya and Dardic žan/zan/ǰan forms, including Pashai and Torwali comparanda. The Kohistani survey zān forms fit that regional snake family; the exact intra-Indo-Aryan route remains open.'),('6943','6943','CDIAL 6943 nadī gives Prakrit ṇaī river and Western Pahari nɔe/nau in its addendum. Vasavi ṇai and Kullu nau/noe match these explicitly documented shapes.'),('9888','9888','CDIAL 9888 martya explicitly gives Torwali maṣ man and neighboring māṣu husband/man. The Torwali and Chiliss māṣ forms match this regional family; the article notes possible contamination with manuṣya.'),('5798','5798','CDIAL 5798 tārā explicitly gives Torwali tā star in section 1. The survey short form selects this branch rather than the tāraka forms.'),('5798-4','5798.4','CDIAL 5798.4 tāraka gives Gujarati tāro, Nepali tāro and regional tāru/tōru star. The u-ending Bhil forms select this masculine branch; regional transmission remains open.')]
rules=[dict(parent=p,citation='CDIAL['+c+']',evidence=e) for p,c,e in spec];ix={x['parent']:i for i,x in enumerate(rules)}
keys=set(json.loads((P/'global-seventeenth-primary-articles.json').read_text()));acc=[];held=[]
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0];why=None
 if r['ID'] in done or not any(z.split('-')[0] in keys for z in ps):continue
 if p=='138':
  p='138-2x'
  if r['Language_ID']=='dog':why='Initial aṅ- has contracted to gũ-; establish the Dogri-specific development rather than assigning the ring family from gloss alone.'
 elif p=='138-2x':why='Bashkarik āŋgūsīr retains final r, unlike the Indo-Aryan aṅguṣṭhiya comparanda. The primary entry mentions Persian aṅguštarī borrowing for a Nuristani form; resolve a supported immediate donor or different formation.'
 if p=='5070':
  if r['Language_ID']=='Bote':why='Bote chãuḍa has a retroflex stop rather than the primary chor- stem. Resolve the distinct child formation.'
  if r['Language_ID']=='kaithal':p='5070-3'
 if p=='5428':why='All primary ṭaṅka/ṭaṅga leg forms have retroflex initial ṭ; these dental taŋg records need source or correspondence evidence, and the two primary branches must be distinguished.'
 if p=='5798-4' and r['Form'].startswith('ṭ'):why='The survey has initial retroflex ṭ, but every tārā/tāraka comparator in this article is dental. Verify the source transcription or a local correspondence before treating it as the same star family.'
 if p=='5798' and len(ps)>1:p='5798-4'
 if p=='9261':why='CDIAL 9261 bījya gives northwestern bīj/biž, not eastern bisi/bicən. Resolve bīja versus bījya and the additional material rather than copying an old exact-match link.'
 if p=='4424':why='CDIAL 4424 explicitly calls Dardic *ghāna/*ghaṇḍa/*ghanta variants possibly unconnected to ghana. Resolve that root competition; Gawarbati gandālā additionally needs its extension accounted for.'
 if p=='9888' and r['Language_ID']=='Kal':why='CDIAL 9888 marks Kalasha moč as possibly borrowed from Kati. That is not the user-authorized uncertainty about borrowing within Indo-Aryan; resolve the immediate Nuristani donor before saving.'
 if p=='2462':p='2462-2'
 if p=='6943' and r['Language_ID'] not in ['kul','Vasavi']:why='Bhatri lʌ̃di requires an n/l correspondence; Mewari nad may instead continue nada. The nadī article alone does not resolve these forms.'
 if p=='5362':why='The main jhāṭa tree comparanda retain initial aspiration; survey jāḍ/jaḍ needs deaspiration evidence, and Nimadi jhadko additionally needs its -ko formation explained.'
 if p=='10539':why='The primary rakta red reflexes have dental t/tt; these raṭo/raṭõ records have a retroflex stop. Verify the survey distinction or a local sound correspondence.'
 if why:held.append(dict(record=r,families=[],reason=why,passNumber=62));continue
 i=ix[p];rule=rules[i];acc.append(dict(record=r,parent=p,family=i,kind='reflex',citation=rule['citation'],evidence=rule['evidence']+' Exact survey form '+r['Form']+' is retained.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('global-seventeenth-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'global_seventeenth_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','global-seventeenth'))
print({'accepted':len(acc),'held':len(held)})
