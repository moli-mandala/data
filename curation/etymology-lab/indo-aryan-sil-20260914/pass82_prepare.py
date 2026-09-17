import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass82';assert not (P/(stem+'-decisions.json')).exists();done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'4661':('CDIAL[4661.2]','CDIAL 4661.2 documents candra moon, including Marathi cā̃d and Lahnda/Punjabi cann. The full Beradi candra is treated as learned vocabulary; Awan caṇ fits the contracted regional moon noun with the survey nasal articulation retained. The local Indo-Aryan transmission route remains open.'),
'4701-2':('CDIAL[4701, -ḍa extension]','CDIAL 4701 explicitly gives Lahnda/Punjabi camṛā hide and the skin forms camṛī in the -ḍa extension, represented by canonical 4701-2 *carmaḍa-. The Awan/Goj camara/camra/camri responses retain that extension; source r versus ṛ and epenthetic vowels remain qualified local details.'),
'3245':('CDIAL[3245]','CDIAL 3245 explicitly gives Awan kuṛī woman and Palula kuṛĭ̄ woman/wife. The Goj kurī and Awan qūṛī survey forms fit that feminine noun, preserving k/q and rhotic articulation. Turner discusses a possible Munda or Dravidian source for the deeper family; this link does not resolve it or local Indo-Aryan transmission.'),
'10930':('CDIAL[10930.1]','CDIAL 10930.1 explicitly includes Kumauni latā/lattā clothes and Punjabi nattā cloth used to wipe an oil press. Buksa lʌta and Dotyali nāt̚tā fit the documented l-/n- variants of the rag/cloth family, preserving the broader elicited sense and expressive etymology.'),
'6225':('CDIAL[6225]','CDIAL 6225 gives Hindi daur/dor and Nepali ḍoro thread with regional dorā forms. Chitwan doaɾa fits the string/thread family with its vowel sequence preserved; the regional transmission remains open.'),
'2712':('CDIAL[2712.1]','CDIAL 2712.1 gives Prakrit kēla and Hindi kelā banana. Khowar kilā fits this widespread contracted banana noun with vowel raising preserved; the link leaves the local Indo-Aryan loan route open.')}
rules=[dict(parent=p,citation=c,evidence=e) for p,(c,e) in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0]
 if r['ID'] in done:continue
 if set(ps)<= {'4701','4701-2'}:p='4701-2'
 elif len(ps)>1:continue
 reason=None
 if p=='6225' and r['Language_ID']=='Buksa':reason='Buksa doɖa thread has a medial retroflex stop where the davara/dora comparanda have r. Verify this source reading or a local stop/flap correspondence before linking.'
 if p=='2712' and r['Language_ID']=='DewasDoneDanuwar':reason='Danuwar kʰera banana adds aspiration absent from the cited kerā series. Check the survey reading/local aspiration before treating it as the same noun.'
 if p=='3727':reason='Dang ghuri knife has a voiced aspirated velar initial; CDIAL 3727 supplies ch-/kh- but no corresponding gh- form here. The search suggestion alone does not establish that development.'
 if p=='1008':reason='The sky forms akʰas/aŋkaś differ from ākāśa in aspiration or an inserted nasal. The full CDIAL article does not document these local forms; check regional evidence/source before assigning the heading.'
 if p=='8330':reason='Awan puṛī has a retroflex flap unlike the pūra comparanda; Dhundari pūrṇa instead preserves the separate pūrṇa formation. Resolve each against the appropriate full entry rather than accepting this pūra search candidate.'
 if p=='13355':reason='The all/whole search candidates need separation of sarva and sāra. The v-bearing Bashkarik forms may belong to sarva, the r-bearing forms need regional vowel evidence, and Dang saraj contains unexplained final material. Neither dictionary entry resolves the full selected form by itself.'
 if reason:held.append(dict(record=r,families=[],reason=reason,passNumber=82));continue
 if p not in ix:continue
 q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind=('borrowed' if p=='4661' and r['Language_ID']=='M' else 'reflex'),citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass82_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
