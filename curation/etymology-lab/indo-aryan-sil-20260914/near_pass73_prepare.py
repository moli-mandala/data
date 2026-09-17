import json
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'9875-3':'CDIAL 9875.3 *marucca explicitly gives Awan maruc pepper. Survey Awan maroc chili selects this u-vowel branch, with o retained; the broader pepper name has been transferred to chili as attested elsewhere in this article.',
'9875-2':'CDIAL 9875.2 *maricca gives Punjabi miric/marc and red-pepper forms, plus Bhojpuri maricā chillies. Goj meric fits this retained-affricate regional pepper family. Rana mitʃi is compared with mircī with r loss retained as a regional phonetic qualification; section 1 marīca instead gives forms lacking the affricate.',
'13161':'CDIAL 13161 saptāha period of seven days explicitly gives Nepali sātā week. Danuwar sata matches that regional form; Hajong septa/sopta retain the pt cluster and are linked as learned Sanskrit loans with their vowels and possible Indo-Aryan intermediary open. Hindi saptah likewise retains the learned cluster.',
'4749':'CDIAL 4749 *cāmala/*cāvala gives Nepali cāmal and Hindi/Bhojpuri cāwar/cāur husked rice. Sunha/Dang cawr/caur forms match the r series, while Dotyali chāmāl matches the nasal/lateral series with initial aspiration retained. Turner leaves the deeper non-Aryan derivation uncertain.',
'3832':'CDIAL 3832 kharva gives Punjabi/Lahnda khabbā and Sindhi khabo left. Goj kabo left matches this k-initial regional family, with deaspiration and singleton b retained; *ḍābba in CDIAL 5539.3 requires a different initial consonant.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0];why=None;kind='reflex'
 if r['ID'] in done:continue
 if ps==['3832','5539-3']:p='3832'
 elif ps==['9875','9875-2']:p='9875-2'
 elif len(ps)>1:continue
 if p not in set(spec)|{'1135','5103'}:continue
 if p=='9875-2':
  if r['Language_ID']=='awan':p='9875-3'
  elif r['Language_ID'] not in ['Goj','Rana']:why='CDIAL 9875 distinguishes maricca and marucca. These Dardic chili forms need local evidence for r-loss, h/x reflexes, or the rounded-vowel branch; resemblance alone does not select section 2.'
 if p=='13161':
  if r['Language_ID']=='Marw':why='Marwari saftɔ week may involve saptāha or contact with Iranian hafta. Resolve the f and immediate donor before choosing a Sanskrit ancestry link.'
  elif r['Language_ID'] in ['H','Hajong']:kind='borrowed'
 if p=='4749' and r['Form']=='cāŋəl':why='Dotyali cāŋəl rice has velar ŋ rather than the m/v or nasal-vowel series documented under 4749. Verify its source reading or a regional consonant development.'
 if p=='5103':why='The janānī/janni wife/woman forms may involve jananī mother or an expanded jani woman stem. The unextended jani article does not settle the added nasal material.'
 if p=='1135':why=('Khowar spā we is not documented by CDIAL 1135 ātman. Investigate asmad and regional pronoun history rather than adopting this near-search match.' if r['Language_ID']=='Kho' else 'Hadoti apnā we needs a distinction between the reflexive/inclusive ātman family and the possessive ātmanaka branch; establish its local pronoun paradigm.')
 if why:held.append(dict(record=r,families=[],reason=why,passNumber=73))
 else:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind=kind,citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('near-pass73-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'near_fourth_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','near-pass73'))
print({'accepted':len(acc),'held':len(held)})
