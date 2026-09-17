import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='near-pass75';assert not (P/(stem+'-decisions.json')).exists();done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'1200':'CDIAL 1200 āpayati gives the regional āv-/āw- come stem and Hindi ānā, explicitly distinguishing its usual preterite from āgata. Simple ābo/abo/ovo come imperatives are compared with this present stem, retaining regional v/b and vowel differences. Goj āna is an infinitival imperative on the same come verb.',
'6141':'CDIAL 6141 dadāti gives Prakrit deī and Old Awadhi dei gives plus Nepali dinu and regional de- imperatives. Chitwan dei is compared with this give stem; the source combines imperative/past elicitation labels, so no unique tense is asserted.',
'6984':'CDIAL 6984 nava nine explicitly gives Khowar nyoh, with its y unexplained and h possibly Iranian-influenced. Survey nīu is compared with that numeral family, retaining its vowel sequence and lack of final h.',
'13902':'CDIAL 13902 svapati gives Punjabi sauṇā and Western Pahari soṇā sleep, alongside the Awan imperative. Awan soṇā/soṇā̃ matches the regional infinitive used as a directive; nasalization and exact intra-Indo-Aryan transmission remain open.',
'6914-2':'CDIAL 6914.2 *nakkha explicitly gives Bashkarik nakh and Torwali nokh nail. Bashkarik nakhy/nek h/nokhy fits this retained-kh branch; the vowel and final y are retained as regional phonetic or inflectional details.',
'12707':'CDIAL 12707 *śriṣṭa explicitly gives Bashkarik šiṭh/šiṭ house. Survey ṣīṭh fits that retroflex-stop house family with sibilant variation preserved; Turner’s Torwali śīr comparison instead points to a different śrīḍhi family.',
'12326-2':'CDIAL 12326.2 *śarṇa gives Bashkarik šan and Wotapuri šen roof. Bashkarik śon and Chiliss śen are compared with that contracted roof family, preserving the vowel differences.',
'5301-2':'CDIAL 5301.2 *yotsnā explicitly gives Bashkarik yūs un moon, distinguished from the initial-j jyotsnā branch. Survey yesūn is compared with the same y-initial moon noun, with its vowel sequence retained.',
'6658':'CDIAL 6658 dvādaśa explicitly gives Bashkarik bāh twelve. Survey bāha/bāhā/bāʔ matches this numeral family; the final vowel and h-to-glottal realization are retained as phonetic qualifications.',
'9238-2':'CDIAL 9238.2 *beṭṭa gives Hindi beṭā son and beṭī daughter, with neighboring beṭo/beṭi. Danuwar beta/beti is compared with these gendered child nouns; dental t in the survey is retained as a source or phonetic uncertainty.'}
hold={
'1045':'This come response needs its tense and any auxiliary or extended ending resolved. CDIAL 1045 āgata is a past participle, while 1200 āpayati supplies the present stem; the imperative or mixed elicitation label alone does not settle a full-form analysis.',
'1200':'The extended come response contains additional r or ambiguous past morphology not accounted for by bare āv-. Resolve the whole construction before saving a single-stem edge.',
'6141':'This give response may contain an additional imperative/light-verb element, a causative, or a past stem. CDIAL 6141 de- alone does not explain the extra d/r/p or the diyo participial formation.',
'10452':'CDIAL 10452 yāti supplies jā- go, but distinguishes the suppletive past from gata. Resolve the g-initial past response or extra p/r/vc morphology before choosing one ancestry for the whole word.',
'13902':'Goj soro/soṛo sleep contains additional r/ṛ not explained by the local so-/soṇā comparanda; determine whether it includes an auxiliary, ending, or a different stem.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e.replace('nek h','nekh').replace('yūs un','yūsun')) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
keys=set(json.loads((P/(stem+'-primary-articles.json')).read_text()))
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0];why=None
 if r['ID'] in done or not any(z.split('-')[0] in keys for z in ps):continue
 if ps==['1045','1200']:
  if r['Form'] in ['abo','ābo'] and r['Gloss']=='come!':p='1200'
  else:why=hold['1045']
 elif ps==['10452','4008']:why=hold['10452']
 elif len(ps)>1:continue
 if p=='1045' and r['Language_ID']=='Goj' and r['Form']=='āna':p='1200'
 elif p=='1045':why=hold[p]
 if p=='1200' and r['Form'] not in ['abo','ābo','ovo','āna']:why=hold[p]
 if p=='6141' and r['Language_ID']!='Chitwan':why=hold[p]
 if p=='10452':why=hold[p]
 if p=='13902' and r['Language_ID']!='awan':why=hold[p]
 if p=='12707' and r['Language_ID']=='Tor':why='CDIAL 12707 explicitly prefers *śrīḍhi for Torwali śīr house, rejecting the unexplained r from ṣṭ. Find and verify the alternate branch before linking.'
 if p=='12326-2' and r['Language_ID']=='Mai':why='Maiyan śānd roof has a final d absent from the šan/šen comparanda in 12326.2. Verify a local nasal-stop development or another roof formation.'
 if p=='9238-2' and r['Form'] in ['keṭa','keṭi']:why='Danuwar keṭa/keṭi boy/girl has initial k, not the b defining the beṭṭa family. Check the independent keṭā/keṭī child family rather than this near match.'
 if why:held.append(dict(record=r,families=[],reason=why,passNumber=75))
 elif p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'near_pass75_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
