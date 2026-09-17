import json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'1200':('1200','CDIAL 1200 āpayati explicitly gives Hindi ānā with stem āw-, Old Marwari āvaï and Gujarati āvvũ come. The article specifically separates the usual preterite from āgata; imperative and infinitive responses can be linked, while mixed come/he-came labels need their tense identified.'),
'2574/3164':('2574','CDIAL 2574 ka includes WPah. ka what/why in its addendum, while 3164 kim includes regional kya/ki what. The bare Tharu/Torwali ka responses do not yet distinguish these pronominal bases securely.'),
'3104/3104-2':('3104-2','CDIAL 3104.2 kalya explicitly includes eastern kāil and Gujarati kāl yesterday/tomorrow. Eastern kail and Bhil kāle fit this branch; northwestern kale competes with the long-vowel kālya branch in section 1.'),
'12803':('12803-3','CDIAL 12803.3 *kṣaṭ explicitly includes Khowar c̣hoi and Lahnda chī/chẽ six. These affricate forms select section 3; western Bhil so still needs distinction from the rounded *ṣuvaṭ branch and contact forms.'),
'12803-3':('12803-3','CDIAL 12803.3 *kṣaṭ explicitly includes Maithili chao and Jaunsari chau six. Bhilali cau fits that affricate-diphthong family; the precise intra-Indo-Aryan transmission remains open.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
acc=[];held=[];rules=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 k='/'.join(x['parents']);r=x['record']
 if k not in E or r['ID'] in done:continue
 parent,ev=E[k];reason=None
 if k=='1200' and 'came' in r['Gloss']:reason='The source labels the same form as come and he came; āpayati versus preterite āgata must be resolved before assigning the entire record.'
 if k=='2574/3164':reason=ev
 if k=='3104/3104-2' and r['Language_ID']=='Goj':reason='Gujari kale yesterday does not securely distinguish kālya section 1 from kalya section 2.'
 if k=='12803' and r['Language_ID'] not in {'Kho','srk'}:reason=ev
 if k not in ix:ix[k]=len(rules);rules.append(dict(parent=parent,citation='CDIAL['+parent.replace('-','.',1)+']',evidence=ev))
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=58))
 else:acc.append(dict(record=r,parent=parent,family=i,citation=rules[i]['citation'],evidence=ev+' Exact survey form '+r['Form']+' is preserved.'))
for fn,obj in [('global-thirteenth-decisions.json',dict(accepted=acc,held=held)),('global-thirteenth-rules.json',rules)]: (P/fn).write_text(json.dumps(obj,ensure_ascii=False,indent=1))
(P/'global_thirteenth_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','global-thirteenth'));(P/'global_thirteenth_decide.py').write_text(Path(__file__).read_text());print(len(acc),len(held))
