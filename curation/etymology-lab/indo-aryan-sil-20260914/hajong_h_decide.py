import json,csv,collections
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
inv=json.loads((P/'inventory.json').read_text());eligible={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};audit=json.loads((P/'audit-records.json').read_text());eligible.update(x['record']['ID'] for x in audit);prior={x['record']['ID']:x['reason'] for x in audit}
E={
'snake':('13271','CDIAL 13271 sarpa explicitly gives Assamese xāp and Bengali sāp snake. Hajong hap shares the corresponding p-final stem.'),
'pig':('13544','CDIAL 13544.1 sūkara explicitly gives Bengali suor, Gujarati suvar and regional sūr/sōr pig. Hajong huvar/huvur/hur/hor fit these first-branch outcomes.'),
 'thread':('13561','CDIAL 13561 sūtra explicitly gives Assamese xutā and Bengali/Oriya sutā thread. Hajong huta/hutu fit this thread stem.'),
'dry':('12548','CDIAL 12548 śuṣka explicitly gives Prakrit sukka and regional sukā/suko dry, with Assamese xukān in an extended form. Simple Hajong huka fits the unextended stem; hukna/huknu/hukase are not assigned by this rule.')}
forms={'snake':{'hap'},'pig':{'huvar','huvur','hur','hor'},'thread':{'huta','hutu'},'dry':{'huka'}}
cm=collections.defaultdict(list)
for r in inv:
 if r['Language_ID']=='Hajong' and r['Gloss'] in E and r['Form'].startswith('h'):cm[r['Tags']].append(r)
acc=[];matrix={};rules=[];ix={}
for r in inv:
 g=r['Gloss']
 if r['Language_ID']!='Hajong' or r['ID'] not in eligible or g not in forms or r['Form'] not in forms[g]:continue
 comparanda=[z for z in cm[r['Tags']] if z['Gloss']!=g];assert len({z['Gloss'] for z in comparanda})>=2
 # Exact source-local joint correspondence, not an inference from already assigned graph status.
 matrix[r['ID']]=dict(target=r,comparanda=comparanda)
 k,ev=E[g];ev+=' At the same exact survey locality, '+', '.join(dict.fromkeys(z['Form']+' '+z['Gloss'] for z in comparanda))+' provide at least two other lexical families corresponding to regional initial s/ś/x. Together these establish the working Hajong h correspondence; the joint analysis and all source records are preserved in hajong-h-matrix.json. This supports the initial sound correspondence without settling intra-Indo-Aryan transmission or the extra morphology of the dry comparanda.'
 if g not in ix:ix[g]=len(rules);rules.append(dict(parent=k,citation='CDIAL['+k+']',evidence=E[g][1]))
 x=dict(record=r,parent=k,family=ix[g],citation='CDIAL['+k+'];kim-ahmad-kim-sangma2011hajong[pp. 44–45, 49, 60]',evidence=ev)
 if r['ID'] in prior:x['priorHold']=prior[r['ID']]
 acc.append(x)
for fn,x in [('hajong-h-decisions.json',dict(accepted=acc,held=[])),('hajong-h-rules.json',rules),('hajong-h-matrix.json',matrix)]: (P/fn).write_text(json.dumps(x,ensure_ascii=False,indent=1))
(P/'hajong_h_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','hajong-h'));(P/'hajong_h_decide.py').write_text(Path(__file__).read_text());print('accepted',len(acc),'resolved holds',sum('priorHold' in x for x in acc))
