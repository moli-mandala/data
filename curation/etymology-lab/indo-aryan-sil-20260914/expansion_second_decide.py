import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv'))}
cs={}
for file in ['phonetic-second-expansion-candidates.json','semantic-second-expansion-candidates.json']:
 for x in json.loads((P/file).read_text()):cs.setdefault(x['record']['ID'],x)
acc=[];held=[];qs=[];ix={}
for fid,x in cs.items():
 if fid in current:continue
 r=raw[fid];w=r['Form'];ps=x['parents'];reason=None
 if len(ps)!=1:raise AssertionError(ps)
 target=ps[0];c=x['comparanda'][target][0]
 if target=='7733' and w=='pālā':reason='Lateral pālā leaf needs comparison with pallava; the pattra match alone is insufficient.'
 elif target=='8857' and w=='patar':reason='Unaspirated patar stone needs local prastara/patra comparison.'
 elif target=='2575':reason='Kui/kuvã who needs distinguishing kaḥ-punar from ko-api or another interrogative formation.'
 elif target=='2830':reason='Final velar nasal in kaŋ ear needs local correspondence beyond the nasal match.'
 elif target=='9758':reason='Northwestern macchī fish remains ambiguous between matsya and the explicitly regional *matsiya branch.'
 # The addendum has no separate stored *puñcha node: retain the parent article,
 # citing the precise subsection rather than confusing it with *pucchaḍa (8249-2).
 extra=''
 if target=='8249':extra=' CDIAL 8249 Addenda 2 specifically places nasalized tails under *puñcha; no separate *puñcha node exists in the current graph, so the containing article is used with this exact subsection recorded (8249-2 is *pucchaḍa, not *puñcha).'
 if target=='4655-2':extra+=' CDIAL 4655.2 explicitly includes regional cār from catvāri; these a-vowel forms are retained here, unlike the o/u-vowel northern caturaḥ branch.'
 if target=='5798-4':extra+=' CDIAL 5798.4 explicitly includes Oriya tarā star, supporting the present Bhatri/Adivasi Oriya vowel variants.'
 if target=='7136':extra+=' CDIAL 7136 Addenda explicitly gives neḍi near.'
 if target=='4428':extra+=' CDIAL 4428 gives śeu. kár house, supporting the northwestern devoiced initial with the local phonation marks preserved.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty needs lexical-reading review.'
 cite='CDIAL[8249, Addenda 2]' if target=='8249' else c['citation']
 ev='Reviewed whole-form continuation of '+c['record']['Language_ID']+' '+c['record']['Form']+' “'+c['record']['Gloss']+'” ('+c['record']['ID']+'). Family evidence: '+c['evidence']+extra+' The present source form was inspected for phonetic and gloss differences; normalization was used only to find it, and its exact spelling is preserved.'
 if target not in ix:ix[target]=len(qs);qs.append(dict(parent=target,citation=cite,evidence=ev))
 if reason:held.append(dict(record=r,families=[ix[target]],reason=reason,passNumber=38));continue
 acc.append(dict(record=r,family=ix[target],parent=target,citation=cite,evidence=ev))
# Resolve an overly broad previous filter: final -a in jura is not an a-vowel root.
x=next(x for x in json.loads((P/'near-fifth-decisions.json').read_text())['held'] if x['record']['ID']=='f_727n4vst7idby');r=x['record'];target='5090-4';ev='CDIAL 5090.4 *jūḍa explicitly gives jūr/juṛa cold. Kochariya East Danuwar jura has the diagnostic u root vowel; final -a is not evidence for the a-vowel jaḍa branch. This corrects an overly broad screening hold in near-fifth-decisions.json.'
ix[target]=len(qs);qs.append(dict(parent=target,citation='CDIAL[5090.4]',evidence=ev));acc.append(dict(record=r,family=ix[target],parent=target,citation='CDIAL[5090.4]',evidence=ev))
(P/'expansion-second-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'expansion-second-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'expansion_second_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','expansion-second'))
print('accepted',len(acc),'held',len(held))
