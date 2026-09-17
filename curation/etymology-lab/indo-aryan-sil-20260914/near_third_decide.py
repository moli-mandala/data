import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'125':'CDIAL 125 aṅgāra explicitly lists Gawri, Kalasha, Khowar aṅgār, Bshk äṅgār, Maiya agār and Palula aṅgōr in the fire/charcoal family. These survey velar-nasal and vowel variants match the explicitly attested northern fire word.',
'3696':'CDIAL 3696 kṣīra explicitly gives Kalasha/Khowar c̣hir and Bshk c̣hīr milk. The survey retroflex affricate notation and variable aspiration identify that milk family, with final rhotic articulation retained.',
'10978':'CDIAL 10978 lavaṇa gives northern luṇ/lūṇ, Palula lo(h)ōṇ and Torwali lōəṇ salt, including kṭg. lv̄ṇ in the addendum. These survey lateral/retroflex and rounded-vowel variants fit this salt family.',
'10158':'CDIAL 10158 mukha covers both mouth and face, with regional muh/mũh/mūī continuations. The simple mũh/mukh variants fit that family; possible learned or intra-IA transmission remains open. Bshk māī requires separate vowel evidence.',
'10702':'CDIAL 10702 rātrī lists Bshk rāt, Kalasha rat and K. rāth night. The rāt/rōt responses, including final aspiration, fit the night family; initial ə- and loss of the final stop require comparison with alternative formations.',
'8265':'CDIAL 8265 putra gives L. puttur, Punjabi putt/puttar and Hindi pūt son. The survey puttor/putor/pūth/pūthar forms retain this distinctive son family, with aspiration noted as a local detail.',
'9153':'CDIAL 9153 barkara explicitly gives Nepali bākhro, Garhwali bākhro/bākhrī and Punjabi/Hindi bakrā/bakrī goat. The surveyed aspiration and feminine forms match those primary comparanda; the extended Rana bakarya needs separate suffix analysis.',
'8857':'CDIAL 8857 prastara lists the patthar/pathar stone family, including Bengali pāthar and Hindi patthar. Turner expressly describes Punjabi transmission for several medial atth/ath forms; uncertain intra-IA transmission is retained under the user’s policy. Unaspirated patar needs separation from the similar patra leaf family.'}
current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv'))}
qs=[dict(parent=k,citation='CDIAL['+k+']',evidence=v) for k,v in E.items()]; idx={x['parent']:i for i,x in enumerate(qs)}; acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=r['Form'];reason=None
 if r['ID'] in current:continue
 if k=='10158' and r['Language_ID']=='Bshk':reason='Bshk māī face needs evidence for the vowel development before identifying mukha.'
 if k=='10702' and w in {'ərāt','ṛāt','raū'}:reason='Compare initial-vowel night formations or establish final-stop loss before choosing rātrī.'
 if k=='9153' and r['Language_ID']=='Rana':reason='The -ya goat response needs suffix analysis beyond the simple barkara family.'
 if k=='8857' and w in {'patar','pataɾa'}:reason='Unaspirated patar stone needs distinguishing the prastara and patra families locally.'
 if reason:held.append(dict(record=r,families=[idx[k]],reason=reason,passNumber=32));continue
 c=x['comparanda'][k][0]['record'];ev=E[k]+' Previously reviewed comparator: '+c['Language_ID']+' '+c['Form']+' “'+c['Gloss']+'” ('+c['ID']+'). Exact survey transcription remains unchanged.'
 acc.append(dict(record=r,family=idx[k],parent=k,citation=qs[idx[k]]['citation'],evidence=ev))
(P/'near-third-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'near-third-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'near_third_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','near-third'))
print('accepted',len(acc),'held',len(held))
