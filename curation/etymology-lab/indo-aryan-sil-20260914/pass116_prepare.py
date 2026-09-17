import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass116';assert not (P/(stem+'-decisions.json')).exists()
spec=[('2668',{'kā̃t'},'thorn','CDIAL[2668.1]','CDIAL 2668.1 kaṇṭa gives Maithili and Bhojpuri kā̃ṭ thorn. Magahi Nepal kā̃t fits this eastern short-stem branch; the source dental notation is preserved as a qualification, not silently corrected.'),('2668-2',{'kaṭu','kāṭu','koṇṭa','koṇta','kə̃nta'},'thorn','CDIAL[2668.2]','CDIAL 2668.2 kaṇṭaka gives Gujarati kā̃ṭɔ, Marathi kā̃ṭā/kāṭā and Oriya kaṇṭā thorn. The Bhil kaṭu/kāṭu and eastern koṇṭa/koṇta/kə̃nta responses fit this noun branch, retaining nasal loss, final u and source dental/retroflex notation. The adjectival kāṇṭaka thorny entry is not selected.'),('13551',{'hvi','suvi','swi','hõi','hoy','hoi','siyo','śuyo','śuⁱye','śuiyo','sūīā'},'needle','CDIAL[13551.1]','CDIAL 13551.1 sūcī gives Prakrit sūī, Gujarati soy, regional suv/suvva and Nepali siyo (explicitly remodeled by the verb sew), alongside sūiya. Selected western hvi/hoi/hoy/hõi and suvi/swi retain s/h weakening and glide/vowel notation; Nepal-area siyo/śuyo/śuⁱye/śuiyo and sūīā retain the remodeled or extended endings. These are qualified regional family matches, not exact quotations of every source form.'),('13551-2',{'śuc','sũn.ci'},'needle','CDIAL[13551.2]','CDIAL 13551.2 *sūñcī explicitly gives Bengali sũc/suc and Oriya suñci needle. Bengali śuc and Oriya sũn.ci fit this subsection, retaining sibilant, nasal and syllable-marker notation. The primary article permits denasalized Bengali suc, so that form is not assigned mechanically to the nonnasal subsection.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};rules=[dict(parent=p,citation=c,evidence=e+' Local Indo-Aryan transmission remains unresolved.') for p,s,g,c,e in spec];acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(p,s,g,c,e) in enumerate(spec):
  if r['Form'] in s and r['Gloss']==g:
   q=rules[i];acc.append(dict(record=r,parent=p,family=i,kind='reflex',citation=c,evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))
