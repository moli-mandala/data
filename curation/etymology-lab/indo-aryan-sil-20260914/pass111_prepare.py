import json,csv
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass111'
assert not (P/(stem+'-decisions.json')).exists()
spec=[('9237','CDIAL[9237.1]',{'biralo','berale','biɽal','bilei','bilui','bileⁱ','bileya'},'cat','CDIAL 9237.1 biḍāla gives Nepali birālo, Bengali birāl/bilāl/bilāi, Oriya bilāi and western Pahari bəraḷe in the addendum. Selected biralo/berale/biɽal and bilei/bilui/bileⁱ/bileya fit the rhotic and reduced bilāi branches; source vowel differences and final glide are retained. Local Indo-Aryan transmission is unresolved.'),('9237-2','CDIAL[9237.2]',{'billo','billu','bilra'},'cat','CDIAL 9237.2 *billa gives Sindhi ḇilo, Punjabi/Hindi billā and Hindi bilrā cat. The selected Dotyali billo/billu and Tharu bilra match this contracted branch, with regional final vowel variation; the link leaves local Indo-Aryan transmission unresolved.'),('13544-2','CDIAL[13544.2]',{'sũŋər̃','suŋər','suŋgār'},'pig','CDIAL 13544.2 *sūṅkara gives Kumauni sũgar, Nepali sũgar/sũgur and western Pahari suṅgur pig. The Dotyali velar-nasal forms fit this specifically nasal branch; vowel differences and weakening/loss of the oral velar after ŋ are retained. Turner calls the reconstruction partly onomatopoeic; local transmission remains unresolved.'),('13544','CDIAL[13544.1]',{'śukor','ʃukor','huor','suɹa','suwər','sura'},'pig','CDIAL 13544.1 sūkara gives Pali sūkara, Prakrit sūara, Bengali suor and Bhojpuri sūwar pig. Selected Bengali śukor/ʃukor preserve the velar (possible learned/contact influence), Bishnupriya huor has regional sibilant weakening to h, and Tharu suɹa/suwər/sura match the contracted branch. These are qualified family links, without a claim to a settled immediate Indo-Aryan donor or identical transmission history.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent=p,citation=c,evidence=e) for p,c,s,g,e in spec];acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(p,c,s,g,e) in enumerate(spec):
  if r['Form'] in s and r['Gloss']==g:acc.append(dict(record=r,parent=p,family=i,kind='reflex',citation=c,evidence=e+' Exact response: '+r['Form']+'.'))
for name,obj in [('decisions',dict(accepted=acc,held=[])),('rules',rules)]: (P/(stem+'-'+name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))
