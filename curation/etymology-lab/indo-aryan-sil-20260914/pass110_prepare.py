import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass110';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='6767',citation='CDIAL[6767]',evidence='CDIAL 6767 dhavala gives eastern dhala/dhalā, Oriya dhaüḷā/dhaḷā and western dhoḷa white. Selected dhola/dhula/dhobla/dobla retain the source rounded vowel and lateral, with b corresponding to v in the fuller form and aspiration variation qualified. Local Indo-Aryan transmission remains unresolved.'),dict(parent='7636',citation='CDIAL[7636]',evidence='CDIAL 7636 pakṣin gives Prakrit pakkhi/pakkhia, Bengali pākhi and Oriya pakhī bird. Hajong pukhi, Bengali pakhi and Bishnupriya pakhiya/pahiya fit this family; source u/a, velar weakening in pahiya and the extended -iya form are retained as regional qualifications. The link does not assert an exact dictionary attestation for every variant or a settled local transmission route.'),dict(parent='3329',citation='CDIAL[3329]',evidence='CDIAL 3329 kurkura gives Prakrit kukkura and Nepali/Assamese/Bengali kukur, Oriya kukura dog. Selected kukul/kukal and kukuṛ/kokuṛ/kukuṛa fit this dog family with lateral/rhotic and source flap/vowel variation retained. This is distinct from kukkuṭa cock; the dictionary calls the family onomatopoeic, so deeper origin and local Indo-Aryan transmission remain unresolved.'),dict(parent='9964',citation='CDIAL[9964]',evidence='CDIAL 9964 mahiṣa documents eastern and regional mahiś/mahis, maĩsa, bhaĩsa and bhaĩsi buffalo variants. The selected eastern mohiś/mohis/moś and moisi/moysi, boĩsi/boisa/boisi and bhois/bhosa-like responses fit this family with rounded vowels, nasalisation, m/bh/b and sibilant notation retained. Shortened forms and possible learned/contact transmission are qualifications; no particular immediate loan route is asserted.')]
sets=[{'dʰola','dʰula','dʰobla','dobla'},{'pukʰi','pakhi','pakhiya','pahiya'},{'kukul','kukal','kukuṛ','kokuṛ','kukuṛa'},{'bʰois','bʰos','mohis','mohiś','moś','boĩsi','moi.ṣa','bə̃i̯sa','bə̃ysa','bəĩsi','bʰoĩsa','bʰʌ̃ysa','bʰẽi̯sa','moisi','moysi','bʰoisa','boisa','boisi'}]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Language_ID'] not in {'B','Hajong','Bishnupriya','Bhatri','AdivasiOriya','Or','Majhi'}:continue
 i=next((i for i,ss in enumerate(sets) if r['Form'] in ss and r['Gloss']==['white','bird','dog','buffalo'][i]),None)
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass110-buffalo-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass110_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))
