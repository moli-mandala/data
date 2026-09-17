import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass115';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
spec=[('f_tkxf73rv5gova',{'pattagobī','pattāgobi','pattāgɔbī','pattā gobi','pattāgobī','pāttā gobi','patta gobī','patːa gobi','pattagobi','patːā gopi','pāttā gobɦi','patta gobi','patːāgopi','pāttāgobhi'},'pattā'),('f_tpsp2a3uyjx72',{'patā gobī','patā gopī','patāgobʰī','pata gopī','patā gɔpī','patāgobi'},'pātā'),('f_wt6ue5yzllris',{'patgobi','pāt gobī','patɡobʰi'},'pāt')]
rules=[]
for parent,forms,head in spec:
 rules.append(dict(parent=parent,citation='CDIAL[7733];CDIAL[4270];centralbank-saral-kannada[PDF p. 11, vegetables, item 11]',evidence='The primary vegetable list explicitly gives Hindi पत्तागोभी cabbage. CDIAL 7733 attests Hindi '+head+' leaf; CDIAL 4270 gives Hindi gobī/kopī/gobhī cabbage. Two ordered component links to the existing Hindi lexical nodes represent leaf + cabbage. Regional length, gemination and gobi/gopi aspiration/voicing notation are preserved; these identify the lexical components without claiming an inherited Sanskrit compound or a settled route of local Indo-Aryan transmission.'))
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='cabbage':continue
 for i,(parent,forms,head) in enumerate(spec):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=parent,components=[parent,'f_jngmzar4rfvqg'],family=i,kind='component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a['vegetable-list']=[dict(localImage='vegetable-primary-page-11.png',text='Hindi column, item 11: पत्तागोभी cabbage. Page image visually inspected in the preceding pass and preserved alongside this audit.')];f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print(len(acc),len(acc)*2)
