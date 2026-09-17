import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass100';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='4219',citation='CDIAL[4219]',evidence='CDIAL 4219 guvāka explicitly gives Assamese guwā areca/betelnut and Oriya/Bhojpuri guā its nut. The selected eastern gua/gua̯ betelnut responses fit this family with the source syllabicity marking retained. The nut is the areca seed used with betel, not the betel leaf; local Indo-Aryan transmission remains unresolved.'),
dict(parent='4770',citation='CDIAL[4770]',evidence='CDIAL 4770 cāla thatch explicitly gives Assamese sāl roof, Bengali cāl thatch and Oriya cāḷa/cāḷā sloping thatch. Hajong tśal and Bhatri cala/cal roof fit this family with source affricate and lateral notation retained. The elicited roof sense is broader than thatch, and local Indo-Aryan transmission remains open. This does not equate separate chad roof or cauni thatching forms with cāla.'),
dict(parent='8214',citation='CDIAL[8214]',evidence='CDIAL 8214 *pilla explicitly gives Oriya pilā small and as a noun child. Adivasi Oriya/Bhatri pila child matches this established Indo-Aryan family. Turner identifies a deeper Dravidian origin, citing Tamil piḷḷai; this comparative link does not posit direct borrowing from Tamil into either survey language or settle their local Indo-Aryan transmission.'),
dict(parent='4963',citation='CDIAL[4963, sense 1]',evidence='CDIAL 4963 chagala sense 1 explicitly gives Bengali cheli and Oriya cheḷi goat. Adivasi Oriya/Bhatri celi and Adivasi Oriya seli fit this regional feminine goat family with source initial aspiration/sibilant and lateral notation retained. The elided-g cheli comparison supports this branch; local Indo-Aryan transmission remains open.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];l=r['Language_ID'];i=None
 if f in {'gua','gua̯'} and g=='betelnut' and l in {'Hajong','Bishnupriya','B'}:i=0
 if f in {'tśal','cala','cal'} and g=='roof' and l in {'Hajong','Bhatri'}:i=1
 if f=='pila' and g=='child' and l in {'AdivasiOriya','Bhatri'}:i=2
 if f in {'celi','seli'} and g=='goat' and l in {'AdivasiOriya','Bhatri'}:i=3
 if l=='Hajong' and f in {'biśi','biśun'} and g=='seed':held.append(dict(record=r,families=[],reason='Homepage Bengali bici seed points to CDIAL 9261, but the full bījya entry lacks that form; 9250 bīja gives eastern biā/bīyā rather than a sibilant-medial form. Resolve the actual eastern bici/biśi derivation and the biśun ending from primary evidence before choosing a parent. A search link alone is not sufficient.',passNumber=100))
 if l=='Hajong' and f in {'hagol','sagor'} and g=='goat':held.append(dict(record=r,families=[],reason='The eastern sāgol/chāgal comparison is strong, but the shortlisted CDIAL 4963 chagala and 5009 chāgala are distinct related entries, and neither full prose entry explicitly resolves the eastern full-g form. Obtain primary eastern evidence to choose the branch; sagor also retains final r versus l uncertainty.',passNumber=100))
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text())
for n in ['goat','seed']:a.update(json.loads((P/('pass100-'+n+'-primary-articles.json')).read_text()))
f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass100_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),len(held),Counter(x['parent'] for x in acc))
