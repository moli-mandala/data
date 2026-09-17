import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass96';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='6778',citation='CDIAL[6778]',evidence='CDIAL 6778 dhānyà explicitly gives Assamese/Bengali dhān growing or unhusked rice and Oriya dhāna. The eastern survey dhan/dʰan paddy-rice responses match this grain family, including the elicited unhusked sense; source vowel-length and aspiration notation are retained. Local Indo-Aryan transmission remains open.'),
dict(parent='4889',citation='CDIAL[4889, sense 3]',evidence='CDIAL 4889 cūrṇa sense 3 explicitly gives Bengali cūṇ/cūṇā, Assamese sūṇ and Oriya cūna lime. Survey tśun/tʃun/cun and Bishnupriya śunu fit this eastern lime family, retaining affricate/sibilant and final-vowel variation. Betelnut specifies use of the lime, not a citrus-fruit sense. Local Indo-Aryan transmission remains open.'),
dict(parent='4560',citation='CDIAL[4560]',evidence='CDIAL 4560 cákṣus gives Prakrit cakkhu, Bengali cauk/cok(h) and Assamese saku eye. Bengali cok and Hajong tśuk are assigned to this eye family with Hajong affricate notation, u versus Bengali o, and lack of final aspiration explicitly retained. Turner treats the eastern forms as probably Prakrit loans; this comparative-family link leaves that historical transmission and later local Indo-Aryan contact unresolved.'),
dict(parent='4701-2',citation='CDIAL[4701, -ḍa extension]',evidence='CDIAL 4701 distinguishes the -ḍa extension of carman, giving Bengali cāmṛā leather, Oriya camaṛā, Gujarati cāmḍũ and Marathi cāmḍẽ with skin/hide senses. The selected cāməṛo/cāməḍu/cāmaḍu/cāmḍu/camḍo and ʦamṛi/tśamra responses fit this extended family rather than bare camma. Hajong nonretroflex r and source affricate/vowel notation are retained as qualifications. Local Indo-Aryan transmission remains open.'),
dict(parent='10055',citation='CDIAL[10055]',evidence='CDIAL 10055 māma gives Prakrit māma/māaya mother’s brother and Nepali/Kumaoni māmā, Hindi māmā and Oriya māmā. The survey mama/māmā responses to mother’s older brother identify the maternal-uncle noun without asserting that the noun itself encodes seniority. Turner explicitly calls this a nursery word with similar Dravidian forms; independent nursery formation and contact are not settled by the family link. This is not CDIAL 10056 māmaká mine.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if f in {'dhan','dʰan'} and g=='paddy rice':i=0
 if f in {'tśun','tʃun','cun','śunu'} and g in {'lime (for betelnut)','lime for betelnut'}:i=1
 if f in {'tśuk','cok'} and g=='eye' and r['Language_ID'] in {'B','Hajong'}:i=2
 if f in {'tśamra','ʦamṛi','cāməṛo','cāmḍu','cāməḍu','cāmaḍu','camḍo'} and g=='skin':i=3
 if f in {'mama','māmā'} and g in {"mother's older brother",'mother’s older brother'}:i=4
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass96_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))
