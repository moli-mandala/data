import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass98';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='10234',citation='CDIAL[10234]',evidence='CDIAL 10234 mūtra explicitly gives Awan mūtur, Punjabi mūtar, Lahnda mutr urine. Pothwari mutor retains the rhotic with a separating vowel, matching that regional family; Siraiki mūtrõ retains the cluster with a final nasalised vowel. Source vowel quality, length and nasalisation are preserved; local Indo-Aryan transmission remains open.'),
dict(parent='6481',citation='CDIAL[6481]',evidence='CDIAL 6481 duhitṛ explicitly gives Lahnda/Punjabi dhī daughter and Punjabi dhīā; Turner discusses the shortened Middle Indo-Aryan kinship forms. The selected northern tī/ti/ti̤(ː), tī̃ and thī variants fit this daughter family with initial devoicing, aspiration/breathy-vowel notation and nasalisation retained as regional qualifications. The link identifies the comparative family without settling local Indo-Aryan transmission.'),
dict(parent='9459',citation='CDIAL[9459]',evidence='CDIAL 9459 bhāra explicitly gives Lahnda/Punjabi bhārā heavy beside bhār load. Selected pārā/pāro/phārā and Pothwari pa̤ra/pa̤r forms fit this adjectival family with initial devoicing and aspiration/breathy-vowel notation retained; final ā/o or its absence is a regional qualification. This does not use the separate rhotic-retroflex or -ī derivative candidates. Local Indo-Aryan transmission remains open.'),
dict(parent='2665',citation='CDIAL[2665]',evidence='CDIAL 2665 kaṇika, replaced by *kaṇikka, explicitly gives Lahnda/Punjabi kaṇak wheat. Pothwari kaṇk fits this noun with medial-vowel syncope, retaining the retroflex nasal and the final velar. The comparison is specifically wheat, not Sanskrit kanaka gold; local Indo-Aryan transmission remains open.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Language_ID'] not in {'Goj','awan','poth','srk'}:continue
 f=r['Form'];g=r['Gloss'];i=None
 if f in {'mutor','mūtrõ'} and g=='urine':i=0
 if f in {'tī','ti','ti̤ː','ti̤','tī̃','tʰī','tʰ ī̃'} and g=='daughter':i=1
 if f in {'pārā','pāro','pʰārā','pa̤ra','pa̤r','pa̤ːr'} and g=='heavy':i=2
 if f=='kaṇk' and g=='wheat':i=3
 if f=='nal' and g=='with':held.append(dict(record=r,families=[],reason='The lexical comparison with Punjabi nāḷ with is strong, but CDIAL 103 explicitly marks its proposed connection to aṅkapāli embrace with a question mark. This is uncertainty about the etymon itself, not merely inheritance versus local Indo-Aryan borrowing. A supported lexical donor route or independent etymological evidence is needed before assigning the remote embrace parent.',passNumber=98))
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass98-daughter-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass98_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),len(held),Counter(x['parent'] for x in acc))
