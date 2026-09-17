import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass136';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5228',citation='CDIAL[5228]',evidence='CDIAL 5228 jihvā explicitly gives Assamese zibhā, Bengali jib, Oriya jibha and the Prakrit jīhā / Maithili jīh / Hindi jīh series. Selected eastern affricated dž-/ḍž-/dz- and western jibe forms preserve source consonants and final vowels. Short jī/ji/jiu/jīū and breathy jʰi/jʰiha are interpreted with the h-containing series and retained as regional weakening/glide/breathiness qualifications, not source corrections. Local Indo-Aryan transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='tongue':continue
 if r['Form'] in {'ḍžibu','džiba','džibʰa','dzibh','dzib','jibe','jʰi','jʰiha','ji','jī','jiu','jīū','ǰiph'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'ju','jibi','jiβi'}:
  reason='Bishnupriya ju tongue: 5228 jihvā has a rounded Tirahi jub continuation, while subsection 2 juhū has separate žū forms. Short ju alone does not select between the two histories.' if r['Form']=='ju' else 'Final-i jibi/jiβi tongue: compare jihvā 5228 (including inflected jibe) and *jihviya 5231, whose Prakrit jibbhiyā means tongue. The available survey form does not establish an inflectional final vowel versus the derivative; modern scraper senses alone do not exclude the latter.'
  held.append(dict(record=r,families=[],reason=reason,passNumber=136))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
