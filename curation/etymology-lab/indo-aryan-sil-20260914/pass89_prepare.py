import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass89';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='7150',citation='CDIAL[7150]',evidence='CDIAL 7150 places the Middle Indo-Aryan replacement *nikka under nikta and explicitly gives Lahnda/Punjabi nikkā small/young, Awan nik shortness and regional nikṛā small. The selected nika/nikkā/nikā/nikī/niko small or short forms fit this family, preserving vowel length, gender ending and non-geminate survey notation. The deeper semantic development and regional transmission are not independently settled.'),dict(parent='9286',citation='CDIAL[9286]',evidence='CDIAL 9286 bubhukṣā hunger gives Lahnda/Punjabi bhukkh and unaspirated Lahnda bukh, plus forms with final kh/k. The bare Pothwari pṳk/pukʰ and Goj pukh responses fit this hunger noun, with initial p versus bh/b and breathy phonation retained as local qualifications. The elicitation gloss hungry is preserved: the analysis treats the bare noun as an elliptical hunger response, not as a full adjectival or verbal construction. Local Indo-Aryan transmission remains open.'),dict(parent='5936',citation='CDIAL[5936]',evidence='CDIAL 5936 tṛṣā thirst explicitly gives Lahnda treh and Punjabi tareh/teh. The selected bare tre responses fit that thirst noun with loss or unmarked realization of final h retained as a local qualification. The survey thirsty gloss is preserved as an elliptical state response; longer treā and tre lagi formations are not included. Transmission within Indo-Aryan remains open.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if f in {'nika','nikkā','nikā','nikī','niko'} and g in {'small','short'}:i=0
 elif f in {'pṳk','pukʰ','pukh'} and g in {'(he is) hungry','be hungry'}:i=1
 elif f=='tre' and g in {'(you are) thirsty','(he is) thirsty'}:i=2
 if f in {'nikā','niko'} and g=='older brother':held.append(dict(record=r,families=[],reason='CDIAL 7150 supports nikka small/young and associated shortness, which makes the older-brother gloss anomalous for this bare response. Check whether a neighboring younger-brother or adjective cell has been missegmented before assigning the family.',passNumber=89))
 if f=='tre' and g=='(you) drink':held.append(dict(record=r,families=[],reason='The close regional lexical comparison is treh thirst (CDIAL 5936), not a drink verb. Inspect the survey row and neighboring thirst prompt before choosing an etymology for this drink-glossed response.',passNumber=89))
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass89_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
