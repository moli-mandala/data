import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass132';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='4159',citation='CDIAL[4159]',evidence='CDIAL 4159 *girati drips/falls explicitly gives Nepali girnu, Maithili girab, Bhojpuri giral and Hindi girnā. Kathoriya gir and Kochila Tharu girale elicited come down match this falling/descending family; the latter retains its finite ending. The directional gloss is compatible with fall, but does not establish whether descent was voluntary. Local Indo-Aryan transmission is unresolved.'),
 dict(parent='1770',citation='CDIAL[1770]',evidence='CDIAL 1770 uttarati explicitly gives descend/alight as well as Bhojpuri utᵃral and Hindi utarnā. Dang uṭər come down is assigned to this descending family, with the survey retroflex notation retained as a phonetic qualification rather than corrected. The shorter Dang uṭ is not included because loss of final r or a different stem needs independent support.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or 'come down' not in r['Gloss'].lower():continue
 i=0 if r['Form'] in {'gir','girale'} else 1 if r['Form']=='uṭər' else None
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'jʰarla','jʰarilak','jʰarlo','jəryo','jʰar','ǰhər'}:
  held.append(dict(record=r,families=[],reason='CDIAL 5328 *jhaṭati and 5346 *jharati both explicitly include Nepali jharnu fall, and 5328 expressly allows the latter etymology. These Danuwar/Dotyali/Majhi/Bote/Tharu come-down forms do not decide between those histories; r/ṛ merger and Dotyali deaspiration do not supply a reliable distinction. Uncertainty is competing roots, not merely an IA borrowing route.',passNumber=132))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
