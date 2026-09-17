import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass84';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
 dict(parent='3503',citation='CDIAL[3503]',evidence='CDIAL 3503 *kōḍi explicitly means a score/twenty and gives Assamese kuri, Bengali kuṛi, Oriya koṛi and Marathi koḍī. The selected eastern kuri/koṛe/koḍe forms fit that score noun with final-vowel and rhotic/stop details preserved. This leaves the proposed deeper Austro-Asiatic origin and transmission within Indo-Aryan open.'),
 dict(parent='11616',citation='CDIAL[11616]',evidence='CDIAL 11616 viṃśati gives Gujarati/Marathi vīs, Punjabi/Lahnda vīh and regional bīs. The western survey βis forms preserve the labial fricative; βih/βihi/vihi/vĩhi/vih̃i match the h-final series with final vowel and nasalization retained. These local forms are linked to the numeral family without settling internal Indo-Aryan transmission.'),
 dict(parent='12284',citation='CDIAL[12284]',evidence='CDIAL 12284 *śatambhara full hundred gives Marathi śẽbhar and Konkani śembhari/śembor. Khandesi śambɦar and Noiri sambar fit this whole extended hundred noun, retaining vowel and aspiration differences and leaving Marathi-area contact open.'),
 dict(parent='7655',citation='CDIAL[7655];CDIAL[11616]',evidence='Jaunsari pañc bisa one hundred transparently multiplies five by twenty. CDIAL 7655 documents pañca/pā̃c five and 11616 documents bīs/bīsa twenty plus Western Pahari score usage. Two ordered component-family edges record this counting expression, without claiming a single inherited Sanskrit compound.'),
 dict(parent='2462-2',citation='CDIAL[2462.2];CDIAL[3503]',evidence='Danuwar ek kori twenty consists of one × score. CDIAL 2462.2 gives Nepali/Maithili ek one under *ēkka; CDIAL 3503 gives Nepali kori score/twenty. The two ordered component-family links preserve the complete phrase and leave local transmission open.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'].lower();i=None
 if g=='twenty':
  if f in {'kuri','koṛe','koṛi','koḍe'}:i=0
  elif f in {'βis','βih','βihi','vihi','vĩhi','vih̃i'}:i=1
  elif f=='ek kori':i=4
 if g in {'hundred','one hundred','one_hundred'}:
  if f in {'śambɦar','sambar'}:i=2
  elif f=='pañc bisa':i=3
 # Actual semantic contradiction in the selected neighboring numeral series.
 if (g=='twenty' and r['Language_ID'] in {'Sv','Dm','Tor','Bshk','Kalk'} and (' ' in f or '/' in f)) or (g=='one hundred' and f in {'bīś','bīśā'}):
  held.append(dict(record=r,families=[],reason='CDIAL 11616 documents the bīś/bīši numeral as twenty; CDIAL 7655 identifies the accompanying pān/pāñc material as five. In this survey series five-score expressions are glossed twenty while nearby bare bīś forms are glossed hundred. Suspect swapped/aligned source cells; inspect the original table before attaching an etymology or changing any gloss.',passNumber=84));continue
 if i is None:continue
 q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if i>=3 else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.')
 if i>=3:x['components']=['7655','11616'] if i==3 else ['2462-2','3503']
 acc.append(x)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
a=json.loads((P/(stem+'-primary-articles.json')).read_text());a['2462']=json.loads((P/'pass83-primary-articles.json').read_text())['2462'];(P/(stem+'-primary-articles.json')).write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass84_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print({'records':len(acc),'rows':sum(len(x.get('components',[])) or 1 for x in acc),'held':len(held)})
