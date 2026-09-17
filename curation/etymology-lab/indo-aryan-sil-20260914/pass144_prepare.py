import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass144';assert not (P/(stem+'-decisions.json')).exists()
base='CDIAL 13119 sadṛkṣa like explicitly supplies Prakrit sarikkha/sārikkha, Punjabi sarkhā, Hindi sarikā/sarīkhā, Old Marwari sārikho and Gujarati sarkhũ. Selected western sarka/harka responses match this comparative family, with s/h realization, aspiration variation and final vowels retained as qualifications. The same gloss denotes likeness/equivalence; local Indo-Aryan transmission is unresolved.'
rules=[dict(parent='13119',citation='CDIAL[13119]',evidence=base),dict(parent='2462-2',citation='CDIAL[2462.2];CDIAL[13119]',evidence='Ordered components ek + sarka/harka: CDIAL 2462 subsection 2 *ekka explicitly gives Hindi and Gujarati ek one. '+base+' Both lexical components of the response are linked, preserving source spacing and inflection; this is not a reconstructed ancient compound.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
simple={'sarka','sārəkā','harkun','hārkũ','harkha','harkā','harka','sārakha','sarke','sarkiyo','harko'}
compound={'ek sarkā','ek sārkā','ek harku','ek harkā','eksariko','eksarika','eksarikā','eksārika','ekharkā','ekhārka','ek sarko','ek harko','ek sarkoi','ek sarka'}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='same':continue
 i=0 if r['Form'] in simple else 1 if r['Form'] in compound else None
 if i is not None:
  q=rules[i];a=dict(record=r,parent=q['parent'],family=i,kind='component' if i else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
  if i:a['components']=['2462-2','13119']
  acc.append(a)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print(len(acc),sum(len(x.get('components',[x['parent']])) for x in acc))
