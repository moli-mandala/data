import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass246';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5589',citation='CDIAL[5589.1]',evidence='Full CDIAL ḍhiḍḍha belly subsection 1 explicitly gives Awankari ḍhiḍ and Punjabi ḍhiḍḍ(h). Selected Awankari, Gojri and Dogri i-vowel ḍiḍ/ṭiḍ forms and close Awankari/Gojri ṭeḍh/teḍ variants fit this regional family, preserving aspiration, initial voicing and vowel notation. Regional aspiration/voicing developments and transmission remain qualified; the separate unaspirated Sindhi branch is not chosen just from surface absence of h. The full same-family slash response is retained. The similar heap entry 5598 is not substituted for the directly documented belly entry.'),dict(parent='5589-7',citation='CDIAL[5589.7]',evidence='Full CDIAL ḍhiḍḍha subsection 7 ḍhaḍḍha explicitly gives Bshk ḍār belly. The five Bshk dār survey responses fit this exact regional comparison, preserving the survey dental versus retroflex d notation and vowel length. This is the specifically numbered a-vowel belly branch, not an inferred link from a generic heap sense. Regional transmission remains qualified.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='belly':continue
 i=None
 if r['Language_ID'] in {'awan','Goj','dog'} and r['Form'] in {'ṭeḍʰ','teḍ','ḍiḍ','ṭiḍ','ḍiḍʰ','ḍiḍ / ṭiḍʰ'}:i=0
 elif r['Language_ID']=='Bshk' and r['Form']=='dār':i=1
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'dʰe','ḍer','her','ḍʰer'}:held.append(dict(record=r,families=[0],passNumber=246,reason='Full CDIAL 5589 explicitly allows Torwali ḍei and Phalura ḍher under either subsection 4 ḍhēḍḍha or subsection 6 ḍhēra. These related d(h)e(r)/her belly responses therefore cannot be assigned to an exact branch on the present evidence; reduced initial consonants also need local checking. The competing branches are an explicit dictionary ambiguity, not just uncertain cross-IA borrowing.'))
 elif r['Form'] in {'ḍeḍ / ḍʰeḍ','ḍeḍ','ḍheḍ'}:held.append(dict(record=r,families=[0],passNumber=246,reason='The belly family is plausible, but these e-vowel Gojri/Vasavi forms may reflect local variation of the ḍhiḍḍha branch or the separate ḍhēḍḍha subsection 4. Full 5589 supplies both but no exact local comparison deciding the branch. Preserve the whole slash response and defer exact parent selection; the heap entry 5598 is not a substitute.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare246.py').read_text());print('accepted',len(acc),'held',len(held))
