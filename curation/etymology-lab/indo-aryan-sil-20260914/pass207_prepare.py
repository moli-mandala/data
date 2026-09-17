import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass207';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='12732',citation='CDIAL[12732]',evidence='Full CDIAL ślakṣṇa gives Punjabi nannā small/young, Nepali nāni little girl, Maithili nanuā young/child and Gujarati nānũ small. Simple nani/nenī/nanu younger-sister responses are interpreted as nominal uses of this small/young family. Source vowel differences and unmarked versus feminine endings are retained; this links the lexical family without claiming that the Sanskrit adjective specifically meant sister or settling local IA transmission.'),
 dict(parent='12732',components=['12732','9661'],citation='CDIAL[12732];CDIAL[9661]',evidence='The Marwari nɛnābʰaī younger-brother expression contains small/young plus brother. Full CDIAL 12732 gives regional nannā/nānũ small and 9661 gives Marwari bhāī brother. Retain source ɛ and quantity notation and save both components in order; no single inherited compound or resolved local borrowing route is asserted.'),
 dict(parent='12732',components=['12732','9349'],citation='CDIAL[12732];CDIAL[9349]',evidence='The Marwari nɛnī bɛhan younger-sister expression contains small/young plus sister. Full CDIAL 12732 gives regional nannā/nānũ small and 9349 gives bahin/bahan and Gujarati bahɛn sister. Feminine agreement and source ɛ notation are retained. Save ordered components without reconstructing an ancient compound or resolving cross-IA transmission.')]
sets=[{'nani','nenī','nanu'},{'nɛnābʰaī'},{'nɛnī bɛhan'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in ('younger brother','younger sister'):continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms and r['Gloss']==('younger brother' if i==1 else 'younger sister'):
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
held=[]
holdforms={'nānkɔbʰayɔ','nīnīyɔbʰaī','nenkī bʰagan','nanlu-bɦai','nandlu-bɦaiś','nānlu bɦāis','nānlu bɦāiś','nānlu dādo','nānko bɦāi','nānlo bɦāy','nānlu bɦāy','nanli-boṇi','nandli-bɦoniṣ','nānli bohəṇis','nanli bohnis','nānli boy','nānki bohṇih','nāli boyĩ','nānki bayi','nāni bohṇis','nānli bohni','nānli beyin','nānli bene','nānkyo','nānkibein','nanḍḍo bɦay','nanḍiḍi ben'}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss'] in ('younger brother','younger sister') and r['Form'] in holdforms:
  held.append(dict(record=r,families=[0],reason='The small/young component plausibly belongs to ślakṣṇa 12732; same-survey standalone nānlo/nānlu/nānko small supports segmentation, but the full article does not establish these -l/-k/-ḍḍ extensions. Other records also have -is/-iś/-iṣ on the kin term or a distinct dādo/bayi stem. Research the local morphology and exact kin term before a complete component analysis; the hold is not solely uncertainty about cross-IA borrowing.',passNumber=207))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc),'held',len(held))
