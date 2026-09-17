import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass119';assert not (P/(stem+'-decisions.json')).exists()
spec=[({'junak'},'CDIAL 5301 jyotsnā gives Assamese zonāk/jonāk moonlight. Bishnupriya junak moon fits that extended eastern noun, retaining the shift from moonlight to the moon itself and the affricate/vowel notation. The -ak is supported by the quoted noun, not discarded as noise.'),({'jɦuṇ','jɦun','jʌn','jən','jana','jono','janha','join'},'CDIAL 5301.1 jyotsnā gives Prakrit joṇhā/juṇhā moonlight, Jaunsari jhūn, Nepali jun moon and Halbi jon moon. The selected regional jh-un/jan/jon/janha/join forms fit this branch, preserving breathy h, final vowels and source vowel notation. Moon versus moonlight is a documented semantic extension within the family.'),({'joṇḍəyya','joṇḍheyye','junneyya','juḍeyye','juṇḍēyya','juneyya','ǰhonhiya','ǰhonihya','ǰonhĩya','ǰonha','dʒʰonha','dʒonija','dʒoniha','dʒonihija','dʒõnih'},'CDIAL 5301.1 and addendum give Old Hindi jonha, Hindi junhāī and Awadhi/Braj jõdhaiyā moonlight, alongside Prakrit joṇhā. The selected Tharu and Bagheli moon responses fit these extended jhonh-/jondh- regional forms, retaining suffix/glide order, nasalization and h/dh variation. The source moon sense is qualified against moonlight; the full extended responses remain intact.')]
rules=[dict(parent='5301',citation='CDIAL[5301.1, addendum]',evidence=e+' Local Indo-Aryan transmission remains unresolved.') for ss,e in spec]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='moon':continue
 for i,(ss,e) in enumerate(spec):
  if r['Form'] in ss:
   q=rules[i];acc.append(dict(record=r,parent='5301',family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
