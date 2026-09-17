import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass191';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='9818',citation='CDIAL[9818]',evidence='Full madhyāhna midday article explicitly gives Pali majjhanha/majjhaṇha and Prakrit majjhaṇha/majjhaṇṇa. Simple survey majhan/majun and mãnjʰan/mãŋjʰan preserve the contracted midday stem; the nasalized forms retain nasal anticipation/placement as a qualification. Consonant-retaining mədʰyan.a/madyanə/madyan/madhiyan/maḍiyan may reflect learned or local IA transmission, which remains unresolved; retroflex and vowel notation are preserved. No multiword response is collapsed to this head. Full madhyaṃdina and madhyānta were inspected separately, including Turner’s warning that Sindhi mañjhandi may instead continue *madhyaṃdiva.'),
 dict(parent='10039',citation='CDIAL[10039]',evidence='Full *mādhyāhnaka/mādhyāhnika of midday article explicitly gives Prakrit majjhaṇhiya. Survey mãnjʰaniyā/majānīyā/majʰanīyā/majʰanīā noon retain the longer -niyā/-nīā formation rather than being assigned to unsuffixed madhyāhna. Nasal anticipation, vowel length and h-loss are preserved as regional qualifications; local IA transmission remains unresolved. The separate Sindhi lunch comparison in the article is not used to justify these forms.')]
sets=[{'mãnjʰan','mãŋjʰan','mədʰyan.a','madyanə','madyan','madhiyan','maḍiyan','majhan','majun'},{'mãnjʰaniyā','majānīyā','majʰanīyā','majʰanīā'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='noon':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
