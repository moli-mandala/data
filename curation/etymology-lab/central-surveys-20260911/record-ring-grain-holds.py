import json,datetime
from pathlib import Path
p=Path(__file__).resolve().parent
h=json.loads((p/'holds.json').read_text()); d=json.loads((p/'dispositions.json').read_text())
groups=[('Malvi','ring','biṭi|binḍi|vinḍi|viṭi|beṭ','CDIAL 12045 explicitly compares Gujarati vīṭī/vĩṭī ring under *vīṭṭa/*vĭ̄ṇṭa in a broad round/rolled word family. The installed parent is the broad vīṭā tipcat head, with no specific ring-family subnode found. Keep the strong family comparison separate until parent granularity and the local b/v, nasal and vowel variants are reviewed.'),('Bagheli','millet','jeba|jabe|geba|jeua|jaua|jo','Full CDIAL 10431 yava gives Bihari/Maithili/Hindi jau barley, but the survey gloss is millet. Possible grain-name substitution, a broad elicitation category, and local phonological variants need source-level review; similarity alone does not establish the botanical/semantic match.'),('Nimadi','noon','madyanə','CDIAL 9818 madhyāhna means midday, but retained dy in the survey form suggests a conservative or learned/contact form rather than the majjhaṇha outcomes. Immediate transmission and local simplification of hn remain unresolved; no donor stage is inferred without evidence.')]
n=0
for l,g,ws,reason in groups:
 for r in d[l]:
  if r['status']=='unexamined' and r['gloss']==g and r['form'] in ws.split('|'):
   h[r['id']]={'survey':l,'form':r['form'],'gloss':g,'reason':reason,'researchStatus':'investigated-unresolved','recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat()};n+=1
(p/'holds.json').write_text(json.dumps(h,ensure_ascii=False,indent=2));print('New investigated holds',n)
