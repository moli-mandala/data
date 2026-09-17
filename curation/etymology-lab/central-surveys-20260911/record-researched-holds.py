import json
from pathlib import Path
p=Path(__file__).resolve().parent;h=json.loads((p/'holds.json').read_text());d=json.loads((p/'dispositions.json').read_text())
groups=[
('Malvi','mango','keri','Existing 3475 kēsarin links are not substantiated by the full CDIAL entry (only a fibrous-mango adjective). Platts kairī points instead to karīra+ikā, but CDIAL 2804/2805 do not document mango. Competing formation/source remains unresolved.'),
('Nimadi','mango','keri|kairi|kāiri','Same unresolved kairī/kerī formation: Platts karīra+ikā versus unsupported existing 3475 kēsarin link. Primary botanical/semantic evidence needed.'),
('Malvi','man','manak|manakh','Full CDIAL 9827, 9828 and 10049 supply related man/person families but do not explicitly explain this kh/k outcome. Existing similarity links are insufficient.'),
('Malvi','path','gelːo|gela|gelo|ghelo|gel','Full CDIAL 4009 supports a -(l)la extension of gati, but existing specific node 4009-2 has malformed-looking label *ga{l}lati. Resolve the parent representation and local variants before proposing an assignment.'),
('Bagheli','path','geyil|geyl|geli|gelli|gəli','CDIAL 4009’s -(l)la extension is a strong lead, but node 4009-2 is labelled *ga{l}lati. Exact parent-label/subsection reconciliation is required.'),
('Malvi','broom','žhāḍu','CDIAL 5328.2 supports the sweeping verb, but a nominal broom formation has not yet been established. Do not link the noun to the bare intransitive falls root.'),
('Bagheli','broom','jaru|jhaḍu|chaḍu','Sweeping family 5328.2 is a lead; nominal formation and local aspiration/r/ḍ variants require evidence before a noun analysis.'),
('Malvi','rain','barkha|varkha|barkā','Full CDIAL varṣa/varṣā entries identify the rain family but do not resolve the kh outcome or immediate learned/contact route. Further primary evidence needed.'),
('Bagheli','rain','berkha|bərkhə','The varṣa/varṣā family is a lead; the kh formation and immediate transmission are not yet sufficiently established.')]
added=0
for l,g,words,reason in groups:
 for r in d[l]:
  if r['status']=='unexamined' and r['gloss']==g and r['form'] in words.split('|'):
   h[r['id']]={'survey':l,'form':r['form'],'gloss':g,'reason':reason,'researchStatus':'investigated-unresolved','recordedAt':'2026-09-11T11:08:00Z'};added+=1
(p/'holds.json').write_text(json.dumps(h,ensure_ascii=False,indent=2));print('New researched holds:',added)
