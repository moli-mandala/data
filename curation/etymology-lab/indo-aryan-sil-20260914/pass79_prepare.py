import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass79';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11480',citation='CDIAL[11480]',evidence='CDIAL 11480 vāṭa includes Nepali bāri garden and southern Bhagalpur Maithili bārī field, deriving from the enclosed-land/garden family. Survey bari/bāri unirrigated field is compared with that regional field noun; the specific irrigation distinction is preserved rather than asserted for every dictionary comparator.'),dict(parent='4994',citation='CDIAL[4994];Platts[p. 455];platts1884[s.v. chāp]',evidence='CDIAL 4994 *chapp section 1 gives chāp seal/stamp; Platts p. 455 explicitly defines chāp as a seal or signet. Survey chāp/ch ap ring is compared with the signet noun, retaining the broader elicited ring sense and the open intra-Indo-Aryan transmission route. This is not the unrelated roof word returned by the first search.')]
rules[1]['evidence']=rules[1]['evidence'].replace('ch ap','chap');acc=[];held=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=r['Form'];g=r['Gloss'];i=None;why=None
 if w in ['bari','bāri'] and g=='unirrigated field':i=0
 if w in ['cʰāp','cʰap'] and g=='ring':i=1
 if w in ['kā̃','kã'] and g in ['where','where?']:why='The homepage links some kahā̃ forms to iha here, but CDIAL 1605 itself does not establish this interrogative derivation. CDIAL 3384 discusses kuha versus kva for where, while Platts p. 868 proposes kim + sthāna for kahā̃. These contracted kā̃/kã forms need a regional historical or morphological account before selecting one parent; no here-to-where edge is inferred from search.'
 if w=='bari' and g.startswith('unirrigated field;'):why='The survey record combines unirrigated field with heavy or kinship glosses. These are potentially homonymous or misaligned elicitation records; do not attach the entire combined record to the vāṭa field family without a source check.'
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+w+' is preserved.'))
 elif why:held.append(dict(record=r,families=[],reason=why,passNumber=79))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass79_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
