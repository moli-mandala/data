"""Classify every retained Nordic mention; original lexical rereading remains required."""
import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
src=p/'nordic-volume-kui-mentions.json'
lexical={210,290,291,293,295,296,413,479,481,484,566,570,623,624}
false={118,217,229,476}
out=[]
for record in json.loads(src.read_text()):
 page=int(record['source_page'])
 if 457<=page<=471:status='chapter-owned; not duplicated here'
 elif page in lexical:status='target lexical or morphological evidence; exact original review required'
 elif page in false:status='non-target homograph or geographic/population mention'
 else:status='comparative/classificatory prose without explicitly quoted target token in retrieved span; retain documentary evidence'
 out.append({'printed_page':page,'status':status,'raw_locator_sentences':record['sentences']})
report={'status':'complete classification of existing Nordic mention snapshot only; not an exhaustive original-volume lexical certification',
 'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'records':out,'target_candidate_pages':sorted(lexical),
 'required_next_gate':'Read original context for each target page and capture full explicit comparison attribution/grammatical gloss. Nordic sentences are locator aid, not accepted literal authority.',
 'specific_risks':['p118 kui is a Ho lexical form, not the Dravidian language.','p481 īmu is the printed locator reading; do not silently harmonize to īnu before original review.','p484 mu is a cited honorific suffix; not automatically a lexical pronoun.','Comparative reconstructed/historical statements must remain qualified; no invented ancestry edges.'],
 'canonical_changes':False}
(p/'outsidechapter-mention-census-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(len(out),'mention records;',len(lexical),'target-bearing candidate pages')
