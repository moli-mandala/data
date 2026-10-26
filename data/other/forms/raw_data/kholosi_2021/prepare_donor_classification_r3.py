"""Classify every attribution, distinguishing borrowing from comparisons/blends."""
import csv
import json
import re
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r3'
R.mkdir(exist_ok=True)
claims=list(map(json.loads,(P/'graph-reconciliation-r2/claim-review-queue.jsonl').open()))
rows={r[10]:r for r in csv.reader((P/'whole-proposal-r2/whole-proposed.csv').open())}
donor=re.compile(r'^(?:Persian|Prs\.|Achomi|Larestani|Larestan|Arabic|Ar\.|Gulf Arabic|Turkic|Turkish|Trk\.|English|Eng\.|Balochi)\b',re.I)
# \b after an abbreviated terminal dot is not a word boundary; accept whitespace.
donor_abbrev=re.compile(r'^(?:Prs\.|Ar\.|Trk\.|Eng\.)\s',re.I)
mixed={'hesāb karen','močak','nevalm','sorāx genen','sut kadden','ters karen',
       'nemāz čāš-jo','nemāz šām-jo','nemāz xāftanj'}
alternatives={'dur','jangal','kerem','gāv','xaru','kend'}
decisions=[];class_by_key={}
for c in claims:
 text=c['literal_claim']; normalized=text.strip();qualified=False
 for prefix in ('perhaps ','probably ','dialectal form of ','from a derivative of ','? '):
  if normalized.lower().startswith(prefix):
   qualified|=prefix in ('perhaps ','probably ','from a derivative of ','? ')
   normalized=normalized[len(prefix):]
 direct=bool(donor.match(normalized) or donor_abbrev.match(normalized))
 form=c['source_form'].rstrip('?').strip()
 qualified|='?' in text
 d=dict(source_unit_key=c['source_unit_key'],claim_key=c['claim_key'],literal_claim=text,
        emitted_entry_keys=c['emitted_entry_keys'],loanword=False,qualified=qualified)
 if (form in alternatives and re.search(r'Skt\.|Sanskrit',text,re.I)) or (form=='bakkāu' and ':pdf:' in c['source_unit_key']):
  d.update(disposition='unresolved borrowed versus other ancestry',reason='Competing origin hypotheses do not establish borrowing for the whole form.')
 elif direct and (form in mixed or c['source_unit_key'].endswith('lexicon.csv:row153')):
  d.update(disposition='donor component or donor-scope ambiguity',reason='The expression combines or may combine donor material with target morphology. Preserve exact component attribution without classifying the whole expression as borrowed.')
 elif direct:
  d.update(loanword=True,disposition='explicit donor attribution',reason='The source directly attributes the complete head to a non-ancestral donor language; this supports borrowing classification even without an installed donor node.')
 else:
  d.update(disposition='no direct donor attribution',reason='Ancestry, local derivation, language-internal origin, unknown etymology or mere comparison; no borrowing tag inferred.')
 if d['loanword']:
  for key in d['emitted_entry_keys']:
   class_by_key.setdefault(key,[]).append(dict(source_unit_key=c['source_unit_key'],qualified=qualified,reason=d['reason']))
 decisions.append(d)
# Explicit orthographic variants inherit origin; never infer it through arbitrary
# component links. Source paradigms already share their parent's emitted unit.
changed=True
while changed:
 changed=False
 for key,row in rows.items():
  parent=row[11]
  if parent in class_by_key and key not in class_by_key:
   class_by_key[key]=[dict(variant_parent_key=parent,qualified=any(e['qualified'] for e in class_by_key[parent]),
    reason='Explicit source variant of a head with a direct donor attribution; not a surface-form guess.')]
   changed=True
result=dict(status='complete claim-class proposal; frozen r2 unchanged',claims=decisions,
 row_classifications=class_by_key,claim_count=len(decisions),classified_rows=len(class_by_key))
(R/'donor-classification-proposal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Classified',len(decisions),'claims; direct or explicit-variant borrowing rows:',len(class_by_key))
