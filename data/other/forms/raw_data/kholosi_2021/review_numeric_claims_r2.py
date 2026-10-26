"""Explicit source-claim dispositions for review; no canonical or r1 mutations."""
import csv
import json
import re
from pathlib import Path

P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r2'
claims=list(map(json.loads,(R/'claim-review-queue.jsonl').open()))
targets={r[0]:r[1] for r in csv.reader((P.parents[4]/'data/cdial/params.csv').open())}
# These are source number/head inconsistencies, not competing proposed ancestries.
number_head_repairs={
 ('ānen','11714'):('1174','ā́nayati'),
 ('sino','12372'):('12732','ślakṣṇá'),
 ('hoyen','9486'):('9416','bhávati'),
 ('tʰal','13774'):('13744','sthála'),
 ('tʰāṛgo','6386'):('6368','dīrghá'),
 ('parono','3283'):('8283','purāṇá'),
 ('perāno','3283'):('8283','purāṇá'),
 ('saodo','12359'):('12357','śaśá'),
}
# Read in full, not inferred from the mere presence of an identifier.
unlinked_forms={
 'bakkāu':'Several alternatives including Persian and onomatopoeia; no unique ancestry selected.',
 'dur':'Persian versus Sanskrit alternatives remain unresolved.',
 'gāv':'Persian versus Sanskrit alternatives remain unresolved.',
 'jangal':'Persian versus Sanskrit alternatives remain unresolved.',
 'kerem':'Persian versus Sanskrit alternatives remain unresolved.',
 'xaru':'Persian versus Sanskrit alternatives remain unresolved.',
 'kend':'Alternative Balochi origin and Sanskrit comparison do not establish unique ancestry.',
 'sanā':'Explicit alternative Sanskrit ancestry hypotheses remain unresolved.',
 'viyāv':'Explicit alternative Sanskrit ancestry hypotheses remain unresolved.',
 'var':'Vara versus a vṛdh derivation remains unresolved.',
 'xallo':'Author explicitly says probably not; rejected suggestion must not become a link.',
 'deh':'PDF sun head rejects/objects to the Sanskrit comparison; CSV day head reviewed separately.',
 'suk karen':'Sindhi comparison with an explicit sound-correspondence objection, not direct ancestry.',
 'puni jelji':'Numeric ancestry is explicitly first-component-only; whole compound requires component analysis.',
 'ke':'Printed number2184 belongs to upadhārayati, not cited kṛte; exact intended target unresolved.',
}
records=[]
for c in claims:
 if not c['cited_target_ids']:continue
 form=c['source_form'].rstrip('?').strip()
 text=c['literal_claim']; ids=[t['cdial_id'] for t in c['cited_target_ids']]
 pdf=c['provenance']=='printed author dictionary'
 record=dict(claim_key=c['claim_key'],source_unit_key=c['source_unit_key'],
             literal_claim=text,source_form=c['source_form'],emitted_entry_keys=c['emitted_entry_keys'],
             printed_ids=ids,proposed_parameter_ids=[],source_qualified=False)
 reason=unlinked_forms.get(form)
 if form=='deh' and not (pdf and 'unknown' in text.lower()):reason=None
 if reason:
  record.update(disposition='unlinked with explicit source-scope reason',reason=reason)
 else:
  selected=ids.copy()
  if form=='bere' and pdf:
   selected=['9187']
   record['comparison_only_ids']=['9193']
   record['reason']='BAHÚ is the proposed derivational ancestry; BAHURA belongs to the Kutchī comparison.'
  elif form=='mers' and pdf:
   selected=['9828']
   record['separate_conflation_claim_ids']=['8289']
   record['reason']='Primary ancestry manuṣya; puruṣa is a separately attributed conflation claim, not a second unqualified reflex.'
  elif len(ids)>1:
   raise AssertionError(('Unreviewed multiple identifiers',c))
  repaired=[]
  for ident in selected:
   correction=number_head_repairs.get((form,ident))
   if correction:
    target,head=correction
    assert targets[target]==head
    repaired.append(target)
    record.setdefault('number_head_corrections',[]).append(dict(printed_id=ident,
       printed_id_target_head=targets[ident],intended_head=head,proposed_id=target,
       rationale='The source explicitly names this head, uniquely found at the corrected ID; printed number is retained as conflicting evidence.'))
   else:repaired.append(ident)
  record['proposed_parameter_ids']=repaired
  record['source_qualified']=bool(re.search(r'probably|perhaps|uncertain|unknown|\?',text,re.I))
  record['disposition']='source-qualified direct ancestry proposal' if record['source_qualified'] else 'direct source ancestry proposal'
  record.setdefault('reason','Source explicitly attributes this ancestry; target existence and head/subhead reviewed. Historical assignments are reconciled separately and never overwritten here.')
 records.append(record)
(R/'numeric-claim-decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print('Reviewed numeric statements:',len(records),'direct/qualified:',sum(bool(r['proposed_parameter_ids']) for r in records))
