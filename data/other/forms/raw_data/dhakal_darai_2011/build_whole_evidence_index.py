"""Build a source-unit index for whole-thesis review, never canonical output.

Keep paths and record keys rather than duplicating original source evidence. No
source unit is dropped for missing gloss, unusual symbols, or being a morpheme.
"""
import collections
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCES=['chapters12-proposal-audit.jsonl','clause-recovery-proposal-audit.jsonl',
 'chapter3-reviewed-expression-records.jsonl','chapter3-additional-attestations-reviewed.jsonl','chapter4-adjudicated-records.jsonl',
 'chapter4-interlinear-reviewed.jsonl']+[f'chapter{n}-reviewed-records.jsonl' for n in range(5,14)]+[
 'appendix3a-reviewed-records.jsonl','appendix3b-reviewed-records.jsonl',
 'appendix3c1-table-reviewed-records.jsonl','appendix3c2-table-reviewed-records.jsonl',
 'appendix3c1-expression-reviewed-records.jsonl','appendix-small-tables-reviewed-records.jsonl',
 'appendix-copula-examples-reviewed-records.jsonl','appendix-heading-attestations.jsonl',
 'appendix4-morphemes-reviewed-records.jsonl','appendix5-reviewed-records.jsonl',
 'front-index-attestations-reviewed.jsonl']

def build():
    keys=set();rows=[];inputs={}
    for name in SOURCES:
        path=P/name;inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for number,line in enumerate(path.open(),1):
            r=json.loads(line)
            key=r.get('source_unit_key',r.get('key'))
            assert key and key not in keys,(name,number,key)
            keys.add(key)
            old=r.get('output')
            form=r.get('form',old[2] if old else r.get('parsed',{}).get('form',''))
            gloss=r.get('gloss',r.get('candidate_gloss',old[3] if old else r.get('parsed',{}).get('gloss','')))
            rows.append(dict(source_unit_key=key,evidence_file=name,evidence_line=number,
              printed_page=r.get('printed_page'),pdf_page=r.get('pdf_page'),
              kind=r.get('kind',r.get('record_type','prior-proposal-attestation')),
              form=form,gloss=gloss,source_status=r.get('status'),
              parent_expression_key=r.get('parent_expression_key'),
              disposition=r.get('disposition'),
              source_reuse_key=r.get('reuse_source_unit_key'),
              integration_status='pending whole-source reconciliation; not an accepted export'))
    return rows,inputs

if __name__=='__main__':
    rows,inputs=build()
    f=P/'whole-evidence-index.jsonl'
    f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary=dict(status='Complete current evidence index; not a lexical output count',
      units=len(rows),files=len(inputs),inputs=inputs,index_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
      kinds=dict(collections.Counter(r['kind'] for r in rows)),
      unresolved='All current additions indexed; unit metadata, source claims, morphology alternates and exact reuse remain under review.')
    (P/'whole-evidence-index-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(summary['units'],'indexed source records from',summary['files'],'files; no canonical output')
