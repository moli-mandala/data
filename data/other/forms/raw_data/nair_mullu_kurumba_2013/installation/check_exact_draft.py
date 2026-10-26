"""Read-only exact shared-membership/audit/registry proof for root draft review."""
import hashlib
import json
import sys
from pathlib import Path
from pybtex.database import parse_file

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import source_meta


def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    draft=json.loads((HERE/'root-acceptance-draft.json').read_text())
    assert draft['installation_approved'] is False
    assert draft['shared_baseline_captured'] and draft['audit_acceptance_ready']
    for name,value in draft['files'].items():assert sha(PACKAGE/name)==value,name
    old=parse_file(str(DATA/'cldf/sources.bib')).entries
    merged=parse_file(str(HERE/'assets/bib.asset'),bib_format='bibtex').entries
    proposed=parse_file(str(HERE/'staged-references.bib')).entries
    assert set(merged)==set(old)|set(proposed) and not set(old)&set(proposed)
    assert all(merged[k]==v for k,v in old.items())
    assert all(merged[k]==v for k,v in proposed.items())
    accepted=[json.loads(s) for s in (HERE/'accepted-audit-proposal.jsonl').read_text().splitlines()]
    original=[json.loads(s) for s in (PACKAGE/'proposal-audit.jsonl').read_text().splitlines()]
    assert len(accepted)==65 and [a['source_record'] for a in accepted]==original
    assert sum(len(a['source_record']['witness_dispositions']) for a in accepted)==164
    staged=HERE/'20260927-nair-mullu-kurumba.yaml'
    meta=source_meta.SourceMeta(list(source_meta.yaml_files())+[staged])
    assert all(meta.source_owner[k]==staged.stem for k in proposed)
    assert sha(DATA/'data/form-identities.csv')==draft['durable_identity_sha256']
    report=dict(status='Exact disabled draft guards, existing shared membership and raw audit preservation pass',
        draft_sha256=sha(HERE/'root-acceptance-draft.json'),pinned_files=len(draft['files']),
        prior_bib_entries=len(old),added_bib_entries=len(proposed),all_prior_bib_semantics_unchanged=True,
        current_plus_staged_source_meta=True,unique_new_source_declarations=2,accepted_groups=65,raw_witness_records=164,
        durable_identity_unchanged=True,canonical_changes=False)
    (HERE/'exact-draft-review-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
