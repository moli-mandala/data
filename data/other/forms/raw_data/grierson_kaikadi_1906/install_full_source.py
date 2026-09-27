"""Install approved whole-source inputs only; never build the database."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P.parents[4]
ARTIFACTS=['proposal.csv','proposal-audit.jsonl','proposal-profile.txt','20260925-grierson-kaikadi.yaml','source-proposed.bib','dialects-proposed.csv','approval-input-manifest.json']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(report_name):
    report=json.loads((P/report_name).read_text())
    assert report['status']=='passed' and report['sample_size']==20 and report['material_errors']==0
    for name in ARTIFACTS:assert digest(P/name)==report['hashes'][name],name
    inputs=json.loads((P/'approval-input-manifest.json').read_text())
    for name,expected in inputs['hashes'].items():assert digest(P/name)==expected,name
    # Check review inputs as well as their key coverage and final artifact hashes.
    for filename in ['independent-full-specimens-review-20260926.json','independent-specimens-p339-342-review-20260926.json','independent-full-addenda-review-20260926.json']:
        review=json.loads((P/filename).read_text())
        assert digest(P/review['input'])==review['input_sha256'],filename
    grammar_review=json.loads((P/'independent-full-grammar-outside-review-20260926.json').read_text())
    for name,expected in grammar_review['input_sha256'].items():assert digest(P/name)==expected,name
    table_review=json.loads((P/'independent-full-table-review-20260926.json').read_text())
    assert digest(P/table_review['input_file'])==table_review['input_sha256']
    table={a['prompt']:a for a in map(json.loads,(P/'table-reviewed.jsonl').read_text().splitlines())}
    for correction in table_review['corrections']:
        field=correction.get('field','forms')
        assert table[correction['prompt']][field]==correction['after'],correction
    final={a['source_unit_key']:a for a in map(json.loads,(P/'specimens-final-reviewed.jsonl').read_text().splitlines())}
    reconciliation=json.loads((P/'independent-specimen-reconciliation.json').read_text())
    for name,expected in reconciliation['input_hashes'].items():assert digest(P/name)==expected,name
    assert digest(P/reconciliation['output'])==reconciliation['output_sha256']
    for correction in reconciliation['corrections']:assert final[correction['key']]['forms']==correction['after']
    rows=list(csv.reader((P/'proposal.csv').open()))
    audit=[json.loads(x)for x in(P/'proposal-audit.jsonl').read_text().splitlines()]
    summary=json.loads((P/'proposal-summary.json').read_text())
    assert len(rows)==summary['proposal_rows'] and len(audit)==1087
    assert len({r[10]for r in rows})==len(rows) and all(len(r)==15 for r in rows)
    legacy=json.loads((P/'legacy164-preservation-manifest.json').read_text())
    assert len(legacy['entry_keys'])==164 and set(legacy['entry_keys'])<={r[10]for r in rows}
    assert sum(a['section'].startswith('specimen') for a in audit)==706
    assert sum(a['section'].startswith('standard') for a in audit)==241
    assert all(a['entry_keys']for a in audit if a.get('prompt',0)>=220)
    assert sum(a['scope']!='target' for a in audit)==51
    covered=set()
    for name,field in [('independent-full-table-review-20260926.json','reviewed_keys'),('independent-full-addenda-review-20260926.json','reviewed_keys'),('independent-full-specimens-review-20260926.json','reviewed_keys'),('independent-specimens-p339-342-review-20260926.json','coverage'),('independent-full-grammar-outside-review-20260926.json','covered')]:
        evidence=json.loads((P/name).read_text())
        covered.update(x if isinstance(x,str) else x['source_unit_key'] for x in evidence[field])
    covered.update(report.get('additional_reviewed_source_unit_keys',[]))
    assert covered=={a['source_unit_key']for a in audit}, 'Full independent source coverage required'
    return rows,audit,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--approval-report',default='independent-full-audit-20260926-pass1.json')
    args=parser.parse_args();rows,audit,report=validate(args.approval_report)
    if args.install:
        for source,destination in [('proposal.csv',DATA/'data/other/forms/20260925-grierson-kaikadi.csv'),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/grierson-kaikadi-1906.txt'),('20260925-grierson-kaikadi.yaml',DATA/'data/other/forms/20260925-grierson-kaikadi.yaml')]:
            shutil.copyfile(P/source,destination)
        manifest={'stage':'installed source inputs only; database/build/browser gates deferred by explicit user instruction','forms':len(rows),'physical_units':len(audit),'legacy_keys_preserved':164,'approval_report':args.approval_report,'hashes':report['hashes']}
        (P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(rows)} source rows and {len(audit)} physical units verified; no database build')
if __name__=='__main__':main()
