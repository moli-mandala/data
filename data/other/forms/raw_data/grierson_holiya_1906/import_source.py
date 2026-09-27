"""Install independently approved complete Holiya source stage; no database build."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P.parents[4]

def validate():
    report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
    assert report['status']=='passed' and report['sample_size']==20 and report['material_errors']==0
    for filename,digest in report['hashes'].items():
        assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest,filename
    rows=list(csv.reader((P/'proposal.csv').open()));audit=[json.loads(x)for x in(P/'proposal-audit.jsonl').read_text().splitlines()]
    assert len(rows)==900 and len(audit)==1282
    assert len({r[10]for r in rows})==900
    covered=set()
    for f in P.glob('independent-*-review-20260926.json'):
        covered.update(json.loads(f.read_text()).get('coverage_source_unit_keys',[]))
    assert covered=={a['source_unit_key']for a in audit}
    return rows,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true');args=parser.parse_args()
    rows,audit=validate()
    if args.install:
        for source,destination in [('proposal.csv',DATA/'data/other/forms/20260926-grierson-holiya.csv'),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/grierson-holiya-1906.txt'),('20260926-grierson-holiya.yaml',DATA/'data/other/forms/20260926-grierson-holiya.yaml')]:
            shutil.copyfile(P/source,destination)
        manifest={'stage':'installed source inputs only; full build deferred by user','forms':len(rows),'physical_units':len(audit),'candidate_forms_before_exact_reuse':1253,'exact_reused_rows':353,'controls_or_bound_morphology':37,'source_glyph_uncertainties':1,'source_editor_quantity_warning':True,'independent_sample':{'checked':20,'material_errors':0},'hashes':json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())['hashes']}
        (P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('900 source rows and1282 audit units verified; no database build')

if __name__=='__main__':main()
