"""Install the independently reviewed full Bhalesi source stage; never build a database."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,shutil
P=Path(__file__).resolve().parent
DATA=P.parents[4]
spec=importlib.util.spec_from_file_location('bhalesi_full_import',P/'prepare_full.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
generate=full.generate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--check-pdf',action='store_true')
    args=parser.parse_args()
    if args.check_pdf:
        with (DATA.parent/'tmp/pdfs/bailey-sainji/bailey1908.pdf').open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()=='953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5'
    rows,audit=generate()
    assert rows==list(csv.reader((P/'proposal.csv').open()))
    assert audit==[json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]
    if args.install:
        report=json.loads((P/'independent-full-audit-20260926-pass3.json').read_text())
        assert report['status']=='passed' and report['sample_size']>=20 and report['material_errors']==0
        for name,digest in report['hashes'].items():
            assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
        for source,target in [('proposal.csv',DATA/'data/other/forms/20260925-bailey-bhalesi.csv'),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/bailey-bhalesi-1908.txt')]:
            shutil.copyfile(P/source,target)
    print(f'{len(rows)} rows; {len(audit)} editorial source units; source-stage only')

if __name__=='__main__':main()
