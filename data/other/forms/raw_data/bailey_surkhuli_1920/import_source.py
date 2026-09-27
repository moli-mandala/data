"""Regenerate the independently reviewed whole Surkhuli source stage."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json
PACKAGE=Path(__file__).resolve().parent
DATA=PACKAGE.parents[4]
OUTPUT=DATA/'data/other/forms/20260925-bailey-surkhuli.csv'
AUDIT=PACKAGE/'audit.jsonl'

def generate():
    spec=importlib.util.spec_from_file_location('surkhuli_whole_prepare',PACKAGE/'prepare_whole_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,audit,_=module.prepare()
    assert rows==list(csv.reader((PACKAGE/'whole-proposed.csv').open()))
    assert audit==[json.loads(x) for x in (PACKAGE/'whole-proposed-audit.jsonl').read_text().splitlines()]
    return rows,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--check-pdf',action='store_true')
    args=parser.parse_args()
    if args.check_pdf:
        path=DATA.parent/'tmp/pdfs/bailey1920/bailey1920.pdf'
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest=='7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39'
    rows,audit=generate()
    if args.install:
        OUTPUT.write_bytes((PACKAGE/'whole-proposed.csv').read_bytes())
        AUDIT.write_bytes((PACKAGE/'whole-proposed-audit.jsonl').read_bytes())
    print(f'{len(rows)} forms, {len(audit)} source units; source stage only')
if __name__=='__main__':main()
