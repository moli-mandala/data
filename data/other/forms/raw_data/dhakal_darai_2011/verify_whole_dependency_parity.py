"""Check the proposed dependency-only patch after approval, before installation.

--regenerate rewrites source-local proposals only, never canonical data. Use from
an owned local tmux job. Default mode checks already-produced bytes read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
P=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--regenerate',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((P/'whole-legacy-baseline-manifest-20260927.json').read_text())
    for name,info in manifest['files'].items():
        assert hashlib.sha256((P/name).read_bytes()).hexdigest()==info['sha256'],name
    if args.regenerate:
        assert (P/'whole_legacy_baseline.py').exists(), 'Apply the reviewed dependency patch first.'
        for script in ['prepare_whole_transcription.py','prepare_whole_csv_preview.py',
                       'prepare_whole_reuse_candidates.py','prepare_whole_staging.py']:
            source=(P/script).read_text()
            assert '20260921-dhakal-darai.csv' not in source,script
            subprocess.run([sys.executable,str(P/script)],check=True)
    for name,expected in manifest['frozen_output_hashes'].items():
        actual=hashlib.sha256((P/name).read_bytes()).hexdigest()
        assert actual==expected,(name,expected,actual)
    print('PASS: immutable 485-row baseline and profile; all four frozen output files are byte-identical.')


if __name__=='__main__':
    main()
