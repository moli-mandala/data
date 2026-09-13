"""Replay the pinned Zoller 2023 snapshot, or decode the exact user-provided PDF again.

  uv run python data/other/forms/raw_data/zoller_2023.py --install
  uv run --with pymupdf python data/other/forms/raw_data/zoller_2023.py --extract --pdf /path/Zoller.pdf

A draft is the default. --remap regenerates language proposals for editorial review;
it must not be combined with --install. Ordinary replay uses the reviewed mapping.
"""
import argparse,subprocess,sys
from pathlib import Path
RAW=Path(__file__).with_suffix('')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extract',action='store_true');p.add_argument('--pdf',type=Path)
    p.add_argument('--install',action='store_true');p.add_argument('--remap',action='store_true')
    a=p.parse_args()
    def run(name,*args):subprocess.run([sys.executable,str(RAW/name),*map(str,args)],check=True)
    if a.remap and a.install:p.error('Review newly proposed language mappings before installing.')
    if a.extract:
        if not a.pdf or not a.pdf.is_file():p.error('--extract requires the original Zoller.pdf via --pdf; it is not redistributed.')
        run('extract.py',a.pdf,'--output',RAW/'pages.jsonl.gz')
        run('table.py',a.pdf);run('records.py')
    if a.remap:run('parse.py');run('map_registry.py')
    run('build.py',*(['--install'] if a.install else []))
if __name__=='__main__':main()
