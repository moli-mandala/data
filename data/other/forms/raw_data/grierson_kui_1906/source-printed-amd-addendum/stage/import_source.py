"""Guarded reviewed Kui source-stage importer; no database generation."""
import importlib.util
from pathlib import Path
import argparse

def main(install=False):
    path=Path(__file__).resolve().parent/'source-printed-amd-addendum/apply_correction.py'
    spec=importlib.util.spec_from_file_location('kui_printed_amd_correction',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main(install=install)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true')
    main(parser.parse_args().install)
