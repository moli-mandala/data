"""Canonical whole-source Bhadrawahi importer."""
import importlib.util
from pathlib import Path
_p=Path(__file__).with_name('import_source_full.py')
_s=importlib.util.spec_from_file_location('bhadrawahi_full_impl',_p)
_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)
generate=_m.generate
read_source=_m.read_source
if __name__=='__main__':_m.main()
