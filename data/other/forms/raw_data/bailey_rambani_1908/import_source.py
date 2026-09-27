"""Install Bailey’s full Rambani chapter; source-only, no database generation."""
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location('rambani_full_import',Path(__file__).with_name('import_source_full.py'))
_mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_mod)
generate=_mod.generate
read_source=_mod.read_source
if __name__=='__main__':_mod.main()
