"""Import the independently audited whole Samuells Juang article; no database build."""
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location('samuells_full_import', Path(__file__).with_name('prepare_full.py'))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)
build = _full.build
if __name__ == '__main__':
    _full.main()
