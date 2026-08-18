import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/legacy_snapshots.py"
SPEC = importlib.util.spec_from_file_location("legacy_snapshots", SCRIPT)
legacy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)


def test_legacy_snapshots_are_pinned_rich_and_reproducible():
    for spec in legacy.SPECS.values():
        rows = legacy.rich_rows(spec)
        assert rows
        assert all(len(row) == legacy.RICH_COLUMNS for row in rows)
        assert all(row[2] for row in rows)
        assert len({row[10] for row in rows}) == len(rows)
        legacy.process(spec, install=False, check=True)
