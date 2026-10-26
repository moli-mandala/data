"""The old wordlist installer must not overwrite whole-report recovery."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


DATA = Path(__file__).resolve().parents[1]
IMPORTER = DATA / "data/other/forms/raw_data/sil_korwa_kodaku_2022/import_korwa_kodaku.py"


def load_importer():
    spec = importlib.util.spec_from_file_location("legacy_korwa_importer_guard_test", IMPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_install_requires_unmodified_historical_output(monkeypatch):
    module = load_importer()
    with TemporaryDirectory(dir=DATA.parent / "tmp") as scratch:
        root = Path(scratch)
        output = root / "legacy.csv"
        output.write_text("historical fixture\n")
        monkeypatch.setattr(module, "OUTPUT", output)
        monkeypatch.setattr(module, "WHOLE_RECOVERY", root / "whole-recovery")
        monkeypatch.setattr(module, "sha256", lambda path: module.HISTORICAL_OUTPUT_SHA256)
        module.assert_legacy_install_safe()
        monkeypatch.setattr(module, "sha256", lambda path: "different output")
        with pytest.raises(RuntimeError, match="refusing legacy overwrite"):
            module.assert_legacy_install_safe()
        output.unlink()
        with pytest.raises(RuntimeError, match="refusing legacy overwrite"):
            module.assert_legacy_install_safe()


@pytest.mark.parametrize("marker", [
    "installation/root-acceptance.json",
    "installation-transaction.json",
    "installation-receipt.json",
])
def test_legacy_install_refuses_whole_recovery_states(monkeypatch, marker):
    module = load_importer()
    with TemporaryDirectory(dir=DATA.parent / "tmp") as scratch:
        root = Path(scratch)
        output = root / "legacy.csv"
        output.write_text("historical fixture\n")
        recovery = root / "whole-recovery"
        path = recovery / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        monkeypatch.setattr(module, "OUTPUT", output)
        monkeypatch.setattr(module, "WHOLE_RECOVERY", recovery)
        monkeypatch.setattr(module, "sha256", lambda path: module.HISTORICAL_OUTPUT_SHA256)
        with pytest.raises(RuntimeError, match="guarded installer"):
            module.assert_legacy_install_safe()
