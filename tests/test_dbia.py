import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DBIA = ROOT / "data" / "dbia"


def load_parser():
    spec = importlib.util.spec_from_file_location("dbia_parse", DBIA / "parse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dbia_extraction_is_conservative_and_audited():
    parser = load_parser()
    entries = parser.extract_entries((DBIA / "dbia.txt").read_text(encoding="utf-8"))
    numbers = [number for number, _letter, _page, _text in entries]

    assert numbers == sorted(set(numbers))
    assert numbers[0] == 1
    assert numbers[-1] == 337
    assert len(entries) == 337

    with (DBIA / "parse_audit.csv").open(encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    missing = {row["Number"] for row in audit if row["Decision"] == "OCR entry boundary not recoverable"}
    assert not missing
    assert {"dbia18", "dbia22", "dbia40", "dbia43", "dbia51", "dbia300"} <= {
        row["DBIA_ID"] for row in audit
    }


def test_dbia_cdial_redirects_are_unique_and_valid():
    with (DBIA / "cdial_redirects.csv").open(encoding="utf-8") as handle:
        redirects = list(csv.DictReader(handle))
    with (ROOT / "data" / "cdial" / "params.csv").open(encoding="utf-8") as handle:
        cdial_ids = {row[0] for row in csv.reader(handle)}

    assert len(redirects) >= 100
    assert len({row["DBIA_ID"] for row in redirects}) == len(redirects)
    assert {row["CDIAL_ID"] for row in redirects} <= cdial_ids
    assert all(row["Reason"] in {"unique normalized headword", "homonym disambiguated by gloss"} for row in redirects)
