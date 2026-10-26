"""Exercise the real generic form parser in a source-local isolated fixture."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import unicodedata as ud
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
sys.path.insert(0, str(DATA))
import make_cldf  # noqa: E402
import source_meta  # noqa: E402

CSV = HERE / "review-candidate.csv"
AUDIT = HERE / "review-audit.jsonl"
SETTINGS = HERE / "proposed-settings.yaml"
FIXTURE = HERE / "parser-fixture" / "data" / "other" / "forms"
STEM = "20260927-nair-kani"
FIELDS = ("lang", "param", "old_form", "gloss", "native", "ipa", "notes", "source", "cognateset", "etymology", "entry_key", "variant_of_key", "borrowed_from_key", "derivation_parent_keys", "tags")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pins = {p.name: sha(p) for p in (CSV, AUDIT, SETTINGS)}
    with CSV.open(newline="", encoding="utf-8") as inp:
        source_rows = list(csv.reader(inp))
    assert len(source_rows) == 36
    FIXTURE.mkdir(parents=True, exist_ok=True)
    fixture_yaml = FIXTURE / f"{STEM}.yaml"
    shutil.copyfile(SETTINGS, fixture_yaml)
    source_meta.yaml_files = lambda: [fixture_yaml]
    source_meta.load.cache_clear()
    checks: dict[str, dict] = {}
    for normalization in ("NFC", "NFD"):
        input_rows = [[ud.normalize(normalization, value) if i == 2 else value for i, value in enumerate(row)] for row in source_rows]
        fixture_csv = FIXTURE / f"{STEM}.csv"
        with fixture_csv.open("w", newline="", encoding="utf-8") as out:
            csv.writer(out).writerows(input_rows)
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(fixture_csv), errors, name=STEM)
        assert not errors.getvalue(), errors.getvalue()
        assert len(parsed) == len(input_rows)
        assert stats == {"converted": 0, "for_conversion": 0}, stats
        for actual, expected in zip(parsed, input_rows):
            assert [getattr(actual, field) for field in FIELDS] == expected, actual.entry_key
            assert ud.normalize("NFC", actual.form) == ud.normalize("NFC", expected[2])
        checks[normalization] = {"rows": len(parsed), "ordered_keys_exact": True, "all_15_source_fields_exact": True, "errors": 0}
    assert pins == {p.name: sha(p) for p in (CSV, AUDIT, SETTINGS)}
    report = {
        "status": "passed isolated actual parser; review candidate uninstalled",
        "inputs_sha256": pins,
        "make_cldf_sha256": sha(DATA / "make_cldf.py"),
        "checks": checks,
        "registry_status": "Kanikkaran and source-qualified Kerala Kani dialect are only proposed; this parser fixture does not claim canonical language/dialect registration",
    }
    (HERE / "actual-parser-review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["status"])


if __name__ == "__main__":
    main()
