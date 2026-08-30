#!/usr/bin/env python3
"""Verify the frozen Adi 2015 manual package before shared integration.

This audit never reads PDF text, OCR, legacy forms, or installed forms as
lexical evidence.  It proves that the already completed rendered-page manual
review regenerates the staged source package exactly, then freezes the
reproducible PDF-page render hashes and the integration-ready contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = HERE.parents[5]
PDF = WORKSPACE / "tmp/pdfs/adi_2015/silesr2015_016.pdf"
RENDER_ROOT = WORKSPACE / "tmp/pdfs/adi_2015/preintegration_render_400"
RENDER_MANIFEST = HERE / "render_hashes.tsv"
OUTPUT_MANIFEST = HERE / "preintegration_manifest.json"

PDF_SHA256 = "8e1500383a02445252a3eb6973a1b011fabea71eb25ad79fc43ba5b78bd1135c"
MANUAL_BUNDLE_SHA256 = "a9a1aac22c77c4cf66230c2fa014a7b151cd676db44810744b06748070bd92f0"
RENDER_MANIFEST_SHA256 = "85b5dee74f2614b3028c5dce9082476b55abd6f60e0c5a42a72ead9c28d84173"
RENDER_TREE_SHA256 = "0746c68daf48349570eb0d37e2d69afb79c22571d69a410484b8437e1efd794c"

FROZEN_HASHES = {
    "source_manifest.json": "b28c3fb704dd7a3db0dcfd7d894c761e974f96a7ecbdba5e779b4cf4563f9626",
    "list_registry.tsv": "9c93a777e2196c83a87ca91062b666d67dd785d7fc218c2f0f116e286fe5a3ca",
    "staged_forms.csv": "edb29a8f65fea0600e3d54bfcf2adef81fd833c47b619de5cd701bd61df4031c",
    "staged_audit.tsv": "6fb69a145419fff42c6b48d8e965acf2dbd9dc06bd297edf2e19f62e4f88877b",
    "unresolved_readings.tsv": "3a4e2be9c39e3b3852e455d59d7a077919e208d536b0c6673c2ec834f1f51803",
    "symbol_inventory.tsv": "a649621372743809a46b6391f11e4abf329016bdf84f8dfa175b94e1b330e886",
    "conversion_profile.tsv": "61f298367f3e9217c170797cc6c4dbebc3c4b86eb90936b3e2f52561ed013d71",
}

MANUAL_HASHES = {
    "items_001_012_hand_keyed.tsv": "f1d39778ec5b87379724c5dd5ffdc640ac94d6c217c0227370adc3f794429f29",
    "items_013_026_hand_keyed.tsv": "b81294c59a5999a29a7ba498df339c9bb175bb7c581a6a4641c530a0be45a7f1",
    "items_027_041_hand_keyed.tsv": "358eb79a623fe7b9f60d9bd53574d9e04f07d71d0c2b2e57d88dd303cd63817d",
    "items_042_055_hand_keyed.tsv": "246573e8c5d3e3e5f28215777794498f51d6efc497cde479ba15bc3768df0671",
    "items_056_070_hand_keyed.tsv": "f518737e8c013036dcab34fdbc01a873dd934eedb01d2a69eee5e91021d08b1b",
    "items_071_084_hand_keyed.tsv": "1bcd6916c5c7eccf9ba1da6386eb4356702d2b27d98fba0a4f64981ab790eef9",
    "items_085_099_hand_keyed.tsv": "9dc0d972349f54814b879351b16ad0fe8dda925faa8eaa2966f4025418e99e76",
    "items_100_114_hand_keyed.tsv": "51436daeeddbe2fba5345139f55130b9205824880bd51615de2853afdd86683f",
    "items_115_128_hand_keyed.tsv": "42925e199003791d7c9a8ee183e0196003cde6e92bcdc789d0db1d6708f3753b",
    "items_129_158_hand_keyed.tsv": "b9355819bff66ffa61f341afdf22d3cb7c201af2237d81ea817d9384cbf22fde",
    "items_159_188_hand_keyed.tsv": "df7c297e1bc79dea56cb8c592b4a5f27fa06a0358f60db81a884c43d0896fab6",
    "items_189_218_hand_keyed.tsv": "ed0aaf8828a3ca8f9a4a2cdc0b41a8afa49d0518765725baa14eafc3ac4b655c",
    "items_219_248_hand_keyed.tsv": "d4bc3075b00bb5e32f62c73af043166ee45a6fd7953af9669b0cbafca26f0466",
    "items_249_278_hand_keyed.tsv": "3187772971ff0b6eb2081627d798cbd6150e93e8570eaa06181fc8b1deaed2e5",
    "items_279_307_hand_keyed.tsv": "5fa34a79b4684c8bfeeb0e10786dbe14ba4994a1a2545e1898cd3403af127284",
}

SITE_ORDER = ["MN", "BR", "RM", "ML", "PL", "AS", "PD", "SM", "BK"]
SITE_STAGED_COUNTS = {
    "MN": 297, "BR": 304, "RM": 312, "ML": 298, "PL": 310,
    "AS": 309, "PD": 316, "SM": 306, "BK": 318,
}
LANGUAGE_STAGED_COUNTS = {
    "MisingPadamMiriMinyong": 613,
    "BoriKarko": 610,
    "BokarRamo": 1249,
    "Milang": 298,
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_importer():
    path = HERE / "import_adi_2015.py"
    spec = importlib.util.spec_from_file_location("sil_adi_2015_importer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def render_manifest_bytes() -> tuple[bytes, str, int]:
    expected = [f"page-{page}.png" for page in range(17, 39)]
    paths = sorted(RENDER_ROOT.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    assert [path.name for path in paths] == expected, (
        "expected the complete reproducible physical-page render set 17--38; "
        "rerun pdftoppm at 400 dpi"
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["Relative_Path", "Bytes", "SHA256"])
    tree_parts = []
    total_bytes = 0
    for path in paths:
        size = path.stat().st_size
        digest = sha256(path)
        total_bytes += size
        writer.writerow([path.name, size, digest])
        tree_parts.append(f"{path.name}\0{digest}\0{size}\n")
    content = stream.getvalue().encode("utf-8")
    tree_sha = sha256_bytes("".join(tree_parts).encode("utf-8"))
    assert sha256_bytes(content) == RENDER_MANIFEST_SHA256
    assert tree_sha == RENDER_TREE_SHA256
    assert total_bytes == 15_131_560
    return content, tree_sha, total_bytes


def verify_pdf() -> None:
    assert PDF.stat().st_size == 743_089
    assert sha256(PDF) == PDF_SHA256
    result = subprocess.run(
        ["pdfinfo", str(PDF)], capture_output=True, text=True, check=True
    )
    assert re.search(r"^Pages:\s+45$", result.stdout, re.MULTILINE)
    assert re.search(r"^Page size:\s+612 x 792 pts", result.stdout, re.MULTILINE)


def verify_frozen_files() -> dict[str, str]:
    for relative, expected in FROZEN_HASHES.items():
        assert sha256(HERE / relative) == expected, f"frozen artifact changed: {relative}"
    chunks = HERE / "manual_chunks"
    paths = sorted(chunks.glob("items_*_hand_keyed.tsv"))
    assert [path.name for path in paths] == list(MANUAL_HASHES)
    parts = []
    for path in paths:
        digest = sha256(path)
        assert digest == MANUAL_HASHES[path.name], f"manual chunk changed: {path.name}"
        parts.append(f"{path.name}\0{digest}\0{path.stat().st_size}\n")
    bundle = sha256_bytes("".join(parts).encode("utf-8"))
    assert bundle == MANUAL_BUNDLE_SHA256
    return {path.name: MANUAL_HASHES[path.name] for path in paths}


def verify_regeneration(importer):
    chunks = [HERE / "manual_chunks" / name for name in MANUAL_HASHES]
    rows = importer.load_manual_ledgers(chunks)
    registry = importer.load_registry(HERE / "list_registry.tsv")
    importer.require_full_review(rows)

    assert len(rows) == 2_763
    assert {(int(row["Item"]), row["Site_Code"]) for row in rows} == {
        (item, site) for item in range(1, 308) for site in SITE_ORDER
    }
    assert {int(row["PDF_Page"]) for row in rows} == set(range(17, 39))
    assert all(int(row["Printed_Page"]) == int(row["PDF_Page"]) - 4 for row in rows)
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=2_670, source_blank=93
    )
    assert not [row for row in rows if row["Review_Status"] in {"ambiguous", "illegible"}]

    forms, audit = importer.build_source_package(rows, registry)
    assert len(forms) == 2_770 and len(audit) == 2_763
    with tempfile.TemporaryDirectory(prefix="adi-preintegration-") as temp:
        temp_path = Path(temp)
        importer.write_source_package(forms, audit, temp_path)
        for name in (
            "staged_forms.csv", "staged_audit.tsv", "unresolved_readings.tsv",
            "symbol_inventory.tsv",
        ):
            assert (temp_path / name).read_bytes() == (HERE / name).read_bytes(), (
                f"independent source package no longer regenerates {name} exactly"
            )

    assert Counter(row["Language_ID"] for row in forms) == Counter(LANGUAGE_STAGED_COUNTS)
    assert len({row["Entry_Key"] for row in forms}) == 2_770
    by_site = Counter()
    citation_re = re.compile(
        r"^padung-sako2015adi\[Appendix B, printed p\. (\d+), "
        r"item (\d+), list ([A-Z]{2})\]$"
    )
    registry_by_site = {row["Site_Code"]: row for row in registry}
    for row in forms:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        assert "�" not in row["Form"] and row["Form"]
        match = citation_re.fullmatch(row["Source"])
        assert match
        printed_page, item, site = match.groups()
        by_site[site] += 1
        assert row["Entry_Key"].startswith(
            f"padung-sako2015adi:item:{int(item):03d}:site:{site}:response:"
        )
        assert row["Tags"] == registry_by_site[site]["Dialect_Tag"]
        assert row["Language_ID"] == registry_by_site[site]["Language_ID"]
        assert int(printed_page) in range(13, 35)
        assert not any(row[field] for field in (
            "Parameter_ID", "Native", "Phonemic", "Notes", "Cognateset",
            "Etymology", "Variant_Of_Key", "Borrowed_From_Key",
            "Derivation_Parent_Keys",
        ))
    assert by_site == Counter(SITE_STAGED_COUNTS)

    response_counts = Counter(
        len(row["Staged_Entry_Keys"].split(" | "))
        for row in audit if row["Review_Status"] == "attested"
    )
    assert response_counts == Counter({1: 2_573, 2: 94, 3: 3})
    blank_counts = Counter(
        row["Site_Code"] for row in audit if row["Review_Status"] == "source_blank"
    )
    assert blank_counts == Counter(
        MN=15, BR=16, RM=6, ML=14, PL=8, AS=9, PD=8, SM=10, BK=7
    )
    assert all(
        row["Disposition"] == (
            "staged" if row["Review_Status"] == "attested" else "blank-excluded"
        )
        for row in audit
    )
    return rows, forms, audit, registry


def verify_profile_inventory(forms) -> dict[str, int]:
    counts = Counter(char for row in forms for char in row["Form"])
    with (HERE / "symbol_inventory.tsv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream, delimiter="\t"))
    assert len(inventory) == 42
    assert {row["Symbol"]: int(row["Count"]) for row in inventory} == counts
    for row in inventory:
        char = row["Symbol"]
        assert row["Codepoint"] == f"U+{ord(char):04X}"
        assert row["Unicode_Name"] == unicodedata.name(char, "UNNAMED")
        assert row["Decision"].startswith("preserve")
    assert "�" not in counts
    assert counts["?"] == 2
    assert counts["̪"] > 0 and counts["̃"] > 0 and counts["ː"] > 0
    with (HERE / "conversion_profile.tsv").open(encoding="utf-8", newline="") as stream:
        profile = list(csv.DictReader(stream, delimiter="\t"))
    assert len({row["Grapheme"] for row in profile}) == len(profile)
    assert set(counts) <= {row["Grapheme"] for row in profile}
    assert next(row for row in profile if row["Grapheme"] == "?")["IPA"] == "?"
    assert next(row for row in profile if row["Grapheme"] == "̪")["IPA"] == ""
    return dict(sorted(counts.items(), key=lambda pair: ord(pair[0])))


def build_manifest(manual_hashes, render_tree, render_bytes, rows, forms, audit, registry):
    return {
        "state": "source-local-preintegration-audit-complete",
        "checklist": {
            "active": True,
            "addenda": [
                "Survey wordlists or comparative tables",
                "OCR-heavy source",
            ],
        },
        "source_key": "padung-sako2015adi",
        "pdf": {
            "path": "tmp/pdfs/adi_2015/silesr2015_016.pdf",
            "bytes": 743_089,
            "pages": 45,
            "sha256": PDF_SHA256,
        },
        "lexical_appendix": {
            "physical_pages": "17--38",
            "printed_pages": "13--34",
            "items": 307,
            "target_lists": 9,
            "conceptual_cells": 2_763,
            "control_lists": 0,
        },
        "manual_review": {
            "chunks": 15,
            "rows": len(rows),
            "bundle_sha256": MANUAL_BUNDLE_SHA256,
            "chunk_sha256": manual_hashes,
            "method": (
                "every cell manually viewed against a rendered source page; PDF text was "
                "character-input scaffold only and supplied or verified no accepted reading"
            ),
        },
        "statuses": {
            "attested": 2_670,
            "source_blank": 93,
            "ambiguous": 0,
            "illegible": 0,
            "unresolved": 0,
            "unreviewed": 0,
        },
        "staging": {
            "rows": len(forms),
            "audit_rows": len(audit),
            "unique_entry_keys": len({row["Entry_Key"] for row in forms}),
            "forms_sha256": FROZEN_HASHES["staged_forms.csv"],
            "audit_sha256": FROZEN_HASHES["staged_audit.tsv"],
            "unresolved_sha256": FROZEN_HASHES["unresolved_readings.tsv"],
            "site_rows": SITE_STAGED_COUNTS,
            "language_rows": LANGUAGE_STAGED_COUNTS,
            "expanded_attested_cells": {"one_response": 2_573, "two_responses": 94, "three_responses": 3},
        },
        "renders": {
            "dpi": 400,
            "physical_pages": "17--38",
            "artifacts": 22,
            "bytes": render_bytes,
            "manifest_sha256": RENDER_MANIFEST_SHA256,
            "tree_sha256": render_tree,
        },
        "profile_inventory": {
            "path": "symbol_inventory.tsv",
            "sha256": FROZEN_HASHES["symbol_inventory.tsv"],
            "source_local_profile": "conversion_profile.tsv",
            "source_local_profile_sha256": FROZEN_HASHES["conversion_profile.tsv"],
            "symbols": 42,
            "replacement_characters": 0,
            "literal_source_question_marks": 2,
            "decision": "lossless NFC diplomatic preservation before any shared display conversion",
        },
        "registry_contract": {
            "base_languages": [
                "MisingPadamMiriMinyong", "BoriKarko", "BokarRamo", "Milang"
            ],
            "dialect_ids": [row["Dialect_ID"] for row in registry],
            "dialect_tags": [row["Dialect_Tag"] for row in registry],
            "site_codes": SITE_ORDER,
            "site_coordinates": "blank because the source supplies no point coordinates",
            "shared_registry_state_at_audit": "all four base-language IDs and all nine dialect IDs absent",
        },
        "integration_contract": {
            "install_exactly": 2_770,
            "exclude_exactly": {"source_blank_cells": 93, "controls": 0, "unresolved": 0},
            "installed_filename": "data/other/forms/20260829-sil-adi.csv",
            "reference_key": "padung-sako2015adi",
            "profile_rule": (
                "route only the immutable source key; preserve diplomatic Original and map "
                "display sequences explicitly from the complete inventory"
            ),
            "etymology": "none; all Parameter_ID values remain blank",
        },
        "frozen_artifacts": FROZEN_HASHES,
        "deferred_gates": [
            "shared installed CSV",
            "shared bibliography and formatted reference",
            "shared language/dialect registries",
            "shared sound-profile file and source-key route",
            "focused shared registry/profile tests",
            "consolidated CLDF/full build and compiled survival checks",
            "global source-audit regeneration",
            "browser database refresh and browser QA",
            "commit and push",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write deterministic audit outputs")
    args = parser.parse_args()

    verify_pdf()
    manual_hashes = verify_frozen_files()
    render_content, render_tree, render_bytes = render_manifest_bytes()
    importer = load_importer()
    rows, forms, audit, registry = verify_regeneration(importer)
    verify_profile_inventory(forms)
    manifest = build_manifest(
        manual_hashes, render_tree, render_bytes, rows, forms, audit, registry
    )
    manifest_content = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    if args.write:
        RENDER_MANIFEST.write_bytes(render_content)
        OUTPUT_MANIFEST.write_bytes(manifest_content)
    else:
        assert RENDER_MANIFEST.read_bytes() == render_content, "render manifest is stale"
        assert OUTPUT_MANIFEST.read_bytes() == manifest_content, "preintegration manifest is stale"

    print(
        "cells=2763 attested=2670 blanks=93 staged=2770 unresolved=0 "
        f"manual_bundle_sha256={MANUAL_BUNDLE_SHA256} "
        f"render_tree_sha256={render_tree}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
