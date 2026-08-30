#!/usr/bin/env python3
"""Verify the frozen Noira 2015 source-local integration contract.

The independently hand-keyed ledgers are hash-checked before any comparison
with ESR 2013-004.  OCR/PDF text, legacy data, and installed forms are never
used to supply or verify a lexical reading.  The Dhule artifact is consulted
only after the Noira freeze, to document the report-identified republication.
This script writes source-local audit artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path

import import_noira_2015 as noira


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
PDF = WORKSPACE_ROOT / "tmp/pdfs/noira_2015/silesr2015_012.pdf"
RENDER_HASHES = HERE / "render_hashes.tsv"
PROFILE_INVENTORY = HERE / "profile_inventory.tsv"
PREINTEGRATION_MANIFEST = HERE / "preintegration_manifest.json"

PDF_HASH = "cb93db089a21e55e878f436632d8282c64c98fca85afe18179f8f3383db35280"
PDF_BYTES = 1_676_716
PDF_PAGES = 96
PDF_TITLE = "Noira Bhils and a Few Other Groups: A Sociolinguistic Study"
PDF_AUTHOR = "Bezily P. Varghese and Sunil Kumar D."

FROZEN_HASHES = {
    "manual_chunks/items_001_027_hand_keyed.tsv": "803e48195031524a0fb94ce0f0fb57483502bb5e3f9e4f78d476fc098204d5cc",
    "manual_chunks/items_028_054_hand_keyed.tsv": "57c27027fdd4972c1b93db6fb09fef88699aa9e5bdb3dd9e6d64a1df05104b11",
    "manual_chunks/items_055_081_hand_keyed.tsv": "30a6087035d3c00a682f2e90c473da7aca46e42f4b97f5367b4132f7bda68e98",
    "manual_chunks/items_082_108_hand_keyed.tsv": "ff1d6f982dcca3a1b59400c1064a913f1fedbbe824af1d35f43d0f17a061d9fe",
    "manual_chunks/items_109_135_hand_keyed.tsv": "8d48dd1c62119aae6b01e88f783e8d650d59cda1ffc81f1933aab03efc056f0e",
    "manual_chunks/items_136_162_hand_keyed.tsv": "e19acd41004baf71e1ad00c32097a5cfc0cfb0aed96cdc8b72b6d84cee470be4",
    "manual_chunks/items_163_189_hand_keyed.tsv": "e462c7cea69be36036982fe82f7f8eca0121747cb6125f6437354db5bd4161f4",
    "manual_chunks/items_190_210_hand_keyed.tsv": "6f018d53bf9128362fb6aad3fe51f7ee37b24c171a3f132a19ef9b4ca56c233e",
    "manual_chunks/hand_keyed_items_001_027.py": "d8fdbdfd16b0282640bfbc112afd538f93de43adc6c20243e6f886954f94fef8",
    "manual_chunks/hand_keyed_items_028_054.py": "fc731f594047ffe182dcc9433c4f82d92b2aa0073d25dd546582f99929a728c4",
    "manual_chunks/hand_keyed_items_055_081.py": "1b5321db6615a13deea9e23f3cf1e12af0292397d6cd83759bec3b77421203e1",
    "manual_chunks/hand_keyed_items_082_108.py": "af881a4ef3864db2ad96f1c6bbe332a07af8ca521339a3b489b761419911a721",
    "manual_chunks/hand_keyed_items_109_135.py": "3aeeb33e548fba5d1b1885fb1408bd506735b87639624a18f971c4f7c885e8da",
    "manual_chunks/hand_keyed_items_136_162.py": "eae670e8e9229c53be386779744110ed83cb9a43914c110fd197695462b922ab",
    "manual_chunks/hand_keyed_items_163_189.py": "ee9bc36934c2c5e70945d14cfdd8aa876c3ace704219d548f9ee34be06b14169",
    "manual_chunks/hand_keyed_items_190_210.py": "c5bdb8094658bfd47e726103ece57179e8ed6aa2159edc95b0ff17ed1c26d673",
    "staged_forms.csv": "c82983a319d6d6fbf5c07063f0655ae3e4e8e3890d625e1bfc2a38f95c811746",
    "exhaustive_audit.tsv": "fba530368ca1a982fad4c7bc3d53ee3a418b518123f33897e5630a675a0faec2",
    "dhule_republication_reconciliation.tsv": "fe0a636c70a7979921b7f5f107a84a7279a1654f24a9aeb964a1be26f0960ee1",
    "list_registry.tsv": "00254725d94e63af0ee9d2036fe302bdef4b5a515182d5ab45ccf05a72768de3",
    "conversion_profile.tsv": "3932523f127f4a13a94915dbd88bc21d2cac5867138bec7f2ce03a061e7f0de5",
    "unresolved_readings.tsv": "21e62e44fd2a03f5bc96a7921e192f55f66e52024a569acfca5ecf0dd6255ba9",
    "import_noira_2015.py": "caca39c1c0b366917cfe8da1f90c311e9633994e774135d535bea0cdbef5743c",
    "source_manifest.json": "bc91f2a4587d2fb28f60731b7d6477ffebf088e18696c4759b5d81c5ead21e0c",
}

TARGET_CODES = {"NCH", "NPN", "NGO", "DBM", "DBA", "KNA", "KTA", "GTA", "NTE", "TKO", "NJA"}
REPUBLISHED_CODES = {"NAS", "BMU", "NTO"}
CONTROL_CODES = {"GUJ", "MAR", "HIN"}
TARGET_LANGUAGES = {"Noiri", "DungraBhili", "Goj", "ko", "Ni"}
RENDER_FIELDS = [
    "Physical_PDF_Page", "Printed_Page", "Relative_Path", "Bytes", "Width",
    "Height", "DPI", "SHA256", "Evidence_Class",
]
PROFILE_FIELDS = [
    "Grapheme", "IPA", "Staged_Input_Occurrences", "Present_In_Staged_Targets",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dicts(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def tsv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def bundle_hash(paths: list[Path]) -> str:
    body = "".join(
        f"{path.relative_to(HERE).as_posix()}\t{sha256(path)}\n" for path in sorted(paths)
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def pdf_metadata() -> dict[str, object]:
    if PDF.stat().st_size != PDF_BYTES or sha256(PDF) != PDF_HASH:
        raise ValueError("Canonical ESR 2015-012 PDF hash/size changed")
    output = subprocess.run(
        ["pdfinfo", str(PDF)], check=True, capture_output=True, text=True
    ).stdout
    info = dict(
        line.split(":", 1) for line in output.splitlines() if ":" in line
    )
    info = {key.strip(): value.strip() for key, value in info.items()}
    if info.get("Title") != PDF_TITLE or info.get("Author") != PDF_AUTHOR:
        raise ValueError("Canonical PDF title/author metadata changed")
    if int(info.get("Pages", "0")) != PDF_PAGES:
        raise ValueError("Canonical PDF page count changed")
    if not info.get("Page size", "").startswith("612 x 792 pts"):
        raise ValueError("Canonical PDF page geometry changed")
    return {
        "path": PDF.relative_to(WORKSPACE_ROOT).as_posix(),
        "bytes": PDF_BYTES,
        "pages": PDF_PAGES,
        "sha256": PDF_HASH,
        "title": PDF_TITLE,
        "author": PDF_AUTHOR,
        "page_size_points": "612 x 792",
    }


def verify_frozen_package() -> tuple[list[dict[str, str]], list[dict[str, str]], list[list[str]]]:
    for relative, expected in FROZEN_HASHES.items():
        actual = sha256(HERE / relative)
        if actual != expected:
            raise ValueError(f"Frozen source-local artifact changed: {relative}: {actual}")

    rows = noira.load_all_manual_cells()
    registry = noira.load_registry()
    forms, audit, counts = noira.build_package(rows, registry)
    noira.validate_profile(forms)
    reconciliation = noira.build_republication_audit(rows)
    if counts != {
        "reviewed_cells": 3570, "attested_cells": 3526,
        "source_blank_cells": 44, "ambiguous_cells": 0, "illegible_cells": 0,
        "expanded_responses": 4385, "new_target_conceptual_cells": 2310,
        "new_target_attested_cells": 2271, "new_target_blank_cells": 39,
        "installed_forms": 2714, "republished_dhule_cells_excluded": 630,
        "republished_dhule_responses_excluded": 834,
        "control_cells_excluded": 630, "control_responses_excluded": 837,
    }:
        raise ValueError(f"Unexpected complete-package census: {counts}")

    frozen_audit = read_dicts(HERE / "exhaustive_audit.tsv", "\t")
    if audit != frozen_audit:
        raise ValueError("Frozen exhaustive audit no longer equals importer output")
    with (HERE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        frozen_forms = list(csv.reader(stream))
    if forms != frozen_forms:
        raise ValueError("Frozen staged forms no longer equal importer output")
    frozen_reconciliation = read_dicts(HERE / "dhule_republication_reconciliation.tsv", "\t")
    if reconciliation != frozen_reconciliation:
        raise ValueError("Frozen Dhule crosswalk no longer equals importer output")

    if len(frozen_forms) != 2714 or len({row[10] for row in frozen_forms}) != 2714:
        raise ValueError("Staged target Entry_Key census changed")
    audit_by_coordinate = {(row["Item"], row["Site_Code"]): row for row in audit}
    key_re = re.compile(r"noira2015:p(?P<page>\d{3}):i(?P<item>\d{3}):(?P<site>[A-Z]+):a(?P<variant>\d+)$")
    variants = Counter()
    for form in frozen_forms:
        if len(form) != len(noira.FORM_FIELDS):
            raise ValueError("Staged row width changed")
        if form[0] not in TARGET_LANGUAGES:
            raise ValueError(f"Unexpected staged parent language: {form[0]}")
        match = key_re.fullmatch(form[10])
        if not match or match["site"] not in TARGET_CODES:
            raise ValueError(f"Invalid immutable staged Entry_Key: {form[10]}")
        source = audit_by_coordinate[(str(int(match["item"])), match["site"])]
        variants[(source["Item"], source["Site_Code"])] += 1
        expected_key = (
            f"noira2015:p{int(source['PDF_Page']):03d}:i{int(source['Item']):03d}:"
            f"{source['Site_Code']}:a{variants[(source['Item'], source['Site_Code'])]}"
        )
        expected_citation = (
            f"{noira.SOURCE_KEY}[Appendix A3, printed p. {source['Printed_Page']}, "
            f"item {source['Item']}, list {source['Site_Code']}]"
        )
        if form[10] != expected_key or form[7] != expected_citation:
            raise ValueError(f"Immutable key/locator drift: {form[10]}")
        if source["Disposition"] != "staged" or form[10] not in source["Entry_Keys"].split(" | "):
            raise ValueError(f"Staged row lacks exact conceptual-cell crosswalk: {form[10]}")

    if len(audit) != 3570 or Counter(row["Scope"] for row in audit) != Counter(
        new_target=2310, republished_dhule=630, comparison_control=630
    ):
        raise ValueError("Target/control/republication classification changed")
    if sum(int(row["Installed_Count"]) for row in audit) != 2714:
        raise ValueError("Conceptual-to-expanded staging crosswalk changed")
    if any(row["Installed_Count"] != "0" or row["Entry_Keys"] for row in audit if row["Scope"] != "new_target"):
        raise ValueError("Control or republished cell leaked into staging")
    if len(reconciliation) != 630 or Counter(row["Noira_Site"] for row in reconciliation) != Counter(
        NAS=210, BMU=210, NTO=210
    ):
        raise ValueError("Dhule republication crosswalk is not exhaustive")
    if Counter(row["Comparison"] for row in reconciliation) != Counter(
        {"literal-ledger-exact": 3, "same-source-representation-differs": 627}
    ):
        raise ValueError("Dhule representation comparison census changed")
    if any(row["Disposition"] != "exclude Noira republication; retain primary ESR 2013-004 route" for row in reconciliation):
        raise ValueError("Dhule republication exclusion policy changed")
    return rows, audit, frozen_forms


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def build_render_rows(render_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for physical in range(33, 79):
        path = render_dir / f"page-{physical}.png"
        if not path.exists():
            raise ValueError(f"Missing audit render: {path}")
        width, height = png_dimensions(path)
        if (width, height) != (1224, 1584):
            raise ValueError(f"Unexpected render geometry: {path}: {width}x{height}")
        rows.append({
            "Physical_PDF_Page": physical,
            "Printed_Page": physical - 6,
            "Relative_Path": path.name,
            "Bytes": path.stat().st_size,
            "Width": width,
            "Height": height,
            "DPI": 144,
            "SHA256": sha256(path),
            "Evidence_Class": "fresh-topology-audit-only; lexical readings remain frozen manual evidence",
        })
    return rows


def load_render_rows() -> list[dict[str, str]]:
    rows = read_dicts(RENDER_HASHES, "\t")
    if len(rows) != 46 or [int(row["Physical_PDF_Page"]) for row in rows] != list(range(33, 79)):
        raise ValueError("Render manifest does not cover physical PDF pages 33-78 exactly")
    if [int(row["Printed_Page"]) for row in rows] != list(range(27, 73)):
        raise ValueError("Render manifest printed-page mapping changed")
    if any((row["Width"], row["Height"], row["DPI"]) != ("1224", "1584", "144") for row in rows):
        raise ValueError("Render manifest geometry/DPI changed")
    return rows


def build_profile_inventory(forms: list[list[str]]) -> list[dict[str, object]]:
    profile_rows = read_dicts(HERE / "conversion_profile.tsv", "\t")
    ordered = sorted((row["Grapheme"] for row in profile_rows), key=len, reverse=True)
    counts: Counter[str] = Counter()
    for row in forms:
        pending = row[2]
        while pending:
            grapheme = next((candidate for candidate in ordered if pending.startswith(candidate)), None)
            if grapheme is None:
                raise ValueError(f"Source-local profile lacks staged input sequence: {pending!r}")
            counts[grapheme] += 1
            pending = pending[len(grapheme):]
    return [{
        "Grapheme": row["Grapheme"],
        "IPA": row["IPA"],
        "Staged_Input_Occurrences": counts[row["Grapheme"]],
        "Present_In_Staged_Targets": "yes" if counts[row["Grapheme"]] else "no",
    } for row in profile_rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write source-local audit artifacts")
    parser.add_argument(
        "--render-dir", type=Path,
        help="fresh 144-dpi page-33.png ... page-78.png topology audit directory",
    )
    args = parser.parse_args()

    pdf = pdf_metadata()
    rows, audit, forms = verify_frozen_package()
    profile_rows = build_profile_inventory(forms)
    render_rows = build_render_rows(args.render_dir) if args.render_dir else load_render_rows()
    render_text = tsv_text(render_rows, RENDER_FIELDS)
    profile_text = tsv_text(profile_rows, PROFILE_FIELDS)
    if args.write:
        RENDER_HASHES.write_text(render_text, encoding="utf-8")
        PROFILE_INVENTORY.write_text(profile_text, encoding="utf-8")
    else:
        for path, expected in [(RENDER_HASHES, render_text), (PROFILE_INVENTORY, profile_text)]:
            if path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"Generated pre-integration artifact is stale: {path.name}")

    statuses = Counter(row["Review_Status"] for row in rows)
    dispositions = Counter(row["Disposition"] for row in audit)
    manual_paths = [HERE / relative for relative in FROZEN_HASHES if relative.startswith("manual_chunks/")]
    render_tree_body = "".join(
        "\t".join(str(row[field]) for field in RENDER_FIELDS) + "\n" for row in render_rows
    )
    registry = noira.load_registry()
    manifest = {
        "state": "source-local-preintegration-audit-complete",
        "source_key": noira.SOURCE_KEY,
        "authority": (
            "independently frozen manual visual transcription from rendered primary-source pages; "
            "OCR/PDF text, legacy data, and installed forms supplied or verified no lexical reading"
        ),
        "pdf": pdf,
        "renders": {
            "scope": "fresh 144-dpi topology/renderability audit only; not lexical transcription evidence",
            "physical_pages": "33-78",
            "printed_pages": "27-72",
            "artifacts": len(render_rows),
            "bytes": sum(int(row["Bytes"]) for row in render_rows),
            "tree_sha256": hashlib.sha256(render_tree_body.encode("utf-8")).hexdigest(),
            "manifest": RENDER_HASHES.name,
            "manifest_sha256": hashlib.sha256(render_text.encode("utf-8")).hexdigest(),
            "sample_visual_checks": [
                "physical:33/printed:27/appendix-start/items:1+",
                "physical:55/printed:49/midpoint/items:103-107",
                "physical:78/printed:72/appendix-end/items:207-210",
            ],
        },
        "frozen_artifacts": {
            **FROZEN_HASHES,
            "manual_ledger_and_generator_bundle_sha256": bundle_hash(manual_paths),
        },
        "topology": {
            "prompts": 210, "lists": 17, "conceptual_cells": 3570,
            "new_target_lists": 11, "new_target_cells": 2310,
            "republished_dhule_lists": 3, "republished_dhule_cells": 630,
            "comparison_control_lists": 3, "comparison_control_cells": 630,
        },
        "statuses": {
            "attested": statuses["attested"], "source_blank": statuses["source_blank"],
            "ambiguous": statuses["ambiguous"], "illegible": statuses["illegible"],
            "unresolved_coordinates": [], "expanded_responses": 4385,
        },
        "staged_target_forms": {
            "rows": len(forms), "unique_entry_keys": len({row[10] for row in forms}),
            "new_target_attested_cells": dispositions["staged"],
            "new_target_blank_cells": 39,
            "sha256": FROZEN_HASHES["staged_forms.csv"],
            "target_site_codes": sorted(TARGET_CODES),
            "parent_language_ids": sorted(TARGET_LANGUAGES),
        },
        "exclusions": {
            "republished_dhule": {
                "site_codes": sorted(REPUBLISHED_CODES), "conceptual_cells": 630,
                "expanded_responses": 834,
            },
            "comparison_controls": {
                "site_codes": sorted(CONTROL_CODES), "conceptual_cells": 630,
                "expanded_responses": 837,
            },
            "all_source_blanks": 44,
            "new_target_blanks": 39,
        },
        "republication_reconciliation": {
            "rows": 630,
            "sha256": FROZEN_HASHES["dhule_republication_reconciliation.tsv"],
            "literal_ledger_exact": 3,
            "same_source_representation_differs": 627,
            "contract": (
                "All 630 cells are excluded by the primary report's source-team/list identity. "
                "The 627 representation-different labels arise because the Dhule audit field embeds "
                "printed similarity labels while Noira stores labels separately; they are not lexical "
                "disagreements and were not used to verify a Noira reading."
            ),
        },
        "profile": {
            "path": "conversion_profile.tsv",
            "sha256": FROZEN_HASHES["conversion_profile.tsv"],
            "rows": len(profile_rows),
            "missing_staged_input_sequences": [],
            "inventory": PROFILE_INVENTORY.name,
            "inventory_sha256": hashlib.sha256(profile_text.encode("utf-8")).hexdigest(),
        },
        "identity_contract": {
            "new_target_list_registry": [
                {
                    "site_code": code,
                    "language_id": registry[code]["Language_ID"],
                    "dialect_id": registry[code]["Dialect_ID"],
                    "display_name": registry[code]["Display_Name"],
                }
                for code in sorted(TARGET_CODES)
            ],
            "existing_parent_languages": ["Goj", "Ni", "ko"],
            "shared_parent_rows_also_used_by_esr_2013_004": ["DungraBhili", "Noiri"],
            "new_parent_languages": [],
            "kotli_mapping": (
                "The report treats Kotli as a named survey variety, publishes two Kotli wordlists, "
                "reports it was said to be a Noiri dialect, and concludes that it has a distinctive "
                "identity requiring further research. Provisionally route Narayanpur and Taradi as "
                "distinct survey-site dialects under canonical Noiri, preserve the source label Kotli "
                "and Taradi respondent alias Adivasi Bhil-Taradi, leave dialect Glottocode/coordinates "
                "blank, and do not equate either with historical Kotali/Khandesi. This is a source-"
                "supported routing decision, not a genealogical determination."
            ),
        },
        "shared_integration_contract": {
            "install_source_local_target_rows_byte_for_byte": 2714,
            "dated_raw_form_path": "data/other/forms/20260828-sil-noira.csv",
            "reference_key": noira.SOURCE_KEY,
            "profile_route": "sil-noira",
            "new_dialect_rows": 11,
            "republished_dhule_cells_audit_only": 630,
            "comparison_control_cells_audit_only": 630,
            "unresolved_lexical_coordinates": 0,
            "scholarly_identity_blockers": [],
            "deferred": [
                "shared bibliography/language/dialect/profile routing edits",
                "dated installed CSV",
                "consolidated CLDF/full build and full tests",
                "graph validation",
                "browser refresh and representative-entry QA",
                "commit/shipping",
            ],
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write:
        PREINTEGRATION_MANIFEST.write_text(manifest_text, encoding="utf-8")
    elif PREINTEGRATION_MANIFEST.read_text(encoding="utf-8") != manifest_text:
        raise ValueError("Generated pre-integration artifact is stale: preintegration_manifest.json")

    print(
        f"cells={len(rows)} attested={statuses['attested']} blanks={statuses['source_blank']} "
        f"staged_forms={len(forms)} republished=630 controls=630 unresolved=0 "
        f"manual_bundle_sha256={manifest['frozen_artifacts']['manual_ledger_and_generator_bundle_sha256']} "
        f"render_tree_sha256={manifest['renders']['tree_sha256']}"
    )


if __name__ == "__main__":
    main()
