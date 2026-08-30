#!/usr/bin/env python3
"""Verify the frozen Bhumij 2015 source-local integration contract.

Lexical readings come only from the already frozen manual rendered-page
ledgers.  PDF text, OCR, the Ho 2024 republication, and installed data are not
reading authorities.  The Ho material is loaded only after the Bhumij freeze
has been verified, and only to audit a report-supported republication.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import build_overlap_reconciliation as overlap
import import_bhumij_2015 as bhumij


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
PDF = WORKSPACE_ROOT / "tmp/pdfs/bhumij_2015/silesr2015_026.pdf"
PROFILE = HERE.parents[4] / "conversion/sil-bhumij.txt"
RENDER_HASHES = HERE / "render_hashes.tsv"
MANIFEST = HERE / "preintegration_manifest.json"

PDF_HASH = "1dadbe266842c5e07e4efc4d937f2d3f09daacbe10052db6a715635db018e395"
PDF_BYTES = 2_725_577
PDF_PAGES = 130
PDF_TITLE = "A Sociolinguistic Survey of the Bhumij People of India"
PDF_AUTHOR = "Troy Bailey and Loren Maggard"
PROFILE_HASH = "c029063b4fc2ba541c72ba71379b185a94582c1ccb62c24e9c8e76ce5f303525"
HO_INPUT_HASH = "f9c888e6f833a5c7cb2182c6a5ed574f404f48be0a2c9f6423599010b5dd8cd5"

FROZEN_HASHES = {
    "staged_forms.tsv": "1196571ed479289507cbebe89467097a3d44f4e05ced96069e9cc4baff0ae4e9",
    "staged_audit.tsv": "39e81dc327aadf933a342034c4780c2c99e0e280a5dd992dde885dfa8bfc8933",
    "profile_inventory.tsv": "06dd0d0ee4f4eb242d84d91cf32fe2a88ba13f637332d06c0a91905fbb234abf",
    "ho_2024_overlap_reconciliation.tsv": "cf84a94529541f437732ab154bf9fb55e22c984eee18d3790b195e293911f076",
    "list_registry.tsv": "ef5e4e22dc72974436dbb86c91dd69240ac074bb3e4ec0ee532173f857565ea4",
    "overlap_registry.tsv": "7825199acefd59dfbcf2a5215aaa06de5c4adab513c9c26ddcffbec31a4545fd",
    "reference.bib": "3a75de313974fc68e51aa95cd45dad77f1070b8a20d35497b046aa98dc6af2d2",
    "source_manifest.json": "0724ec45fb2cc6cdd741b756c2a83336aa27a2f8f37c0d08b359c88b6fd5fe6e",
    "import_bhumij_2015.py": "a0c689088e483dfdee726c38f43c8ddb7fb9a5bb8174c0bb008cc6e08138c7d7",
    "build_overlap_reconciliation.py": "27eb34d3c2659fc9fed095b7d41334012f5b3a28e2a1906f128a1b81bd6a6900",
}

MANUAL_BUNDLES = {
    "ledgers": ("items_*_hand_keyed.tsv", 61, 1_247_520,
                "093a32a39e410b6529e978c36b1ec6ff4a16734927cf03c1df431c8c04df00a5"),
    "generators": ("hand_keyed_item*.py", 61, 338_768,
                   "ae56b5dc0dc920e3b13070b481211b519283660523e1270e152a9c63effc9729"),
    "audits": ("AUDIT_ITEM*.md", 61, 52_798,
               "8a3232d2c7979c19dccfd8eacf5562fb1d9ef88d9451a239c12e00c7d1175931"),
}

TARGETS = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}
CONTROLS = {"MCH", "MDI", "MDH", "MJH", "HDI", "SDI", "SNA", "ORI"}
OVERLAP_TARGETS = {"BAI", "CHA", "DUM", "LAD", "POD"}
EXPECTED_DIALECTS = {
    "BAI": "bhumij1989-baigodia",
    "CHA": "bhumij1989-champi",
    "DIG": "bhumij1996-dighinuasahi",
    "DUM": "bhumij1989-dumadie",
    "LAD": "bhumij1989-ladhiramsai",
    "MAD": "bhumij1989-madhupur",
    "MOH": "bhumij1996-mohuldiha",
    "MUN": "bhumij1996-munduy",
    "POD": "bhumij1989-podadiha",
    "UDA": "bhumij-mundari1989-udala",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dicts(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def tree_hash(paths: list[Path]) -> str:
    body = "".join(
        f"{path.relative_to(HERE).as_posix()}\t{sha256(path)}\n"
        for path in sorted(paths)
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def pdf_metadata() -> dict[str, object]:
    if PDF.stat().st_size != PDF_BYTES or sha256(PDF) != PDF_HASH:
        raise ValueError("Canonical ESR 2015-026 PDF hash/size changed")
    output = subprocess.run(
        ["pdfinfo", str(PDF)], check=True, capture_output=True, text=True
    ).stdout
    info = {
        key.strip(): value.strip()
        for line in output.splitlines() if ":" in line
        for key, value in [line.split(":", 1)]
    }
    if info.get("Title") != PDF_TITLE or info.get("Author") != PDF_AUTHOR:
        raise ValueError("Canonical PDF title/author metadata changed")
    if int(info.get("Pages", "0")) != PDF_PAGES:
        raise ValueError("Canonical PDF page count changed")
    if not info.get("Page size", "").startswith("612 x 792 pts"):
        raise ValueError("Canonical PDF page geometry changed")
    if info.get("Page rot") != "0":
        raise ValueError("Canonical PDF page rotation changed")
    return {
        "path": PDF.relative_to(WORKSPACE_ROOT).as_posix(),
        "bytes": PDF_BYTES,
        "pages": PDF_PAGES,
        "sha256": PDF_HASH,
        "title": PDF_TITLE,
        "author": PDF_AUTHOR,
        "page_size_points": "612 x 792",
        "page_rotation": 0,
    }


def verify_hashes() -> dict[str, object]:
    for relative, expected in FROZEN_HASHES.items():
        actual = sha256(HERE / relative)
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {relative}: {actual}")
    bundles: dict[str, object] = {}
    for name, (pattern, expected_files, expected_bytes, expected_hash) in MANUAL_BUNDLES.items():
        paths = sorted((HERE / "manual_chunks").glob(pattern))
        actual = tree_hash(paths)
        if (len(paths), sum(path.stat().st_size for path in paths), actual) != (
            expected_files, expected_bytes, expected_hash
        ):
            raise ValueError(f"Frozen manual {name} bundle changed")
        bundles[name] = {
            "files": expected_files, "bytes": expected_bytes, "tree_sha256": expected_hash,
        }
    ho_input = HERE.parent / "sil_ho_2024/staged_audit.tsv"
    if sha256(ho_input) != HO_INPUT_HASH:
        raise ValueError("Post-freeze Ho 2024 reconciliation input changed")
    if sha256(PROFILE) != PROFILE_HASH:
        raise ValueError("Read-only shared Bhumij profile changed")
    return bundles


def verify_render_manifest() -> dict[str, object]:
    rows = read_dicts(RENDER_HASHES)
    if len(rows) != 43:
        raise ValueError("Render manifest must contain all 43 lexical pages")
    if [int(row["Physical_PDF_Page"]) for row in rows] != list(range(34, 77)):
        raise ValueError("Render physical-page topology changed")
    if [int(row["Printed_Page"]) for row in rows] != list(range(29, 72)):
        raise ValueError("Render printed-page topology changed")
    if any((row["Width"], row["Height"], row["DPI"]) != ("1224", "1584", "144") for row in rows):
        raise ValueError("Render geometry/DPI changed")
    if any("topology-audit-only" not in row["Evidence_Class"] for row in rows):
        raise ValueError("Render evidence boundary changed")
    body = "".join(
        "\t".join(row[field] for field in [
            "Physical_PDF_Page", "Printed_Page", "Relative_Path", "Bytes",
            "Width", "Height", "DPI", "SHA256",
        ]) + "\n" for row in rows
    )
    tree = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if tree != "8fd671a5638be3afbebd73fbeef1955a771e0659a327da3eb69f295e696ff37d":
        raise ValueError("Fresh lexical-page render tree changed")
    return {
        "scope": "fresh 144-dpi topology/renderability audit only; not lexical transcription evidence",
        "physical_pages": "34-76", "printed_pages": "29-71", "artifacts": 43,
        "bytes": sum(int(row["Bytes"]) for row in rows), "tree_sha256": tree,
        "manifest": RENDER_HASHES.name, "manifest_sha256": sha256(RENDER_HASHES),
        "sample_visual_checks": [
            "physical:34/printed:29/appendix-start/items:1-5",
            "physical:55/printed:50/midpoint/items:107-111",
            "physical:76/printed:71/appendix-end/items:209-210",
        ],
    }


def verify_frozen_package() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    ledgers = sorted((HERE / "manual_chunks").glob("items_*_hand_keyed.tsv"))
    rows = bhumij.load_manual_ledgers(ledgers)
    bhumij.require_full_review(rows)
    registry = bhumij.load_registry(HERE)
    forms = bhumij.stage_target_forms(rows, registry)
    audit = bhumij.build_audit(rows, registry)
    frozen_forms = read_dicts(HERE / "staged_forms.tsv")
    frozen_audit = read_dicts(HERE / "staged_audit.tsv")
    if forms != frozen_forms or audit != frozen_audit:
        raise ValueError("Frozen staged output no longer equals guarded importer output")

    statuses = Counter(row["Review_Status"] for row in rows)
    if statuses != Counter(attested=3690, source_blank=90):
        raise ValueError(f"Unexpected source status census: {statuses}")
    if sum(len(row["Manual_Transcription"].split(" | ")) for row in rows if row["Review_Status"] == "attested") != 3876:
        raise ValueError("Expanded-response census changed")
    if len(forms) != 2100 or len({row["Entry_Key"] for row in forms}) != 2100:
        raise ValueError("Target form/immutable Entry_Key census changed")
    if Counter(row["Disposition"] for row in audit) != Counter({
        "target-staged": 2054,
        "target-source-blank-excluded": 46,
        "comparison-control-excluded": 1636,
        "comparison-control-blank-excluded": 44,
    }):
        raise ValueError("Target/control disposition census changed")
    if any(row["Review_Status"] in {"ambiguous", "illegible"} for row in rows):
        raise ValueError("An unresolved lexical coordinate remains")
    if any("\ufffd" in value for row in rows for value in row.values()):
        raise ValueError("Replacement character in frozen ledger")
    if any(not unicodedata.is_normalized("NFC", value) for row in rows for value in row.values()):
        raise ValueError("Non-NFC frozen ledger value")

    by_coordinate = {(row["Item"], row["Site_Code"]): row for row in rows}
    qualifier = by_coordinate[("195", "LAD")]
    if not (
        qualifier["Manual_Transcription"] == "sɛnodʒɑnʌ, dolɑ"
        and qualifier["Confidence"] == "medium"
        and qualifier["Uncertainty"] == "source appends '(?)' after dolɑ; printed form itself is legible"
    ):
        raise ValueError("Source-marked item 195/LAD (?) qualifier contract changed")

    key_re = re.compile(r"(?P<dialect>.+)-i(?P<item>\d{3})-a(?P<variant>\d{2})$")
    for form in forms:
        match = key_re.fullmatch(form["Entry_Key"])
        if not match or match["dialect"] not in EXPECTED_DIALECTS.values():
            raise ValueError(f"Invalid immutable target Entry_Key: {form['Entry_Key']}")
        site = next(code for code, dialect in EXPECTED_DIALECTS.items() if dialect == match["dialect"])
        source = by_coordinate[(str(int(match["item"])), site)]
        expected_source = (
            f"{bhumij.SOURCE_KEY}[Appendix B.3, printed p. {source['Printed_Page']}, "
            f"item {source['Item']}, list {site}]"
        )
        locator = (
            f"physical p.{source['PDF_Page']} / printed p.{source['Printed_Page']} / "
            f"item {source['Item']} / list {site} / {source['Column']} column"
        )
        if form["Source"] != expected_source or locator not in form["Notes"]:
            raise ValueError(f"Source locator drift: {form['Entry_Key']}")
        variant = int(match["variant"])
        base = form["Entry_Key"][:-2]
        if form["Variant_Of_Key"] != ("" if variant == 1 else f"{base}01"):
            raise ValueError(f"Variant link drift: {form['Entry_Key']}")
        if form["Language_ID"] != "unr" or not form["Tags"].startswith(f"dialect:unr:{match['dialect']}:"):
            raise ValueError(f"Target identity routing drift: {form['Entry_Key']}")

    inventory, unmatched = bhumij.profile_inventory(forms, HERE)
    if unmatched or len(inventory) != 53 or inventory != read_dicts(HERE / "profile_inventory.tsv"):
        raise ValueError("Preservation profile inventory/coverage changed")
    return rows, audit, forms


def verify_identity_policy() -> list[dict[str, str]]:
    registry = read_dicts(HERE / "list_registry.tsv")
    by_site = {row["Site_Code"]: row for row in registry}
    if set(by_site) != TARGETS | CONTROLS or len(registry) != 18:
        raise ValueError("List registry topology changed")
    if {row["Site_Code"] for row in registry if row["Scope"] == "target" and row["Install"] == "yes"} != TARGETS:
        raise ValueError("Target policy changed")
    if {row["Site_Code"] for row in registry if row["Scope"] == "comparison_control" and row["Install"] == "no"} != CONTROLS:
        raise ValueError("Control exclusion policy changed")
    for site, dialect in EXPECTED_DIALECTS.items():
        if by_site[site]["Language_ID"] != "unr" or by_site[site]["Dialect_ID"] != dialect:
            raise ValueError(f"Target language/dialect mapping changed: {site}")
    if by_site["UDA"]["Source_Language_Label"] != "Mundari? Bhumij?":
        raise ValueError("Udala mixed source label was resolved or lost")
    if "retain mixed label" not in by_site["UDA"]["Notes"]:
        raise ValueError("Udala uncertainty note changed")
    if any(row["Dialect_ID"] for row in registry if row["Site_Code"] in CONTROLS):
        raise ValueError("Comparison control gained an installable dialect")
    return registry


def verify_overlap() -> list[dict[str, str]]:
    rows = overlap.build_rows()
    if rows != read_dicts(HERE / "ho_2024_overlap_reconciliation.tsv"):
        raise ValueError("Frozen Ho 2024 overlap audit no longer equals reconstruction")
    if len(rows) != 1050 or len({(row["Item"], row["Bhumij_Site_Code"]) for row in rows}) != 1050:
        raise ValueError("Ho 2024 republication crosswalk is not exhaustive")
    if Counter(row["Bhumij_Site_Code"] for row in rows) != Counter({site: 210 for site in OVERLAP_TARGETS}):
        raise ValueError("Ho 2024 five-list coverage changed")
    if Counter(row["Representation_Comparison"] for row in rows) != Counter({
        "blank-parity": 11,
        "unicode-exact-after-label-removal": 221,
        "publication-transcription-differs": 818,
    }):
        raise ValueError("Ho 2024 representation-comparison census changed")
    if any(row["Status_Parity"] != "yes" for row in rows):
        raise ValueError("Ho 2024 status parity changed")
    if {row["Canonical_Publication"] for row in rows} != {bhumij.SOURCE_KEY}:
        raise ValueError("Primary publication route changed")
    return rows


def expected_manifest() -> dict[str, object]:
    pdf = pdf_metadata()
    bundles = verify_hashes()
    renders = verify_render_manifest()
    rows, audit, forms = verify_frozen_package()
    registry = verify_identity_policy()
    overlap_rows = verify_overlap()
    target_responses = len(forms)
    control_responses = sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Site_Code"] in CONTROLS and row["Review_Status"] == "attested"
    )
    return {
        "state": "source-local-preintegration-audit-complete",
        "source_key": bhumij.SOURCE_KEY,
        "authority": (
            "frozen cell-by-cell manual visual transcription from rendered primary-source pages; "
            "OCR/PDF text, legacy data, installed forms, and the later Ho publication supplied or "
            "verified no lexical reading"
        ),
        "pdf": pdf,
        "renders": renders,
        "frozen_artifacts": {**FROZEN_HASHES, "manual_bundles": bundles},
        "topology": {
            "prompts": 210, "lists": 18, "conceptual_cells": 3780,
            "target_or_mixed_lists": 10, "target_or_mixed_cells": 2100,
            "comparison_control_lists": 8, "comparison_control_cells": 1680,
        },
        "statuses": {
            "attested": 3690, "source_blank": 90, "ambiguous": 0,
            "illegible": 0, "pending": 0, "expanded_responses": 3876,
            "unresolved_coordinates": [],
        },
        "staged_target_forms": {
            "rows": target_responses, "unique_entry_keys": len({row["Entry_Key"] for row in forms}),
            "attested_conceptual_cells": 2054, "source_blank_cells": 46,
            "variant_rows_beyond_first": sum(bool(row["Variant_Of_Key"]) for row in forms),
            "site_codes": sorted(TARGETS), "language_ids": ["unr"],
            "sha256": FROZEN_HASHES["staged_forms.tsv"],
        },
        "comparison_controls": {
            "site_codes": sorted(CONTROLS), "lists": 8, "conceptual_cells": 1680,
            "attested_cells": 1636, "source_blank_cells": 44,
            "expanded_responses_audit_only": control_responses,
        },
        "source_qualifier": {
            "coordinate": "physical:73/printed:68/item:195/site:LAD/left",
            "marker": "(?)", "form": "sɛnodʒɑnʌ, dolɑ", "confidence": "medium",
            "policy": "preserve marker in Uncertainty; do not add it to the lexical form",
        },
        "identity_contract": {
            "parent_language_id": "unr", "new_dialect_rows": 10,
            "target_dialects": [
                {"site_code": row["Site_Code"], "dialect_id": row["Dialect_ID"],
                 "display_name": row["Display_Name"], "source_label": row["Source_Language_Label"]}
                for row in registry if row["Site_Code"] in TARGETS
            ],
            "udala_policy": (
                "retain source label 'Mundari? Bhumij?' as a distinct mixed Bhumij/Mundari survey "
                "dialect under unr; do not resolve the report's uncertainty"
            ),
        },
        "profile": {
            "route": "sil-bhumij", "read_only_path": "data/conversion/sil-bhumij.txt",
            "sha256": PROFILE_HASH, "mapping_rows": len(read_dicts(PROFILE)),
            "staged_character_inventory": "profile_inventory.tsv",
            "staged_characters": 53, "missing_staged_input_sequences": [],
            "inventory_sha256": FROZEN_HASHES["profile_inventory.tsv"],
        },
        "ho_2024_republication": {
            "lists": 5, "conceptual_cells": len(overlap_rows),
            "blank_parity": 11, "unicode_exact_after_label_removal": 221,
            "publication_transcription_differs": 818,
            "canonical_publication": bhumij.SOURCE_KEY,
            "sha256": FROZEN_HASHES["ho_2024_overlap_reconciliation.tsv"],
            "comparison_input_sha256": HO_INPUT_HASH,
            "contract": (
                "Same elicitation identity is established from locality, date, speaker, and recorder; "
                "the Ho 2024 forms are post-freeze comparison evidence only and did not verify any "
                "Bhumij reading. Install Bailey & Maggard 2015; keep the later republication audit-only."
            ),
        },
        "shared_integration_contract": {
            "install_source_local_target_rows_byte_for_byte": 2100,
            "comparison_control_cells_audit_only": 1680,
            "ho_2024_same_elicitation_cells_audit_only": 1050,
            "new_dialect_rows": 10, "profile_route": "sil-bhumij",
            "reference_key": bhumij.SOURCE_KEY, "unresolved_lexical_coordinates": 0,
            "scholarly_identity_blockers": [],
            "deferred": [
                "shared bibliography/language/dialect/profile routing edits",
                "dated installed target CSV",
                "consolidated CLDF/full build and full tests",
                "graph validation",
                "browser refresh and representative-entry QA",
                "commit/shipping",
            ],
        },
    }


def main() -> None:
    expected = expected_manifest()
    actual = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("preintegration_manifest.json is stale")
    print(
        "Bhumij preintegration OK: cells=3780 attested=3690 blanks=90 "
        "responses=3876 staged_forms=2100 controls=1680 ho_republication=1050 unresolved=0"
    )


if __name__ == "__main__":
    main()
