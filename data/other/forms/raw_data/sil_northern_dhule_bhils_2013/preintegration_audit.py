#!/usr/bin/env python3
"""Freeze and verify the Northern Dhule Bhils source-local integration contract.

The Dhule manual review is validated and hash-checked before this script opens
either later-source reconciliation artifact.  No Noira/Bareli form is accepted
as evidence for a Dhule reading; those sources are used only to identify later
republications after the independent 2013 transcription has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import import_northern_dhule_bhils as dhule


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
DATA_ROOT = WORKSPACE_ROOT / "data"
RENDER_ROOT = WORKSPACE_ROOT / "tmp/pdfs/northern_dhule_bhils_2013"
NOIRA_RECONCILIATION = (
    DATA_ROOT
    / "data/other/forms/raw_data/sil_noira_2015/dhule_republication_reconciliation.tsv"
)
BARELI_AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-bareli-pauri-audit.csv"
BARELI_PDF = WORKSPACE_ROOT / "tmp/pdfs/bareli/silesr2018_011.pdf"
BARELI_TEXT = WORKSPACE_ROOT / "tmp/pdfs/bareli/silesr2018_011.txt"

RENDER_HASHES = HERE / "render_hashes.tsv"
PROFILE_INVENTORY = HERE / "profile_inventory.tsv"
CROSS_SOURCE_RECONCILIATION = HERE / "cross_source_reconciliation.tsv"
PREINTEGRATION_MANIFEST = HERE / "preintegration_manifest.json"

FROZEN_HASHES = {
    "manual_review.tsv": "8855fe166c387cdd7bcd87cfea5a7af37890e31bbfcdbf65ff08b5fe94ec3bd5",
    "manual_chunks/items_001_035_hand_keyed.tsv": "d2bca4c849e2833fbacfcd16679e68196ad7e6a24e2af08c1a5d645226744aaf",
    "manual_chunks/items_036_070_hand_keyed.tsv": "2c6b9698957a29f16e8f68a1496f91647b0b8f4a5422d24a076f048aa713ee8a",
    "manual_chunks/items_071_105_hand_keyed.tsv": "9d7f14859f46d038f8d9bf98220b77da707c71c7a08527e4f6ff6d3694c2891e",
    "manual_chunks/items_106_140_hand_keyed.tsv": "71e92e01ede2a349d7b8d0a699adc5cfdfa39e6aeb46ed180f09b030758adecf",
    "manual_chunks/items_141_175_hand_keyed.tsv": "6096457045cb317ba3fb8995b7cb1e7a071e453b0b22117cb779afeb0debf7bf",
    "manual_chunks/items_176_210_hand_keyed.tsv": "a12bb9e5258dc56d4dc71602824dc28ae1c981853c59e58b622fa8e1054c356a",
    "staged_forms.csv": "5641b9d7ecfb44e6e644efba35e65223260291b7a8724b1fd25fac2fc94d3ed4",
    "staged_audit.tsv": "4bc5aa3bf41e79622494fea7426b6c77532ab4600c263a32179aa6c248b9c302",
    "unresolved_readings.tsv": "c3ab989d0d9d0403d4f60c2edab6ecd5276b8ab583da2536d042752467a18f6c",
    "list_registry.tsv": "635b57460fb9b5a6fa682941ae39a857ef26542271c6e73bf5cbeb7c3631dc62",
    "conversion_profile.tsv": "b0bca6f983bbcf87dc43769c804ae02a73db45d58fad1e86f975fb8b9f7456ce",
}
PDF_HASH = "edeeeda98cb76624df1a0d70c765cc816ea463d75bc79ec20883c62e6fc1c482"
NOIRA_RECONCILIATION_HASH = "fe0a636c70a7979921b7f5f107a84a7279a1654f24a9aeb964a1be26f0960ee1"
BARELI_AUDIT_HASH = "d6336d93714998fc8c41e67da06090fa00d79dc302e20873c8d7a8b4f5ab0bd4"
BARELI_PDF_HASH = "02128358a61e175ba2a07b2862f6072167a3609cf71264e235ae21284fe2ceea"
BARELI_TEXT_HASH = "d3ade5778e88350b89b10d629516bde742a56aa962394cc51430f380f9a096d7"

BARELI_DIALECTS = {
    "MAN": "sil-bareli-2018-bareli-pauri-mandvi",
    "AML": "sil-bareli-2018-rathwi-pauri-amalwadi",
    "SEG": "sil-bareli-2018-rathwi-pauri-segwi",
    "SHA": "sil-bareli-2018-bareli-pauri-shahana",
}

RECONCILIATION_FIELDS = [
    "Related_Source", "Relation", "Dhule_Site", "Related_Site", "Item", "Gloss",
    "Dhule_PDF_Page", "Related_PDF_Pages", "Dhule_Status", "Dhule_Form",
    "Related_Status", "Related_Forms", "Related_Entry_Keys", "Comparison",
    "Dhule_Disposition", "Related_Disposition", "Evidence",
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


def bundle_hash(paths: list[Path], root: Path) -> str:
    lines = [f"{path.relative_to(root).as_posix()}\t{sha256(path)}\n" for path in sorted(paths)]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def verify_frozen_manual_package() -> tuple[list[dict[str, str]], list[dict[str, str]], list[list[str]]]:
    if dhule.PDF.stat().st_size != 9_214_722 or sha256(dhule.PDF) != PDF_HASH:
        raise ValueError("Canonical ESR 2013-004 PDF hash/size changed")
    for relative, expected in FROZEN_HASHES.items():
        actual = sha256(HERE / relative)
        if actual != expected:
            raise ValueError(f"Frozen source-local artifact changed: {relative}: {actual}")

    base = dhule.validate_base()
    specs = dhule.validate_registry()
    effective = dhule.overlay_manual_chunks(base)
    counts = dhule.require_complete(effective)
    if counts != Counter(attested=2703, blank=24, ambiguous=3):
        raise ValueError(f"Unexpected manual status census: {counts}")

    audit = dhule.build_audit(effective, specs)
    expected_audit = read_dicts(HERE / "staged_audit.tsv", "\t")
    if audit != expected_audit:
        raise ValueError("Frozen staged audit no longer equals importer output")
    target_rows = dhule.staged_rows(effective, specs)
    with (HERE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        frozen_forms = list(csv.reader(stream))
    expected_forms = [[row[field] for field in dhule.RAW_FORM_FIELDS] for row in target_rows]
    if expected_forms != frozen_forms:
        raise ValueError("Frozen staged forms no longer equal importer output")
    if len(frozen_forms) != 2497 or len({row[10] for row in frozen_forms}) != 2497:
        raise ValueError("Target staged Entry_Key census changed")
    for row in target_rows:
        item = int(row["Entry_Key"].split(":item:", 1)[1].split(":", 1)[0])
        site = row["Entry_Key"].rsplit(":site:", 1)[1]
        expected_key = f"watters2013northerndhule:item:{item:03d}:site:{site}"
        expected_locator = f"item {item}, list {site}]"
        if row["Entry_Key"] != expected_key or expected_locator not in row["Source"]:
            raise ValueError(f"Immutable key/locator drift: {row['Entry_Key']}")
    return effective, specs, frozen_forms


def build_render_hash_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for path in sorted(RENDER_ROOT.rglob("*.png")):
        relative = path.relative_to(RENDER_ROOT).as_posix()
        if relative.startswith("rendered_300/"):
            evidence_class = "primary-300dpi"
        elif relative.startswith("rendered_400/"):
            evidence_class = "primary-400dpi"
        elif relative.startswith("rendered_900/") or "900" in path.name:
            evidence_class = "targeted-900dpi-rereview"
        elif relative.startswith("locator_100/"):
            evidence_class = "locator-only-100dpi"
        else:
            evidence_class = "auxiliary-render"
        rows.append({
            "Relative_Path": relative,
            "Bytes": path.stat().st_size,
            "SHA256": sha256(path),
            "Evidence_Class": evidence_class,
        })
    if len(rows) != 234:
        raise ValueError(f"Expected 234 current render artifacts, found {len(rows)}")
    return rows


def build_profile_inventory(frozen_forms: list[list[str]]) -> list[dict[str, str | int]]:
    profile = read_dicts(HERE / "conversion_profile.tsv", "\t")
    counts = Counter(char for row in frozen_forms for char in row[2])
    mapped = {row["Grapheme"] for row in profile}
    missing = set(counts) - mapped
    if missing:
        raise ValueError(f"Source-local profile lacks staged input characters: {sorted(missing)}")
    return [
        {
            "Grapheme": row["Grapheme"],
            "IPA": row["IPA"],
            "Installed_Input_Occurrences": counts[row["Grapheme"]],
            "Present_In_Staged_Targets": "yes" if counts[row["Grapheme"]] else "no",
        }
        for row in profile
    ]


def build_reconciliation(effective: list[dict[str, str]]) -> list[dict[str, str]]:
    # This function is called only after verify_frozen_manual_package has pinned
    # every independent Dhule reading and staged artifact above.
    if sha256(NOIRA_RECONCILIATION) != NOIRA_RECONCILIATION_HASH:
        raise ValueError("Noira-to-Dhule reconciliation input changed")
    if sha256(BARELI_AUDIT) != BARELI_AUDIT_HASH:
        raise ValueError("Bareli source audit changed")
    if sha256(BARELI_PDF) != BARELI_PDF_HASH or sha256(BARELI_TEXT) != BARELI_TEXT_HASH:
        raise ValueError("Bareli republication evidence hash changed")

    by_dhule = {(row["Site_Code"], int(row["Item"])): row for row in effective}
    output: list[dict[str, str]] = []
    for row in read_dicts(NOIRA_RECONCILIATION, "\t"):
        item = int(row["Item"])
        source = by_dhule[(row["Dhule_Site"], item)]
        noira_form = row["Noira_Manual_Transcription"]
        output.append({
            "Related_Source": "varghesekumar2015noira",
            "Relation": "later-republication-of-dhule-list",
            "Dhule_Site": row["Dhule_Site"],
            "Related_Site": row["Noira_Site"],
            "Item": row["Item"],
            "Gloss": row["Gloss"],
            "Dhule_PDF_Page": row["Dhule_PDF_Page"],
            "Related_PDF_Pages": row["Noira_PDF_Page"],
            "Dhule_Status": source["Review_Status"],
            "Dhule_Form": row["Dhule_Manual_Transcription"],
            "Related_Status": "attested" if noira_form else "blank",
            "Related_Forms": noira_form,
            "Related_Entry_Keys": "",
            "Comparison": row["Comparison"],
            "Dhule_Disposition": (
                "primary-target-install-candidate"
                if row["Dhule_Site"] != "TOR" and source["Review_Status"] == "attested"
                else "source-audit-only"
            ),
            "Related_Disposition": row["Disposition"],
            "Evidence": "Noira package's exhaustive 630-cell republication crosswalk; consulted only after Dhule freeze",
        })

    bareli_rows = read_dicts(BARELI_AUDIT)
    by_bareli: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in bareli_rows:
        if row["Dialect_ID"] in BARELI_DIALECTS.values():
            by_bareli[(row["Dialect_ID"], int(row["Concept"]))].append(row)
    if len(by_bareli) != 840:
        raise ValueError(f"Expected 840 Bareli republication cells, found {len(by_bareli)}")
    for site, dialect_id in BARELI_DIALECTS.items():
        for item in range(1, 211):
            source = by_dhule[(site, item)]
            related = by_bareli[(dialect_id, item)]
            installed = [row for row in related if row["Status"] == "installed"]
            forms = [row["Form"] for row in installed]
            dhule_form = (
                dhule.strip_similarity_labels(source["Manual_Transcription"])
                if source["Review_Status"] == "attested"
                else ""
            )
            if source["Review_Status"] == "blank" and not forms:
                comparison = "both-blank"
            elif source["Review_Status"] == "attested" and not forms:
                comparison = "dhule-attested-bareli-excluded"
            elif len(forms) == 1 and forms[0] == dhule_form:
                comparison = "exact-single-form"
            else:
                comparison = "same-source-representation-differs"
            output.append({
                "Related_Source": "varkey-vunnamatla2018bareli",
                "Relation": "later-republication-of-dhule-list",
                "Dhule_Site": site,
                "Related_Site": dialect_id,
                "Item": str(item),
                "Gloss": source["Gloss"],
                "Dhule_PDF_Page": source["PDF_Page"],
                "Related_PDF_Pages": "|".join(dict.fromkeys(row["PDF_Page"] for row in related)),
                "Dhule_Status": source["Review_Status"],
                "Dhule_Form": dhule_form,
                "Related_Status": "attested" if forms else "excluded",
                "Related_Forms": " | ".join(forms),
                "Related_Entry_Keys": "|".join(row["Entry_Key"] for row in installed),
                "Comparison": comparison,
                "Dhule_Disposition": (
                    "primary-target-install-candidate"
                    if source["Review_Status"] == "attested"
                    else "source-audit-only"
                ),
                "Related_Disposition": (
                    "retain later citation/reading; merge only if complete compiled lexical identity agrees"
                    if forms
                    else "retain Bareli exclusion in its source audit"
                ),
                "Evidence": (
                    "ESR 2018-011 Appendix C.2 identifies this list as sourced from Watters's Dhule district report; "
                    "later form text is post-freeze reconciliation evidence only"
                ),
            })
    if len(output) != 1470:
        raise ValueError(f"Expected 1,470 cross-source reconciliation rows, found {len(output)}")
    return output


def tsv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write source-local audit artifacts")
    args = parser.parse_args()

    effective, specs, frozen_forms = verify_frozen_manual_package()
    render_rows = build_render_hash_rows()
    profile_rows = build_profile_inventory(frozen_forms)
    reconciliation = build_reconciliation(effective)

    status_counts = Counter(row["Review_Status"] for row in effective)
    target_sites = {row["Site_Code"] for row in specs if row["Scope"] == "target"}
    target_counts = Counter(
        row["Review_Status"] for row in effective if row["Site_Code"] in target_sites
    )
    control_counts = Counter(
        row["Review_Status"] for row in effective if row["Site_Code"] == "TOR"
    )
    reconciliation_counts = Counter(
        (row["Related_Source"], row["Comparison"]) for row in reconciliation
    )
    manual_chunks = [HERE / relative for relative in FROZEN_HASHES if relative.startswith("manual_chunks/")]
    render_manifest_body = "".join(
        f"{row['Relative_Path']}\t{row['Bytes']}\t{row['SHA256']}\t{row['Evidence_Class']}\n"
        for row in render_rows
    )
    render_tree_hash = hashlib.sha256(render_manifest_body.encode("utf-8")).hexdigest()

    generated_text = {
        RENDER_HASHES: tsv_text(
            render_rows, ["Relative_Path", "Bytes", "SHA256", "Evidence_Class"]
        ),
        PROFILE_INVENTORY: tsv_text(
            profile_rows,
            ["Grapheme", "IPA", "Installed_Input_Occurrences", "Present_In_Staged_Targets"],
        ),
        CROSS_SOURCE_RECONCILIATION: tsv_text(reconciliation, RECONCILIATION_FIELDS),
    }
    if args.write:
        for path, content in generated_text.items():
            path.write_text(content, encoding="utf-8")
    else:
        for path, content in generated_text.items():
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Generated pre-integration artifact is stale: {path.name}")

    manifest = {
        "state": "source-local-preintegration-audit-complete",
        "source_key": dhule.SOURCE_KEY,
        "authority": "manual visual transcription from rendered source images; OCR/PDF text/later data supplied or verified no reading",
        "pdf": {"bytes": dhule.PDF.stat().st_size, "pages": 133, "sha256": PDF_HASH},
        "renders": {
            "artifacts": len(render_rows),
            "bytes": sum(int(row["Bytes"]) for row in render_rows),
            "tree_sha256": render_tree_hash,
            "manifest": RENDER_HASHES.name,
            "manifest_sha256": hashlib.sha256(generated_text[RENDER_HASHES].encode("utf-8")).hexdigest(),
        },
        "frozen_artifacts": {
            **FROZEN_HASHES,
            "manual_cell_bundle_sha256": bundle_hash(manual_chunks, HERE),
        },
        "topology": {
            "prompts": 210,
            "lists": 13,
            "target_lists": 12,
            "control_lists": 1,
            "conceptual_cells": 2730,
            "target_cells": 2520,
            "control_cells": 210,
        },
        "statuses": {
            "all": {status: status_counts[status] for status in ["attested", "blank", "ambiguous", "illegible", "unreviewed"]},
            "target": {status: target_counts[status] for status in ["attested", "blank", "ambiguous", "illegible", "unreviewed"]},
            "control": {status: control_counts[status] for status in ["attested", "blank", "ambiguous", "illegible", "unreviewed"]},
        },
        "staged_target_forms": {
            "rows": len(frozen_forms),
            "unique_entry_keys": len({row[10] for row in frozen_forms}),
            "sha256": FROZEN_HASHES["staged_forms.csv"],
        },
        "unresolved_coordinates": [
            "item:010/site:KEL/pdf:92/printed:84/column:left",
            "item:031/site:MUN/pdf:97/printed:89/column:left",
            "item:074/site:TOR/pdf:105/printed:97/column:right",
        ],
        "profile": {
            "path": "conversion_profile.tsv",
            "sha256": FROZEN_HASHES["conversion_profile.tsv"],
            "rows": len(profile_rows),
            "missing_staged_input_characters": [],
            "inventory": PROFILE_INVENTORY.name,
            "inventory_sha256": hashlib.sha256(generated_text[PROFILE_INVENTORY].encode("utf-8")).hexdigest(),
        },
        "reconciliation": {
            "rows": len(reconciliation),
            "audit": CROSS_SOURCE_RECONCILIATION.name,
            "audit_sha256": hashlib.sha256(
                generated_text[CROSS_SOURCE_RECONCILIATION].encode("utf-8")
            ).hexdigest(),
            "noira_input_sha256": NOIRA_RECONCILIATION_HASH,
            "bareli_audit_sha256": BARELI_AUDIT_HASH,
            "counts": {
                f"{source}|{comparison}": count
                for (source, comparison), count in sorted(reconciliation_counts.items())
            },
            "contract": (
                "ESR 2013-004 remains the primary reading for its independently frozen lists. "
                "Later-source evidence is preserved; exact complete lexical identities may merge citations, "
                "while differing publication transcriptions remain distinct attestations and never overwrite one another."
            ),
        },
        "shared_integration_contract": {
            "install_source_local_target_rows": 2497,
            "exclude_toranmal_control_cells": 210,
            "exclude_target_blanks": 21,
            "exclude_target_ambiguous": 2,
            "new_target_rows_not_republished_in_bareli": 1665,
            "dhule_rows_from_four_lists_republished_in_bareli": 832,
            "reuse_existing_bareli_dialect_ids": 4,
            "new_dialect_ids": 8,
            "new_base_languages": ["Vasavi", "Noiri"],
            "existing_base_languages": ["PauriBareli", "RathwiBareli"],
            "existing_bibliography_alias_to_retire": "bhildhule",
            "deferred": [
                "shared bibliography/language/dialect/profile routing edits",
                "durable identity and citation reconciliation",
                "dated installed CSV",
                "consolidated CLDF/full build and full tests",
                "browser refresh/QA",
                "commit/shipping",
            ],
        },
    }
    if args.write:
        PREINTEGRATION_MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"cells={sum(status_counts.values())} target_forms={len(frozen_forms)} "
        f"renders={len(render_rows)} reconciliation={len(reconciliation)} "
        f"manual_bundle_sha256={manifest['frozen_artifacts']['manual_cell_bundle_sha256']} "
        f"render_tree_sha256={render_tree_hash}"
    )


if __name__ == "__main__":
    main()
