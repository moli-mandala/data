#!/usr/bin/env python3
"""Build source-local Koch post-freeze reconciliation and staging artifacts.

The frozen manual ledgers are authoritative. Legacy data are comparison-only and
may never supply or verify a manual reading.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
MANUAL_MANIFEST = PACKAGE / "source_manifest.json"
LEGACY_INSTALLED = PACKAGE / "legacy_20260826-sil-kochbd.csv"
LEGACY_AUDIT = PACKAGE / "legacy_20260826-sil-kochbd-audit.csv"
BASE_PROFILE = PACKAGE / "legacy_sound_profile_base.txt"
RECONCILIATION = PACKAGE / "reconciliation.tsv"
STAGING_AUDIT = PACKAGE / "staging_audit.tsv"
STAGED_FORMS = PACKAGE / "staged_forms.csv"
SITE_METADATA = PACKAGE / "site_metadata.tsv"
REFERENCE_METADATA = PACKAGE / "reference_metadata.json"
EXCLUSION_POLICY = PACKAGE / "exclusion_policy.json"
SOUND_INVENTORY = PACKAGE / "sound_inventory.tsv"
SOUND_PROFILE = PACKAGE / "sound_profile.txt"
SOUND_DECISIONS = PACKAGE / "sound_profile_decisions.json"
POST_FREEZE_MANIFEST = PACKAGE / "post_freeze_manifest.json"

SOURCE_KEY = "kim-ahmad-kim-sangma2011kochbd"
SOURCE_PDF_SHA256 = "d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10"
SITE_CODES = "bcqrlm0"
SITE_ORDER = {code: index for index, code in enumerate(SITE_CODES)}
SITES = {
    "b": {
        "site_id": "kochbd2011-b-nokshi",
        "site_name": "Nokshi",
        "source_variety": "Tintekiya Koch",
        "role": "target",
        "language_id": "Koch",
        "glottocode": "koch1250",
        "dialect_tag": "dialect:Koch:kochbd2011-b-nokshi:Nokshi%20%28Tintekiya%20Koch%29",
        "location": "Nokshi, Jhinaigati thana, Sherpur district, Bangladesh",
        "administrative_context": "The report describes Nokshi as a western Tintekiya Koch survey village.",
        "evidence": "physical p. 15 / printed p. 14, section 3.2; physical p. 42 / printed p. 41, Appendix A.2; physical p. 87 / printed p. 86, Appendix F.1",
    },
    "c": {
        "site_id": "kochbd2011-c-kholchanda",
        "site_name": "Kholchanda",
        "source_variety": "Tintekiya Koch",
        "role": "target",
        "language_id": "Koch",
        "glottocode": "koch1250",
        "dialect_tag": "dialect:Koch:kochbd2011-c-kholchanda:Kholchanda%20%28Tintekiya%20Koch%29",
        "location": "Kholchanda, Nalitabari thana, Sherpur district, Bangladesh",
        "administrative_context": "The report describes Kholchanda as the easternmost Tintekiya Koch survey village.",
        "evidence": "physical p. 15 / printed p. 14, section 3.2; physical p. 42 / printed p. 41, Appendix A.2; physical p. 87 / printed p. 86, Appendix F.2",
    },
    "q": {
        "site_id": "kochbd2011-q-uttor-nokshi",
        "site_name": "Uttor Nokshi",
        "source_variety": "Chapra Koch",
        "role": "target",
        "language_id": "Koch",
        "glottocode": "koch1250",
        "dialect_tag": "dialect:Koch:kochbd2011-q-uttor-nokshi:Uttor%20Nokshi%20%28Chapra%20Koch%29",
        "location": "Uttor Nokshi, Jhinaigati thana, Sherpur district, Bangladesh",
        "administrative_context": "The report identifies Uttor Nokshi as the surveyed Chapra Koch village.",
        "evidence": "physical p. 15 / printed p. 14, section 3.2; physical p. 42 / printed p. 41, Appendix A.2; physical p. 89 / printed p. 88, Appendix F.4",
    },
    "r": {
        "site_id": "kochbd2011-r-chandabhoi",
        "site_name": "Chandabhoi",
        "source_variety": "Tintekiya Koch",
        "role": "target",
        "language_id": "Koch",
        "glottocode": "koch1250",
        "dialect_tag": "dialect:Koch:kochbd2011-r-chandabhoi:Chandabhoi%20%28Tintekiya%20Koch%29",
        "location": "Chandabhoi, Dalu thana, West Garo Hills, India",
        "administrative_context": "The report identifies Chandabhoi as the cross-border Tintekiya Koch comparison target.",
        "evidence": "physical p. 15 / printed p. 14, section 3.2; physical p. 42 / printed p. 41, Appendix A.2; physical p. 89 / printed p. 88, Appendix F.5",
    },
    "l": {
        "site_id": "kochbd2011-l-bharatpur",
        "site_name": "Bharatpur",
        "source_variety": "A’tong",
        "role": "control",
        "language_id": "Garo",
        "glottocode": "garo1247",
        "dialect_tag": "dialect:Garo:kochbd2011-l-bharatpur:Bharatpur%20%28A%E2%80%99tong%29",
        "location": "Bhoratpur/Bharatpur A’tong wordlist site; country and district not specified in the report's sample discussion",
        "administrative_context": "Section 3.2 spells Bhoratpur; Appendix A.2 spells Bharatpur. Both source spellings are preserved.",
        "evidence": "physical p. 15 / printed p. 14, section 3.2; physical p. 16 / printed p. 15, Figure 6; physical p. 42 / printed p. 41, Appendix A.2",
    },
    "m": {
        "site_id": "kochbd2011-m-nalchapra",
        "site_name": "Nalchapra",
        "source_variety": "A’tong",
        "role": "control",
        "language_id": "Garo",
        "glottocode": "garo1247",
        "dialect_tag": "dialect:Garo:kochbd2011-m-nalchapra:Nalchapra%20%28A%E2%80%99tong%29",
        "location": "Namchapra/Nolchapra/Nalchapra A’tong wordlist site; country and district not specified in the report's sample discussion",
        "administrative_context": "The report prints Namchapra in section 3.2, Nolchapra in Figure 6, and Nalchapra in Appendix A.2. All source spellings are preserved.",
        "evidence": "physical p. 15 / printed p. 14, section 3.2; physical p. 16 / printed p. 15, Figure 6; physical p. 42 / printed p. 41, Appendix A.2",
    },
    "0": {
        "site_id": "kochbd2011-0-bangla",
        "site_name": "Bangla",
        "source_variety": "Standard Bangla",
        "role": "control",
        "language_id": "B",
        "glottocode": "beng1280",
        "dialect_tag": "dialect:B:kochbd2011-0-bangla:Bangla%20%28Bangla%29",
        "location": "Standard Bangla comparison list; no elicitation locality claimed",
        "administrative_context": "The report calls code 0 the standard Bangla wordlist.",
        "evidence": "physical p. 7 / printed p. 6, section 1.1.1; physical p. 16 / printed p. 15, Figure 6; physical p. 42 / printed p. 41, Appendix A.2",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_manual() -> tuple[dict, list[dict[str, str]]]:
    manifest = json.loads(MANUAL_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    ranks: Counter[tuple[int, str]] = Counter()
    for chunk in manifest["manual_chunks"]:
        path = PACKAGE / chunk["expanded_cells"]
        assert sha256(path) == chunk["expanded_cells_sha256"]
        for raw in read_tsv(path):
            coordinate = (int(raw["item"]), raw["site_code"])
            ranks[coordinate] += 1
            rows.append({**raw, "manual_variant_rank": str(ranks[coordinate])})
    assert manifest["state"] == "manual_review_complete"
    assert manifest["pending_items"] == []
    assert len(rows) == 2159
    assert len({(row["item"], row["site_code"]) for row in rows}) == 2149
    return manifest, rows


def load_legacy() -> tuple[list[dict[str, str]], list[list[str]]]:
    with LEGACY_AUDIT.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    with LEGACY_INSTALLED.open(encoding="utf-8", newline="") as stream:
        installed = list(csv.reader(stream))
    assert len(audit) == 2088
    assert len(installed) == 1480
    assert all(len(row) == 15 for row in installed)
    return audit, installed


def expand_legacy(audit: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    ranks: Counter[tuple[int, str]] = Counter()
    for raw in audit:
        codes = raw["Site_Code"] or SITE_CODES
        if not raw["Site_Code"]:
            assert raw["Reason"] == "printed gap: the item was not elicited at any site"
        for code in codes:
            coordinate = (int(raw["Item"]), code)
            ranks[coordinate] += 1
            if raw["Status"] == "installed":
                status = "attested"
            elif raw["Reason"] == "contains a glyph with no verified reading":
                status = "unresolved"
            elif raw["Reason"] == "printed gap: the item was not elicited at this site":
                status = "blank"
            else:
                assert raw["Reason"] == "printed gap: the item was not elicited at any site"
                status = "not_used"
            rows.append({
                **raw,
                "expanded_site_code": code,
                "legacy_expanded_rank": str(ranks[coordinate]),
                "normalized_status": status,
                "normalized_group": raw["Group"] or "0",
            })
    assert len(rows) == 2208
    return rows


def match_legacy_to_manual(
    manual: list[dict[str, str]], legacy: list[dict[str, str]]
) -> tuple[dict[int, int | None], dict[int, list[int]]]:
    by_coordinate: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(manual):
        by_coordinate[(int(row["item"]), row["site_code"])].append(index)

    legacy_to_manual: dict[int, int | None] = {}
    manual_to_legacy: defaultdict[int, list[int]] = defaultdict(list)
    for old_index, old in enumerate(legacy):
        coordinate = (int(old["Item"]), old["expanded_site_code"])
        manual_indices = by_coordinate[coordinate]
        if old["normalized_status"] == "not_used" and not any(
            manual[index]["status"] == "not_used" for index in manual_indices
        ):
            # The legacy parser emitted a second global not-used row at items
            # 7, 10 and 12 even though the page prints ordinary responses.
            legacy_to_manual[old_index] = None
            continue
        group = old["normalized_group"]
        candidates = [
            index
            for index in manual_indices
            if group in (manual[index]["group"] or "0").split(" | ")
        ]
        if (
            not candidates
            and len(manual_indices) == 1
            and not manual[manual_indices[0]]["group"]
            and group == "1"
        ):
            # Item 34's line is visibly unnumbered; the legacy parser supplied 1.
            candidates = manual_indices
        if len(candidates) > 1:
            exact = [index for index in candidates if manual[index]["form"] == old["Raw_Form"]]
            if len(exact) == 1:
                candidates = exact
        assert len(candidates) == 1, (coordinate, old, candidates)
        current_index = candidates[0]
        legacy_to_manual[old_index] = current_index
        manual_to_legacy[current_index].append(old_index)
    assert len(manual_to_legacy) == len(manual)
    assert Counter(len(indices) for indices in manual_to_legacy.values()) == {1: 2131, 2: 28}
    assert sum(index is None for index in legacy_to_manual.values()) == 21
    return legacy_to_manual, dict(manual_to_legacy)


def assign_entry_keys(
    manual: list[dict[str, str]],
    legacy: list[dict[str, str]],
    manual_to_legacy: dict[int, list[int]],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    aliases: dict[int, list[str]] = {}
    used: defaultdict[tuple[int, str], set[int]] = defaultdict(set)
    for old in legacy:
        if old["Entry_Key"]:
            suffix = int(old["Entry_Key"].rsplit(":", 1)[1])
            used[(int(old["Item"]), old["expanded_site_code"])].add(suffix)

    assigned: dict[int, str] = {}
    pending: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(manual):
        existing = sorted({
            legacy[old_index]["Entry_Key"]
            for old_index in manual_to_legacy[index]
            if legacy[old_index]["Entry_Key"]
        }, key=lambda key: int(key.rsplit(":", 1)[1]))
        aliases[index] = existing
        if existing:
            assigned[index] = existing[0]
        else:
            pending[(int(row["item"]), row["site_code"])].append(index)

    for coordinate, indices in pending.items():
        next_variant = max(used[coordinate], default=0) + 1
        for index in indices:
            while next_variant in used[coordinate]:
                next_variant += 1
            assigned[index] = (
                f"silkochbd2011:i{coordinate[0]:03d}:{coordinate[1]}:{next_variant}"
            )
            used[coordinate].add(next_variant)
            next_variant += 1
    assert len(assigned) == len(manual)
    assert len(set(assigned.values())) == len(manual)
    return assigned, aliases


def build_reconciliation(
    manual: list[dict[str, str]],
    legacy: list[dict[str, str]],
    legacy_to_manual: dict[int, int | None],
    manual_to_legacy: dict[int, list[int]],
    entry_keys: dict[int, str],
) -> tuple[list[dict[str, str]], Counter[str], Counter[str]]:
    rows = []
    legacy_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()

    def comparison(current: dict[str, str], old: dict[str, str]) -> str:
        if current["status"] == "attested" and old["normalized_status"] == "unresolved":
            return "manual_recovered_legacy_unresolved"
        if current["status"] == "attested" and old["normalized_status"] == "attested":
            return "form_exact" if current["form"] == old["Raw_Form"] else "form_difference"
        if current["status"] == "ambiguous" and old["normalized_status"] == "unresolved":
            return "manual_ambiguous_legacy_unresolved"
        if current["status"] == "ambiguous" and old["normalized_status"] == "attested":
            return "manual_excludes_legacy_installed_ambiguous"
        if current["status"] == old["normalized_status"] == "blank":
            return "blank_match"
        if current["status"] == old["normalized_status"] == "not_used":
            return "not_used_match"
        raise AssertionError((current, old))

    for old_index, old in enumerate(legacy):
        current_index = legacy_to_manual[old_index]
        if current_index is None:
            comp = "legacy_spurious_not_used_collision"
            legacy_counts[comp] += 1
            rows.append({
                "legacy_row": str(old_index + 1),
                "item": old["Item"],
                "site_code": old["expanded_site_code"],
                "manual_entry_key": "",
                "manual_variant_rank": "",
                "manual_physical_page": "",
                "manual_printed_page": "",
                "manual_gloss": "",
                "manual_group": "",
                "manual_status": "",
                "manual_form": "",
                "manual_visible_base": "",
                "manual_note": "",
                "legacy_pdf_page": old["PDF_Page"],
                "legacy_group": old["normalized_group"],
                "legacy_status": old["normalized_status"],
                "legacy_raw_form": old["Raw_Form"],
                "legacy_entry_key": old["Entry_Key"],
                "legacy_reason": old["Reason"],
                "legacy_rows_for_manual": "0",
                "legacy_alias_retired": "false",
                "gloss_match": "",
                "group_match": "",
                "comparison": comp,
            })
            continue
        current = manual[current_index]
        comp = comparison(current, old)
        legacy_counts[comp] += 1
        canonical = entry_keys[current_index]
        rows.append({
            "legacy_row": str(old_index + 1),
            "item": old["Item"],
            "site_code": old["expanded_site_code"],
            "manual_entry_key": canonical,
            "manual_variant_rank": current["manual_variant_rank"],
            "manual_physical_page": current["physical_page"],
            "manual_printed_page": current["printed_page"],
            "manual_gloss": current["gloss"],
            "manual_group": current["group"] or "0",
            "manual_status": current["status"],
            "manual_form": current["form"],
            "manual_visible_base": current["visible_base"],
            "manual_note": current["note"],
            "legacy_pdf_page": old["PDF_Page"],
            "legacy_group": old["normalized_group"],
            "legacy_status": old["normalized_status"],
            "legacy_raw_form": old["Raw_Form"],
            "legacy_entry_key": old["Entry_Key"],
            "legacy_reason": old["Reason"],
            "legacy_rows_for_manual": str(len(manual_to_legacy[current_index])),
            "legacy_alias_retired": str(bool(old["Entry_Key"] and old["Entry_Key"] != canonical)).lower(),
            "gloss_match": str(current["gloss"] == old["Gloss"]).lower(),
            "group_match": str(
                old["normalized_group"] in (current["group"] or "0").split(" | ")
                or (not current["group"] and old["normalized_group"] == "1")
            ).lower(),
            "comparison": comp,
        })

    for current_index, current in enumerate(manual):
        olds = [legacy[index] for index in manual_to_legacy[current_index]]
        if current["status"] == "attested" and any(
            old["normalized_status"] == "unresolved" for old in olds
        ):
            comp = "manual_recovered_legacy_unresolved"
        elif current["status"] == "attested":
            comp = "form_exact" if all(current["form"] == old["Raw_Form"] for old in olds) else "form_difference"
        elif current["status"] == "ambiguous" and any(
            old["normalized_status"] == "attested" for old in olds
        ):
            comp = "manual_excludes_legacy_installed_ambiguous"
        elif current["status"] == "ambiguous":
            comp = "manual_ambiguous_legacy_unresolved"
        elif current["status"] == "blank":
            comp = "blank_match"
        else:
            assert current["status"] == "not_used"
            comp = "not_used_match"
        unique_counts[comp] += 1

    assert legacy_counts == {
        "form_exact": 705,
        "form_difference": 568,
        "manual_recovered_legacy_unresolved": 540,
        "manual_excludes_legacy_installed_ambiguous": 207,
        "manual_ambiguous_legacy_unresolved": 23,
        "blank_match": 25,
        "not_used_match": 119,
        "legacy_spurious_not_used_collision": 21,
    }
    assert unique_counts == {
        "form_exact": 696,
        "form_difference": 556,
        "manual_recovered_legacy_unresolved": 537,
        "manual_excludes_legacy_installed_ambiguous": 203,
        "manual_ambiguous_legacy_unresolved": 23,
        "blank_match": 25,
        "not_used_match": 119,
    }
    return rows, legacy_counts, unique_counts


def build_staging(
    manual: list[dict[str, str]], entry_keys: dict[int, str]
) -> tuple[list[list[str]], list[dict[str, str]], Counter[str]]:
    staged = []
    audit = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(manual):
        code = row["site_code"]
        site = SITES[code]
        key = entry_keys[index]
        if row["status"] == "ambiguous":
            disposition = "excluded_ambiguous"
        elif row["status"] == "blank":
            disposition = "excluded_blank"
        elif row["status"] == "not_used":
            disposition = "excluded_not_used"
        elif site["role"] == "control":
            disposition = "excluded_control"
        else:
            disposition = "staged_target"
            group_note = (
                f"lexical-similarity group {row['group']}"
                if row["group"]
                else "lexical-similarity group not printed"
            )
            citation = (
                f"{SOURCE_KEY}[p. {row['printed_page']}, wordlist item {row['item']}, "
                f"site {code} {site['site_name']}]"
            )
            staged.append([
                site["language_id"], "", row["form"], row["gloss"], "", row["form"],
                group_note, citation, "", "", key, "", "", "", site["dialect_tag"],
            ])
        counts[disposition] += 1
        reason = {
            "staged_target": "resolved Koch target-site attestation",
            "excluded_ambiguous": "visible base retained but an unresolved modifier is not inferred",
            "excluded_blank": "printed no-entry cell",
            "excluded_not_used": "item printed not used for all sites",
            "excluded_control": "A’tong or standard Bangla comparison form retained audit-only",
        }[disposition]
        audit.append({
            "entry_key": key,
            "item": row["item"],
            "site_code": code,
            "manual_variant_rank": row["manual_variant_rank"],
            "role": site["role"],
            "source_variety": site["source_variety"],
            "language_id": site["language_id"],
            "dialect_id": site["site_id"],
            "status": row["status"],
            "form": row["form"],
            "visible_base": row["visible_base"],
            "gloss": row["gloss"],
            "group": row["group"],
            "physical_page": row["physical_page"],
            "printed_page": row["printed_page"],
            "evidence_sha256": row["evidence_sha256"],
            "manual_note": row["note"],
            "disposition": disposition,
            "reason": reason,
        })
    assert counts == {
        "staged_target": 1017,
        "excluded_control": 772,
        "excluded_ambiguous": 226,
        "excluded_not_used": 119,
        "excluded_blank": 25,
    }
    assert len(staged) == 1017 and all(len(row) == 15 for row in staged)
    return staged, audit, counts


def write_sites() -> None:
    fields = [
        "site_code", "site_id", "site_name", "source_variety", "role", "language_id",
        "glottocode", "dialect_tag", "location", "administrative_context", "latitude",
        "longitude", "coordinate_quality", "coordinate_note", "evidence",
    ]
    rows = []
    for code in SITE_CODES:
        rows.append({
            "site_code": code,
            **SITES[code],
            "latitude": "",
            "longitude": "",
            "coordinate_quality": "",
            "coordinate_note": "The report supplies locality and map context but no exact site coordinate; no coordinate is invented.",
        })
    write_tsv(SITE_METADATA, rows, fields)


def write_reference() -> None:
    metadata = {
        "id": SOURCE_KEY,
        "entry_type": "techreport",
        "title": "The Koch of Bangladesh: A Sociolinguistic Survey",
        "authors": ["Seung Kim", "Sayed Ahmad", "Amy Kim", "Mridul Sangma"],
        "institution": "SIL International",
        "series": "SIL Electronic Survey Report",
        "number": "2011-023",
        "publication_date": "2011-03",
        "year": 2011,
        "official_archive_url": "https://www.silbangladesh.org/resources/archives/41580",
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "source_pdf_pages": 91,
        "included": "Appendix A.3 resolved attestations at Koch target codes b, c, q and r across items 1-307.",
        "excluded": "A’tong comparison codes l and m, standard Bangla code 0, printed no-entry and not-used rows, and every modifier-bearing ambiguous row remain audit-only.",
        "license": "Freely published by SIL International; only extracted linguistic facts are staged.",
        "ocr": "No. Every lexical reading was manually transcribed from rendered pages; OCR, PDF text, legacy decoding and installed forms were locator or post-freeze comparison only.",
        "provenance": [
            "data/other/forms/raw_data/sil_kochbd_2011_manual/source_manifest.json",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/reconciliation.tsv",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/staging_audit.tsv",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/staged_forms.csv",
        ],
        "etymology_provenance": "none; source lexical-similarity groups are not etymological claims",
        "jambu_editor": "Aryaman Arora and OpenAI Codex",
    }
    REFERENCE_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_exclusions(counts: Counter[str], manual: list[dict[str, str]]) -> None:
    role_status = Counter((row["role"], row["status"]) for row in manual)
    policy = {
        "source_expanded_rows": 2159,
        "source_conceptual_cells": 2149,
        "staged_rows": counts["staged_target"],
        "excluded_rows": 2159 - counts["staged_target"],
        "dispositions": dict(sorted(counts.items())),
        "role_status_rows": {
            f"{role}_{status}": count
            for (role, status), count in sorted(role_status.items())
        },
        "control_policy": "Codes l and m are A’tong comparisons and code 0 is standard Bangla; all are audit-only.",
        "blank_policy": "Printed no-entry rows remain explicit audit rows and are not staged.",
        "not_used_policy": "Every globally unused item remains seven explicit audit coordinates and is not staged.",
        "ambiguity_policy": "No unresolved modifier is inferred. All 226 ambiguous expanded rows are excluded; they represent 225 ambiguity-only conceptual cells plus the unresolved variant at mixed coordinate item 241/site r.",
        "variant_policy": "Separately printed distinct variants are retained. Identical repeated source responses merged by the frozen ledger retain all printed group labels and one immutable canonical Entry_Key; retired legacy aliases are explicit in reconciliation.tsv.",
        "etymology_policy": "No cognate or borrowing edge is inferred from lexical-similarity groups.",
    }
    EXCLUSION_POLICY.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_sound_artifacts(manual: list[dict[str, str]], staged: list[list[str]]) -> None:
    all_forms = [row["form"] for row in manual if row["status"] == "attested"]
    staged_forms = [row[2] for row in staged]
    all_counts = Counter(char for form in all_forms for char in form)
    staged_counts = Counter(char for form in staged_forms for char in form)
    inventory = [{
        "character": char,
        "codepoint": f"U+{ord(char):04X}",
        "unicode_name": unicodedata.name(char, "UNKNOWN"),
        "all_attested_count": str(all_counts[char]),
        "staged_target_count": str(staged_counts[char]),
        "combining_class": str(unicodedata.combining(char)),
    } for char in sorted(all_counts, key=ord)]
    write_tsv(SOUND_INVENTORY, inventory, [
        "character", "codepoint", "unicode_name", "all_attested_count",
        "staged_target_count", "combining_class",
    ])
    SOUND_PROFILE.write_bytes(BASE_PROFILE.read_bytes())
    decisions = {
        "input_layer": "manually transcribed source IPA",
        "output_layer": "Jambu display transcription",
        "phonemic_policy": "Preserve manual source IPA byte-for-byte in Phonemic and raw Form/Original; the profile is applied only to display Form.",
        "base_profile": "conversion/sil-bangladesh.txt",
        "base_profile_sha256": sha256(BASE_PROFILE),
        "source_local_profile": "sound_profile.txt",
        "source_local_profile_is_exact_base_snapshot": True,
        "additions": [],
        "inventory_scope": {
            "all_attested_rows": len(all_forms),
            "staged_target_rows": len(staged_forms),
            "unique_codepoints": len(inventory),
        },
        "unresolved_mappings": [],
    }
    SOUND_DECISIONS.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    manual_manifest, manual = load_manual()
    legacy_audit, legacy_installed = load_legacy()
    legacy = expand_legacy(legacy_audit)
    legacy_to_manual, manual_to_legacy = match_legacy_to_manual(manual, legacy)
    entry_keys, aliases = assign_entry_keys(manual, legacy, manual_to_legacy)
    reconciliation, reconciliation_counts, unique_counts = build_reconciliation(
        manual, legacy, legacy_to_manual, manual_to_legacy, entry_keys
    )
    write_tsv(RECONCILIATION, reconciliation, list(reconciliation[0]))
    staged, staging_audit, staging_counts = build_staging(manual, entry_keys)
    with STAGED_FORMS.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(staged)
    write_tsv(STAGING_AUDIT, staging_audit, list(staging_audit[0]))
    write_sites()
    write_reference()
    write_exclusions(staging_counts, manual)
    write_sound_artifacts(manual, staged)

    legacy_keys = {row[10] for row in legacy_installed}
    audit_keys = {row["Entry_Key"] for row in legacy_audit if row["Status"] == "installed"}
    assert legacy_keys == audit_keys and len(legacy_keys) == 1480
    target_legacy_keys = {row[10] for row in legacy_installed if row[0] == "Koch"}
    staged_keys = {row[10] for row in staged}
    canonical_legacy_target_keys = {
        entry_keys[index]
        for index, row in enumerate(manual)
        if row["role"] == "target"
        and row["status"] == "attested"
        and aliases[index]
    }
    assert len(target_legacy_keys) == 875
    assert len(canonical_legacy_target_keys) == 728
    assert len(staged_keys - target_legacy_keys) == 289
    assert len(target_legacy_keys - staged_keys) == 147
    assert sum(
        len(keys) - 1
        for index, keys in aliases.items()
        if manual[index]["role"] == "target" and keys
    ) == 20

    ambiguous_coordinates = sorted({
        f"item-{int(row['item']):03d}/site-{row['site_code']}"
        for row in manual if row["status"] == "ambiguous"
    })
    outputs = [
        RECONCILIATION, STAGING_AUDIT, STAGED_FORMS, SITE_METADATA,
        REFERENCE_METADATA, EXCLUSION_POLICY, SOUND_INVENTORY,
        SOUND_PROFILE, SOUND_DECISIONS,
    ]
    manifest = {
        "state": "source_local_post_freeze_complete",
        "policy": "Frozen manual readings are authoritative; legacy and PDF text are comparison/locator-only and supplied or verified no reading.",
        "manual_manifest_sha256": sha256(MANUAL_MANIFEST),
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "legacy_installed_sha256": sha256(LEGACY_INSTALLED),
        "legacy_audit_sha256": sha256(LEGACY_AUDIT),
        "legacy_profile_sha256": sha256(BASE_PROFILE),
        "manual_conceptual_cells": manual_manifest["conceptual_cells"],
        "manual_expanded_rows": len(manual),
        "manual_status_counts": manual_manifest["status_counts"],
        "expanded_row_status_counts": dict(sorted(Counter(row["status"] for row in manual).items())),
        "legacy_expanded_rows": len(legacy),
        "legacy_reconciliation_counts": dict(sorted(reconciliation_counts.items())),
        "manual_unique_reconciliation_counts": dict(sorted(unique_counts.items())),
        "staging_counts": dict(sorted(staging_counts.items())),
        "staged_rows": len(staged),
        "excluded_rows": len(manual) - len(staged),
        "legacy_installed_rows": len(legacy_installed),
        "legacy_audit_rows": len(legacy_audit),
        "legacy_target_key_migration": {
            "legacy_target_keys": len(target_legacy_keys),
            "retained_canonical_keys": len(canonical_legacy_target_keys),
            "new_manual_keys": len(staged_keys - target_legacy_keys),
            "retired_legacy_keys": len(target_legacy_keys - staged_keys),
            "retired_duplicate_aliases": 20,
            "retired_ambiguous_or_other": len(target_legacy_keys - staged_keys) - 20,
        },
        "site_identity_state": "resolved_from_report_sections_3_2_figure_6_and_appendix_A_2",
        "coordinate_state": "exact_coordinates_not_printed; source-local metadata leaves them blank",
        "unresolved_lexical_coordinates": ambiguous_coordinates,
        "ambiguity_only_conceptual_cells": manual_manifest["status_counts"]["ambiguous"],
        "coordinates_with_unresolved_variants": len(ambiguous_coordinates),
        "unresolved_expanded_rows": sum(row["status"] == "ambiguous" for row in manual),
        "mixed_resolved_unresolved_coordinates": ["item-241/site-r"],
        "deferred_shared_actions": [
            "replace the legacy installed source CSV with staged_forms.csv",
            "update shared dialect registry rows from site_metadata.tsv and remove invented coordinates",
            "replace the shared BibTeX entry from reference_metadata.json",
            "route sound_profile.txt explicitly for the source in the shared build",
            "update shared source checklist and integration manifest",
            "run consolidated CLDF build and opaque form-identity reconciliation",
            "run graph validation and the full test suite",
            "rebuild the browser database and perform source/language/form QA",
        ],
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    POST_FREEZE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("reconciled 2,159 manual rows against 2,208 expanded legacy rows; staged 1,017 Koch target attestations")


if __name__ == "__main__":
    main()
