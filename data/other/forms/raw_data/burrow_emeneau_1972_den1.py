#!/usr/bin/env python3
"""Install Burrow & Emeneau's 1972 *Dravidian Etymological Notes*, part I.

The copyrighted PDF is not redistributed.  The checked-in raw layer consists of one JSON file
per printed page, produced by page-isolated low-cost agents under
``burrow_emeneau_1972_den1_prompt.md``.  This importer validates that evidence, resolves the
paper's old DED/DEDS numbers against current DEDR entries, and writes conservative lexical,
entry-text, audit, sample, reconciliation, and manifest artifacts.

Run from ``data/``.  Passing the original PDF verifies its identity; CI can reproduce all
installed artifacts from the checked-in page JSON files::

    uv run python data/other/forms/raw_data/burrow_emeneau_1972_den1.py \
      --pdf "../../Downloads/Burrow-DravidianEtymologicalNotes-1972 (1).pdf"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Running this file directly places only raw_data/ on sys.path.  Add the data repository root so
# the existing ``data`` package and ``utils`` module resolve identically in direct and -m runs.
DATA_ROOT = Path(__file__).resolve().parents[4]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from data.cross_family import build_ded_resolver, source_entries
from data.dedr.abbrevs import (
    abbrevs as dedr_abbrevs,
    dialects as dedr_dialects,
    replacements as dedr_replacements,
)
from utils import change as language_changes


SOURCE_ID = "burrow-emeneau1972den1"
SNAPSHOT_DATE = "2026-08-19"
PDF_SHA256 = "c02a55891ee9a9f5b6d741eddb7d5155db312bd55b1a4a4390eb0f3a31874af1"
PDF_PAGES = 23
PRINTED_PAGES = tuple(range(397, 419))

ROOT = DATA_ROOT
RAW_DIR = ROOT / "data/other/forms/raw_data"
AGENT_DIR = RAW_DIR / "burrow_emeneau_1972_den1_agent"
FORM_OUTPUT = ROOT / "data/other/forms/20260819-burrow-emeneau-den1.csv"
TEXT_OUTPUT = ROOT / "data/other/entry_texts/20260819-burrow-emeneau-den1.csv"
AUDIT_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den1-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den1-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den1-manifest.json"
RECONCILIATION_OUTPUT = RAW_DIR / "20260819-burrow-emeneau-den1-reconciliation.json"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
TEXT_FIELDS = ["Form_ID", "Position", "Kind", "Format", "Content", "Source"]
AUDIT_FIELDS = [
    "Snapshot_Date", "Unit_ID", "Parent_Unit_ID", "PDF_Page", "Printed_Page",
    "Entry_Label", "Series", "Item_Type", "Language", "Dialect_Or_Source", "Raw_Form",
    "Raw_Gloss", "Raw_Status", "Raw_Relation", "Old_Targets", "Resolved_Targets",
    "Final_Status", "Final_Language_ID", "Final_Parameter_ID", "Emitted_Key",
    "Resolution", "Agent_Correction", "Review", "Material_Error", "Source",
    "Record_SHA256",
]

PAGE_KINDS = {"front_matter", "bibliography", "lexical_entries"}
SERIES = {"DED", "DEDS", "DBIA", "unknown"}
OPERATIONS = {
    "add_forms", "correct_form", "correct_gloss", "delete_form", "delete_entry",
    "move_or_merge", "cross_reference", "loan_reanalysis", "etymological_note",
    "new_group", "source_correction", "no_lexical_change",
}
FORM_STATUSES = {
    "active", "queried", "corrected", "deleted", "comparison_only", "loan", "reborrowed",
}
FORM_RELATIONS = {"reflex", "borrowed", "variant", "derived", "comparison_only", "unclear"}
TARGET_SYSTEMS = {"DED", "DEDS", "DBIA", "CDIAL", "none"}
LINK_RELATIONS = {
    "entry_membership", "borrowed", "variant", "derived", "compare", "move", "delete", "none",
}
CLAIM_STATUSES = {"accepted", "probable", "suggested", "queried", "rejected", "unresolved"}
EDITORIAL_ACTIONS = {"add", "correct", "delete", "move", "retain", "context_only"}

# The page agent used a non-contract label once for a tentative derivation.  Keep the raw evidence
# unchanged and make the editorial normalization explicit here.
STATUS_CORRECTIONS = {
    ("p401:u057", 0): (
        "queried",
        "agent status 'suggested' normalized to contract status 'queried'; the prose is a "
        "tentative derivation and is retained as entry text only",
    ),
    ("p402:u023", 0): (
        "queried",
        "agent status 'probable' normalized to 'queried'; *oti-ñān is a tentative "
        "reconstruction in comparative prose, not an independently attested reflex",
    ),
    ("p402:u035", 1): (
        "queried",
        "agent status 'probable' normalized to 'queried'; *ott- is explanatory "
        "reconstruction inside an already queried comparison",
    ),
}

LINK_RELATION_CORRECTIONS = {
    ("p408:u009", 0): (
        "move",
        "agent relation 'merge' normalized to controlled relation 'move'; the printed proposal "
        "to merge 2418 with 1709 is probable, and its omitted Kuwi forms remain in raw prose only",
    ),
}

# High-value image checks and unambiguous current-DEDR alignments discovered during editorial
# reconciliation.  These repair the agent field used for installation; the raw page JSON remains
# untouched and its value stays visible in the audit.
FORM_CORRECTIONS = {
    ("p399:u007", 0): ("aṛxā", "current-DEDR descendant restores the source retroflex/fricative sequence"),
    ("p399:u007", 1): ("aṛxā-cēxel", "current-DEDR descendant restores retroflexion, fricative, and length"),
    ("p399:u008", 0): ("aḍeṅgů, ḍeṅgů", "current-DEDR descendant restores both printed variants and diacritics"),
    ("p399:u011", 0): ("dāparamu", "current-DEDR descendant confirms plain d and vowel length"),
    ("p399:u011", 1): ("dāparincu", "current-DEDR descendant confirms plain d and vowel length"),
    ("p400:u057", 0): ("iḷusan", "image/current-DEDR check restored retroflex ḷ"),
    ("p401:u063", 0): ("talay-ēru", "image/current-DEDR check restored vowel length"),
    ("p402:u003", 0): ("erm ney", "agent omitted the second lexical component ney"),
    ("p402:u003", 1): ("erom nay", "agent omitted the second lexical component nay"),
    ("p402:u003", 2): ("arm/aṛm nay", "agent collapsed the printed alternant and component boundary"),
    ("p402:u008", 0): ("ēya", "image/current-DEDR check restored vowel length"),
    ("p403:u039", 0): ("ēṟu kali", "image/current-DEDR check restored ṟ"),
    ("p404:u049", 0): ("kuŋg-", "current-DEDR alignment restored the printed velar nasal"),
    ("p407:u006", 0): ("jicoṇa", "image check resolves the correction as jicona > jicoṇa"),
    ("p407:u012", 0): ("sīmpu", "image check restored vowel length"),
    ("p407:u012", 1): ("cīmpu", "image check restored vowel length"),
    ("p407:u036", 1): ("cuṭṭa", "image/current-DEDR check restored retroflexion"),
    ("p407:u036", 2): ("cuṭṭānā", "image/current-DEDR check restored retroflexion and length"),
    ("p413:u006", 0): ("boḷi", "current-DEDR descendant restores retroflex ḷ"),
}

TARGET_OVERRIDES = {
    ("p399:u007", 0): ("d59", "old 54 split; the leguminous-greens form occurs in d59"),
    ("p399:u007", 1): ("d59", "old 54 split; the plant-kingdom compound occurs in d59"),
    ("p399:u008", 0): ("d63", "old 56 split; 'hide' occurs in d63"),
    ("p399:u011", 0): ("d79", "old 69 split; 'getting, obtaining' occurs in d79"),
    ("p399:u011", 1): ("d79", "old 69 split; 'be obtained, happen' occurs in d79"),
    ("p400:u057", 0): ("d512", "old 435 split; iḷusan occurs in the fatigue/weakness descendant"),
    ("p401:u063", 0): ("d811", "old 694 split; talay-ēru 'headache' occurs in the burn descendant"),
    ("p402:u003", 0): ("d817", "old 700 split; wild-dog compound occurs in d817"),
    ("p402:u003", 1): ("d817", "old 700 split; wild-dog compound occurs in d817"),
    ("p402:u003", 2): ("d817", "old 700 split; wild-dog compound occurs in d817"),
    ("p402:u008", 0): ("d911", "old 728 split; ēya 'poor person' occurs in d911"),
    ("p403:u039", 0): ("d1379", "old 1162 split; Konda ēṟu kali occurs in d1379"),
    ("p404:u049", 0): ("d1767", "old 1472 split; Gondi kuŋg- occurs in d1767"),
    ("p406:u036", 0): ("d2335", "old 1928 split; 'beat/slap' belongs to the clapping descendant"),
    ("p406:u037", 0): ("d2338", "old 1930 split; 'lean, thin' belongs to the weakness descendant"),
    ("p407:u006", 0): ("d800", "old 2127 split; Pengo jicoṇa 'fan' occurs in d800"),
    ("p407:u012", 0): ("d2618", "old 2153 split; the nose-blowing form occurs in d2618"),
    ("p407:u012", 1): ("d2618", "old 2153 split; the nose-clearing form occurs in d2618"),
    ("p407:u036", 1): ("d2715", "old 2238 split; smoking-pipe form occurs in d2715"),
    ("p407:u036", 2): ("d2715", "old 2238 split; smoking verb occurs in d2715"),
    ("p413:u006", 0): ("d4556", "old 3722 split; Tulu boḷi 'milk' occurs in d4556"),
}

FORCE_EXCLUDE = {
    ("p407:u013", 0): (
        "combined source variants span two later DEDR descendants (jumbu/jimbu d2621; "
        "ujumbu d709); retain the unsplit page-agent unit in audit pending variant expansion"
    ),
}

# Later installments can supply reviewed record-level target sets when their printed ``S`` labels
# are new DEN sequence numbers rather than historical DEDS entry numbers.  Part I leaves this
# empty and continues to use the ordinary old-edition resolver.
RECORD_TARGET_OVERRIDES: dict[str, list[str]] = {}

LANGUAGE_ALIASES = {
    "Betta Kuruba": "BettaKurumba",
    "Betta Kurumba": "BettaKurumba",
    "Kuruba": "BettaKurumba",
    "Kurukh": "Kurux",
    "Oraon": "Kurux",
    "Kanarese": "Kannada",
    "Gadba": "Gadaba",
    "Brahui": "Brahui",
    "Dravidian": "PDr",
    "Proto-Dravidian": "PDr",
    "Sanskrit": "Sk",
    "Prakrit": "Pk",
    "Marathi": "M",
}

# These short labels are bibliographic sigla in this article, even though the inherited DEDR
# parser also associates them with a dialect.  Item/page citations make the source role explicit.
DIALECT_LABEL_DENY = {("Tamil", "rs")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_csv(
    path: Path, fields: list[str], rows: list[dict[str, str]], *, header: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def load_language_ids() -> tuple[set[str], dict[str, str]]:
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {row["ID"] for row in rows}
    by_name = {row["Name"].casefold(): row["ID"] for row in rows if row["Name"]}
    return ids, by_name


def normalized_dialect_label(value: str) -> str:
    """Normalize a short DEDR locality label without treating citations as dialects."""
    value = re.sub(r"<[^>]+>", "", value or "")
    value = re.sub(r"\bdial(?:ect)?s?\.?", "", value, flags=re.IGNORECASE)
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def load_dialect_registry() -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Return registered dialect parents and conservative source-label alignments."""
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parents = {row["ID"]: row["Language_ID"] for row in rows}
    labels: dict[tuple[str, str], str] = {}

    def register(parent: str, label: str, dialect_id: str) -> None:
        normalized = normalized_dialect_label(label)
        if normalized and dialect_id in parents and parents[dialect_id] == parent:
            key = (parent, normalized)
            # A label must be unique within its parent language to be safe for publication.
            if key not in labels or labels[key] == dialect_id:
                labels[key] = dialect_id

    for row in rows:
        register(row["Language_ID"], row["ID"], row["ID"])
        register(row["Language_ID"], row["Name"], row["ID"])

    for (raw_label, raw_language), values in dedr_dialects.items():
        dialect_id = values[1]
        parent = language_changes.get(raw_language, raw_language)
        if dialect_id:
            register(parent, raw_label, dialect_id)
    return parents, labels


def resolve_output_lect(
    language: str, source_label: str, dialect_labels: dict[tuple[str, str], str]
) -> str:
    """Resolve an unambiguous registered dialect while retaining the base language otherwise.

    Source labels mix dialect abbreviations, dictionaries, authors, and page citations.  Only an
    exact whole-label match, or one unique comma/semicolon-delimited registered dialect name, is
    promoted.  Mixed labels such as ``Onti, Tappu dialects`` deliberately remain at base-language
    level rather than acquiring an arbitrary locality.
    """
    value = re.sub(r"\s+", " ", (source_label or "").strip())
    if not value:
        return language
    value = dedr_replacements.get(value, value)
    pieces = [value, *re.split(r"[,;]", value)]
    matches = {
        dialect_labels[(language, normalized)]
        for piece in pieces
        if (normalized := normalized_dialect_label(piece))
        and (language, normalized) not in DIALECT_LABEL_DENY
        and (language, normalized) in dialect_labels
    }
    return next(iter(matches)) if len(matches) == 1 else language


def resolve_language(name: str, abbrev: str, ids: set[str], by_name: dict[str, str]) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    candidates = [
        LANGUAGE_ALIASES.get(name, name),
        by_name.get(name.casefold(), ""),
    ]
    bare_abbrev = (abbrev or "").strip().rstrip(".")
    dedr_value = dedr_abbrevs.get(bare_abbrev, dedr_abbrevs.get((abbrev or "").strip(), ""))
    candidates.extend([language_changes.get(dedr_value, dedr_value), language_changes.get(bare_abbrev, bare_abbrev)])
    for candidate in candidates:
        if candidate in ids:
            return candidate
    return ""


def load_pages() -> list[dict]:
    pages = []
    seen_units: set[str] = set()
    for printed_page in PRINTED_PAGES:
        path = AGENT_DIR / f"p{printed_page}.json"
        with path.open(encoding="utf-8") as handle:
            page = json.load(handle)
        assert page["printed_page"] == printed_page, path
        assert page["pdf_page"] == printed_page - 395, path
        assert page["page_kind"] in PAGE_KINDS, path
        if printed_page in {397, 398}:
            assert not page["records"], f"preliminary page unexpectedly has records: {path}"
        else:
            assert page["page_kind"] == "lexical_entries", path
        for ordinal, record in enumerate(page["records"], 1):
            expected = f"p{printed_page}:u{ordinal:03d}"
            assert record["unit_id"] == expected, (path, record["unit_id"], expected)
            assert record["unit_id"] not in seen_units, record["unit_id"]
            seen_units.add(record["unit_id"])
            assert record["series"] in SERIES, record["unit_id"]
            assert set(record["operations"]) <= OPERATIONS, record["unit_id"]
            for index, form in enumerate(record.get("forms", [])):
                status = form.get("form_status", "")
                if (record["unit_id"], index) not in STATUS_CORRECTIONS:
                    assert status in FORM_STATUSES, (record["unit_id"], index, status)
                assert form.get("relation_to_entry", "") in FORM_RELATIONS, (
                    record["unit_id"], index, form.get("relation_to_entry", "")
                )
            for link_index, link in enumerate(record.get("links", [])):
                assert link.get("target_system", "") in TARGET_SYSTEMS, (record["unit_id"], link)
                relation = link.get("relation", "")
                if (record["unit_id"], link_index) not in LINK_RELATION_CORRECTIONS:
                    assert relation in LINK_RELATIONS, (record["unit_id"], link)
                assert link.get("claim_status", "") in CLAIM_STATUSES, (record["unit_id"], link)
                assert link.get("editorial_action", "") in EDITORIAL_ACTIONS, (record["unit_id"], link)
        pages.append(page)
    return pages


def old_number(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^[Ss]", "", value)
    match = re.search(r"\d+", value)
    return match.group(0) if match else ""


def old_entry_keys(record: dict) -> list[tuple[str, str]]:
    edition = {"DED": "main", "DEDS": "supplement"}.get(record.get("series", ""))
    number = old_number(record.get("entry_label", ""))
    result: list[tuple[str, str]] = []
    if edition and number:
        result.append((edition, number))
    for link_index, link in enumerate(record.get("links", [])):
        link_edition = {"DED": "main", "DEDS": "supplement"}.get(link.get("target_system"))
        link_number = old_number(link.get("target_id", ""))
        relation = LINK_RELATION_CORRECTIONS.get(
            (record["unit_id"], link_index), (link.get("relation"), "")
        )[0]
        if link_edition and link_number and relation in {
            "entry_membership", "move", "delete", "variant", "derived"
        }:
            result.append((link_edition, link_number))
    return list(dict.fromkeys(result))


def primary_old_key(record: dict) -> tuple[str, str] | None:
    edition = {"DED": "main", "DEDS": "supplement"}.get(record.get("series", ""))
    number = old_number(record.get("entry_label", ""))
    if edition and number:
        return edition, number
    for link in record.get("links", []):
        edition = {"DED": "main", "DEDS": "supplement"}.get(link.get("target_system"))
        number = old_number(link.get("target_id", ""))
        if edition and number and link.get("relation") == "entry_membership":
            return edition, number
    return None


def current_targets(record: dict, resolver: dict[tuple[str, str], set[str]]) -> list[str]:
    return sorted({target for key in old_entry_keys(record) for target in resolver.get(key, ())})


def primary_targets(record: dict, resolver: dict[tuple[str, str], set[str]]) -> list[str]:
    key = primary_old_key(record)
    return sorted(resolver.get(key, ())) if key else []


def normalized_form(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "").casefold()
    value = value.lstrip("? ").replace("–", "-").replace("—", "-")
    value = re.sub(r"[\s·]+", "", value)
    return value.strip("[]() ,.;:")


def load_dedr_inventory(
    ids: set[str], by_name: dict[str, str]
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Index current DEDR raw forms for split resolution and transcription corroboration."""
    result: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as handle:
        dialect_parents = {
            row["ID"]: row["Language_ID"] for row in csv.DictReader(handle)
        }
    path = ROOT / "data/dedr/dedr_new.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            # Current DEDR rows may already use a registered dialect ID (e.g. koya, mudu).
            # The article's page extraction keeps that locality in a separate source-label field,
            # so compare at the parent-language level when disambiguating an old entry split.
            language = dialect_parents.get(row[0]) or resolve_language("", row[0], ids, by_name)
            if language:
                result[(row[1].split(".", 1)[0], language)].append((row[2], row[3]))
    return result


def form_skeleton(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = value.translate(str.maketrans({
        "ŋ": "n", "ṟ": "r", "ṯ": "t", "ṛ": "r", "ḷ": "l", "ḻ": "l",
        "ɖ": "d", "ṭ": "t", "ṇ": "n", "ñ": "n", "ζ": "z", "š": "s",
        "ï": "i", "ů": "u", "ˀ": "", "·": "",
    }))
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^a-z]+", "", value)


def gloss_words(value: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]+", form_skeleton(value))
        if len(word) > 2 and word not in {"the", "and", "with", "from", "into", "that"}
    }


def corroborate_form(
    value: str, gloss: str, target: str, language: str,
    inventory: dict[tuple[str, str], list[tuple[str, str]]],
) -> tuple[str, str]:
    """Return a source form only when the later DEDR independently corroborates it.

    A unique diacritic-insensitive match is safe because the article is the source of these DEDR
    additions; we adopt the DEDR string as the corrected diplomatic field and retain the agent's
    raw string in the audit.  Looser candidates remain audit-only.
    """
    candidates = inventory.get((target, language), [])
    normalized = normalized_form(value)
    exact = list(dict.fromkeys(form for form, _ in candidates if normalized_form(form) == normalized))
    if exact:
        return value, "agent transcription exactly corroborated by current DEDR"
    skeleton = form_skeleton(value)
    skeleton_matches = list(dict.fromkeys(
        form for form, _ in candidates if skeleton and form_skeleton(form) == skeleton
    ))
    if len(skeleton_matches) == 1:
        return skeleton_matches[0], "agent structure corroborated; diacritics restored from unique current-DEDR match"
    if len(skeleton_matches) > 1:
        wanted = gloss_words(gloss)
        gloss_matches = list(dict.fromkeys(
            form for form, candidate_gloss in candidates
            if form in skeleton_matches and wanted & gloss_words(candidate_gloss)
        ))
        if len(gloss_matches) == 1:
            return gloss_matches[0], "diacritics restored from unique form-and-gloss current-DEDR match"
    return "", "agent transcription lacks a unique exact or diacritic-insensitive current-DEDR corroboration"


def choose_target(
    candidates: list[str], language: str, form: str, dedr_forms: dict[tuple[str, str, str], int]
) -> tuple[str, str]:
    if len(candidates) == 1:
        return candidates[0], "unique old-number resolution"
    if not candidates:
        return "", "old DED/DEDS number is absent from current DEDR"
    norm = normalized_form(form)
    matches = [candidate for candidate in candidates if dedr_forms.get((candidate, language, norm), 0)]
    if len(matches) == 1:
        return matches[0], "split old entry resolved by exact language-and-form match in current DEDR"
    if not matches:
        return "", "old entry split in current DEDR; no exact language-and-form match"
    return "", "old entry split in current DEDR; exact form occurs in multiple descendants"


def text_kind(record: dict) -> str:
    operations = set(record.get("operations", []))
    if operations & {"delete_form", "delete_entry", "move_or_merge", "correct_form", "correct_gloss", "source_correction"}:
        return "correction"
    if "loan_reanalysis" in operations:
        return "analysis"
    if operations & {"cross_reference", "etymological_note"}:
        return "comparison"
    return "source-note"


def source_locator(record: dict, printed_page: int) -> str:
    return record.get("source_locator") or (
        f"{SOURCE_ID}[p. {printed_page}, entry {record.get('entry_label', '?')}]"
    )


def audit_row(
    *, page: dict, record: dict, unit_id: str, parent: str, item_type: str,
    language: str = "", dialect: str = "", raw_form: str = "", raw_gloss: str = "",
    raw_status: str = "", raw_relation: str = "", old_targets: list[tuple[str, str]],
    resolved_targets: list[str], final_status: str, final_language: str = "",
    final_parameter: str = "", emitted_key: str = "", resolution: str,
    correction: str = "", value_hash: str,
) -> dict[str, str]:
    review = (
        "source-image/current-DEDR reconciled"
        if final_status == "installed_form"
        else "page-agent structure reviewed; running text pending diplomatic verification"
        if final_status == "raw_segment_audited"
        else "source structure reviewed; exclusion or unresolved state is explicit"
    )
    material_error = "unreviewed" if final_status == "raw_segment_audited" else "no"
    return dict(zip(AUDIT_FIELDS, [
        SNAPSHOT_DATE, unit_id, parent, str(page["pdf_page"]), str(page["printed_page"]),
        record.get("entry_label", ""), record.get("series", ""), item_type, language, dialect,
        raw_form, raw_gloss, raw_status, raw_relation,
        "|".join(f"{edition}:{number}" for edition, number in old_targets),
        "|".join(resolved_targets), final_status, final_language, final_parameter, emitted_key,
        resolution, correction, review, material_error,
        source_locator(record, page["printed_page"]), value_hash,
    ]))


def build(pages: list[dict]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict]:
    language_ids, language_by_name = load_language_ids()
    dialect_parents, dialect_labels = load_dialect_registry()
    dedr_inventory = load_dedr_inventory(language_ids, language_by_name)
    dedr_forms = {
        (target, language, normalized_form(form)): 1
        for (target, language), values in dedr_inventory.items()
        for form, _ in values
    }
    resolver = build_ded_resolver(source_entries("dedr"))

    forms: list[dict[str, str]] = []
    texts: list[dict[str, str]] = []
    audits: list[dict[str, str]] = []
    seen_form_keys: dict[tuple[str, str, str, str, str], str] = {}
    exclusions = Counter()
    resolutions = Counter()

    for page in pages:
        for ordinal, record in enumerate(page["records"], 1):
            unit_id = record["unit_id"]
            old_targets = old_entry_keys(record)
            if unit_id in RECORD_TARGET_OVERRIDES:
                resolved = primary = RECORD_TARGET_OVERRIDES[unit_id]
            else:
                resolved = current_targets(record, resolver)
                primary = primary_targets(record, resolver)
            locator = source_locator(record, page["printed_page"])
            raw_text = (record.get("raw_entry_text") or record.get("comparison_or_correction_text") or "").strip()

            # Page-agent prose is retained as raw/audit evidence, not published as a diplomatic
            # entry-text block.  The pilot reliably extracts structure, but its running-text
            # diacritics are not accurate enough without line-by-line editorial review.
            record_status = "raw_segment_audited" if raw_text else "no_lexical_content"
            record_resolution = (
                f"structured raw segment retained in audit; {len(resolved)} current DEDR target(s) resolved"
                if raw_text else
                "preliminary or empty page-local context; no lexical installation"
            )
            audits.append(audit_row(
                page=page, record=record, unit_id=unit_id, parent="", item_type="record",
                old_targets=old_targets, resolved_targets=resolved, final_status=record_status,
                resolution=record_resolution, value_hash=record_hash(record),
            ))

            for form_index, form in enumerate(record.get("forms", []), 1):
                child_id = f"{unit_id}:f{form_index:03d}"
                raw_status = form.get("form_status", "")
                correction = ""
                if (unit_id, form_index - 1) in STATUS_CORRECTIONS:
                    raw_status, correction = STATUS_CORRECTIONS[(unit_id, form_index - 1)]
                raw_relation = form.get("relation_to_entry", "")
                language = resolve_language(
                    form.get("language_name", ""), form.get("language_abbrev", ""),
                    language_ids, language_by_name,
                )
                dialect = form.get("dialect_or_source_label", "")
                output_lect = resolve_output_lect(language, dialect, dialect_labels)
                raw_form = form.get("form_original", "")
                raw_gloss = form.get("gloss", "")
                final_form = raw_form
                if (unit_id, form_index - 1) in FORM_CORRECTIONS:
                    final_form, form_correction = FORM_CORRECTIONS[(unit_id, form_index - 1)]
                    correction = "; ".join(filter(None, [correction, form_correction]))

                accepted = raw_status in {"active", "corrected"} and raw_relation in {
                    "reflex", "variant", "derived"
                }
                if not accepted:
                    reason = {
                        "queried": "tentative form retained in source prose without a rank-1 reflex edge",
                        "deleted": "source-deleted form retained in the audit and correction prose only",
                        "comparison_only": "comparison form retained as prose, not installed as a reflex",
                        "loan": "loan reanalysis retained as typed source prose, not an inherited reflex",
                        "reborrowed": "reborrowing analysis retained as typed source prose, not an inherited reflex",
                    }.get(raw_status, "non-reflex or unclear form retained in source prose only")
                    exclusion_key = raw_status or raw_relation or "other"
                    if raw_status in {"active", "corrected"}:
                        exclusion_key = f"{raw_status}_{raw_relation or 'unclear'}"
                    exclusions[exclusion_key] += 1
                    audits.append(audit_row(
                        page=page, record=record, unit_id=child_id, parent=unit_id,
                        item_type="form", language=form.get("language_name", ""), dialect=dialect,
                        raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                        raw_relation=raw_relation, old_targets=old_targets,
                        resolved_targets=primary, final_status="excluded_nonaccepted",
                        resolution=reason, correction=correction, value_hash=record_hash(form),
                    ))
                    continue

                if not language:
                    exclusions["unresolved_language"] += 1
                    audits.append(audit_row(
                        page=page, record=record, unit_id=child_id, parent=unit_id,
                        item_type="form", language=form.get("language_name", ""), dialect=dialect,
                        raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                        raw_relation=raw_relation, old_targets=old_targets,
                        resolved_targets=primary, final_status="unresolved_language",
                        resolution="language/source label does not resolve to a Jambu language",
                        correction=correction, value_hash=record_hash(form),
                    ))
                    continue

                if (unit_id, form_index - 1) in FORCE_EXCLUDE:
                    reason = FORCE_EXCLUDE[(unit_id, form_index - 1)]
                    exclusions["variant_split_pending"] += 1
                    audits.append(audit_row(
                        page=page, record=record, unit_id=child_id, parent=unit_id,
                        item_type="form", language=form.get("language_name", ""), dialect=dialect,
                        raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                        raw_relation=raw_relation, old_targets=old_targets,
                        resolved_targets=primary, final_status="variant_split_pending",
                        final_language=output_lect, resolution=reason,
                        correction=correction, value_hash=record_hash(form),
                    ))
                    continue

                if (unit_id, form_index - 1) in TARGET_OVERRIDES:
                    target, target_resolution = TARGET_OVERRIDES[(unit_id, form_index - 1)]
                    assert target in primary, (unit_id, target, primary)
                else:
                    target, target_resolution = choose_target(primary, language, final_form, dedr_forms)
                resolutions[target_resolution] += 1
                if not target:
                    exclusions["unresolved_target"] += 1
                    audits.append(audit_row(
                        page=page, record=record, unit_id=child_id, parent=unit_id,
                        item_type="form", language=form.get("language_name", ""), dialect=dialect,
                        raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                        raw_relation=raw_relation, old_targets=old_targets,
                        resolved_targets=primary, final_status="unresolved_target",
                        final_language=output_lect, resolution=target_resolution,
                        correction=correction, value_hash=record_hash(form),
                    ))
                    continue

                corroborated_form, corroboration = corroborate_form(
                    final_form, raw_gloss, target, language, dedr_inventory
                )
                if not corroborated_form:
                    exclusions["unreconciled_transcription"] += 1
                    audits.append(audit_row(
                        page=page, record=record, unit_id=child_id, parent=unit_id,
                        item_type="form", language=form.get("language_name", ""), dialect=dialect,
                        raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                        raw_relation=raw_relation, old_targets=old_targets,
                        resolved_targets=primary, final_status="unreconciled_transcription",
                        final_language=output_lect, final_parameter=target,
                        resolution=f"{target_resolution}; {corroboration}",
                        correction=correction, value_hash=record_hash(form),
                    ))
                    continue
                if corroborated_form != final_form:
                    correction = "; ".join(filter(None, [
                        correction,
                        f"agent form {raw_form!r} reconciled as {corroborated_form!r}",
                    ]))
                final_form = corroborated_form

                dedupe_key = (
                    output_lect, target, normalized_form(final_form),
                    raw_gloss.casefold().strip(), raw_relation,
                )
                if dedupe_key in seen_form_keys:
                    exclusions["duplicate"] += 1
                    audits.append(audit_row(
                        page=page, record=record, unit_id=child_id, parent=unit_id,
                        item_type="form", language=form.get("language_name", ""), dialect=dialect,
                        raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                        raw_relation=raw_relation, old_targets=old_targets,
                        resolved_targets=primary, final_status="duplicate_excluded",
                        final_language=output_lect, final_parameter=target,
                        emitted_key=seen_form_keys[dedupe_key],
                        resolution="exact source duplicate already emitted from an earlier page-local unit",
                        correction=correction, value_hash=record_hash(form),
                    ))
                    continue

                key = f"{SOURCE_ID}:{child_id}"
                seen_form_keys[dedupe_key] = key
                notes = "; ".join(filter(None, [
                    f"source label: {dialect}" if dialect else "",
                    form.get("grammatical_information", ""),
                    form.get("source_detail", ""),
                ]))
                old_key = primary_old_key(record)
                old_label = f"{old_key[0]} {old_key[1]}" if old_key else record.get("entry_label", "")
                etymology = (
                    f"DEN I (1972) records this {raw_relation} under old {old_label}; "
                    f"resolved to current DEDR {target}."
                )
                tags = " ".join(filter(None, [
                    "source-addition" if raw_status == "active" else "source-correction",
                    raw_relation if raw_relation != "reflex" else "",
                ]))
                forms.append(dict(zip(FORM_FIELDS, [
                    output_lect, target, final_form, raw_gloss, "", "", notes, locator, "",
                    etymology, key, "", "", "", tags,
                ])))
                audits.append(audit_row(
                    page=page, record=record, unit_id=child_id, parent=unit_id,
                    item_type="form", language=form.get("language_name", ""), dialect=dialect,
                    raw_form=raw_form, raw_gloss=raw_gloss, raw_status=raw_status,
                    raw_relation=raw_relation, old_targets=old_targets,
                    resolved_targets=primary, final_status="installed_form",
                    final_language=output_lect, final_parameter=target, emitted_key=key,
                    resolution=f"{target_resolution}; {corroboration}", correction=correction,
                    value_hash=record_hash(form),
                ))

    summary = {
        "page_count": len(pages),
        "record_count": sum(len(page["records"]) for page in pages),
        "raw_form_count": sum(len(record.get("forms", [])) for page in pages for record in page["records"]),
        "installed_form_count": len(forms),
        "entry_text_count": len(texts),
        "audit_count": len(audits),
        "installed_dialect_form_count": sum(
            row["Language_ID"] in dialect_parents for row in forms
        ),
        "installed_lect_counts": dict(sorted(Counter(
            row["Language_ID"] for row in forms
        ).items())),
        "exclusions": dict(sorted(exclusions.items())),
        "target_resolutions": dict(sorted(resolutions.items())),
    }
    return forms, texts, audits, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="optional original PDF for identity verification")
    args = parser.parse_args()
    if args.pdf:
        assert args.pdf.is_file(), args.pdf
        actual = sha256(args.pdf)
        if actual != PDF_SHA256:
            raise ValueError(f"PDF SHA-256 {actual} does not match expected {PDF_SHA256}")

    pages = load_pages()
    forms, texts, audits, summary = build(pages)
    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert all(
        row["Material_Error"] == "no"
        for row in audits if row["Final_Status"] == "installed_form"
    )
    assert len(audits) == summary["record_count"] + summary["raw_form_count"]

    # Manual form imports are headerless because make_cldf.py reads rich rows positionally.
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(TEXT_OUTPUT, TEXT_FIELDS, texts)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audits)
    sample = sorted(audits, key=lambda row: hashlib.sha256(row["Unit_ID"].encode()).hexdigest())[:20]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample)

    RECONCILIATION_OUTPUT.write_text(
        json.dumps({
            "source_id": SOURCE_ID,
            "snapshot_date": SNAPSHOT_DATE,
            "policy": (
                "Page-agent JSON is raw evidence. Only active/corrected direct Dravidian reflex, "
                "variant, or derivational forms with conservative current-DEDR resolution are installed."
            ),
            "known_agent_corrections": [
                {"unit_id": unit_id, "form_index": index + 1, "normalized_status": value[0], "decision": value[1]}
                for (unit_id, index), value in sorted(STATUS_CORRECTIONS.items())
            ] + [
                {"unit_id": unit_id, "link_index": index + 1, "normalized_relation": value[0], "decision": value[1]}
                for (unit_id, index), value in sorted(LINK_RELATION_CORRECTIONS.items())
            ],
            **summary,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MANIFEST_OUTPUT.write_text(
        json.dumps({
            "source_id": SOURCE_ID,
            "snapshot_date": SNAPSHOT_DATE,
            "title": "Dravidian Etymological Notes: Supplement to DED, DEDS, and DBIA, Pt. I",
            "authors": ["T. Burrow", "M. B. Emeneau"],
            "year": 1972,
            "stable_url": "https://www.jstor.org/stable/600566",
            "doi": "10.2307/600566",
            "pdf_sha256": PDF_SHA256,
            "pdf_pages": PDF_PAGES,
            "article_printed_pages": [397, 418],
            "pdf_redistributed": False,
            "rights": "Copyright JSTOR/JAOS scan; only extracted linguistic facts and audit metadata are checked in.",
            "extraction": {
                "method": "one gpt-5.6-luna agent per rendered printed page, followed by editorial reconciliation",
                "contract": "data/other/forms/raw_data/burrow_emeneau_1972_den1_prompt.md",
                "raw_page_directory": "data/other/forms/raw_data/burrow_emeneau_1972_den1_agent",
            },
            "outputs": {
                "forms": str(FORM_OUTPUT.relative_to(ROOT)),
                "entry_texts": str(TEXT_OUTPUT.relative_to(ROOT)),
                "audit": str(AUDIT_OUTPUT.relative_to(ROOT)),
                "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
                "reconciliation": str(RECONCILIATION_OUTPUT.relative_to(ROOT)),
            },
            **summary,
            "sample_count": len(sample),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"installed {len(forms)} forms and {len(texts)} entry-text blocks from "
        f"{summary['record_count']} numbered page segments; audited {len(audits)} units"
    )


if __name__ == "__main__":
    main()
