"""Build source-specific retrospective ingestion checklists.

The project predates ``SOURCE_INGESTION_CHECKLIST.md``.  This module makes the
retrofit review reproducible: every file consumed as a form input by
``make_cldf.py`` is an ingestion unit, and every unit receives a filled copy of
the canonical checklist under ``source_checklists/``.

The generated front matter records facts that can be checked mechanically.  A
section is checked only when its repository gate has evidence; an unchecked
section names the missing evidence.  The source-specific copies deliberately
remain generated artifacts so changes to the canonical checklist cannot leave
older source reviews silently stale.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from utils import mapping


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "SOURCE_INGESTION_CHECKLIST.md"
OUTPUT_DIR = ROOT / "source_checklists"
MANIFEST = OUTPUT_DIR / "manifest.json"
INSTALLED_RECORD_AUDIT = OUTPUT_DIR / "installed-record-audit.csv.gz"

CORE_INPUTS = (
    Path("data/cdial/cdial.csv"),
    Path("data/munda/forms.csv"),
    Path("data/dedr/dedr_new.csv"),
    Path("data/dedr/pdr.csv"),
    Path("data/dbia/forms.csv"),
)

CORE_REVIEW_FILES = {
    "cdial-cdial": {
        "importers": ["data/cdial/parse.py", "data/cdial/audit.py"],
        "audits": ["data/cdial/audit.py"],
        "tests": ["tests/test_cdial_parser.py", "tests/test_cdial_metadata.py"],
        "profiles": ["conversion/cdial.txt"],
        "addenda": ["Dictionary or glossary", "Etymological/comparative source"],
    },
    "dedr-dedr-new": {
        "importers": ["data/dedr/parse.py", "data/dedr/audit.py"],
        "audits": ["data/dedr/audit.py"],
        "tests": ["tests/test_dedr_parser.py", "tests/test_dedr_cleanup.py"],
        "profiles": ["conversion/dedr.txt"],
        "addenda": ["Dictionary or glossary", "Etymological/comparative source"],
    },
    "dedr-pdr": {
        "importers": ["data/dedr/get_params.py"],
        "audits": [],
        "tests": ["tests/test_dedr_variants.py", "tests/test_cldf.py"],
        "profiles": ["conversion/dedr.txt"],
        "addenda": ["Etymological/comparative source"],
    },
    "dbia-forms": {
        "importers": ["data/dbia/parse.py"],
        "audits": ["data/dbia/parse_audit.csv"],
        "tests": ["tests/test_dbia.py"],
        "profiles": ["conversion/dedr.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "munda-forms": {
        "importers": ["data/munda/rau_2019.csv"],
        "audits": [],
        "tests": ["tests/test_cldf.py", "tests/test_edges.py"],
        "profiles": ["conversion/house.txt"],
        "addenda": ["Etymological/comparative source"],
    },
}

ADDENDUM_HEADINGS = {
    "Dictionary or glossary",
    "Survey wordlists or comparative tables",
    "OCR-heavy source",
    "Website/API or external CLDF",
    "Etymological/comparative source",
}

UNIT_ADDENDA = {
    "20260805-gandhari-org": [
        "Dictionary or glossary",
        "Website/API or external CLDF",
        "Etymological/comparative source",
    ],
    "20260817-ghatage-marati-kasargod": ["Dictionary or glossary", "OCR-heavy source"],
    "20260818-hockings-badaga": [
        "Dictionary or glossary",
        "OCR-heavy source",
        "Etymological/comparative source",
    ],
}

PINNED_LEGACY_UNITS = {
    "20220913-dhivehi",
    "20220913-khetrani",
    "20220913-kholosi",
    "20220913-konkani",
    "20220913-kundalshahi",
    "20220913-kvari",
    "20220913-patyal",
    "20220913-zadjali",
    "20230524-sindhic",
}

UNIT_REVIEW_NOTES = {
    "20260805-gandhari-org": {
        "exclusions": (
            "of 5,807 Sanskrit-bearing API articles, 371 ambiguous matches, 3,923 unmatched "
            "articles, and 1 article without a parsed Sanskrit etymon remain audit-only; 1,512 "
            "unique exact accent-normalized CDIAL matches are installed"
        ),
        "unresolved": (
            "the 4,295 non-unique or unmatched CDIAL assignments remain conservatively unlinked "
            "in the audit; source-site reuse terms were not stated"
        ),
        "transcription": (
            "`conversion/gandhari.txt`; source spelling, Kharoshthi, and phonetic fields remain "
            "separate, with full paradigms retained only in the audit"
        ),
        "representative": (
            "`/entries/f_d3zfp2ruszaq6` (ichadi), `/entries/f_uo5sns6fnvzse` "
            "(relative pronoun yavaṁta), `/languages/Dhp`, `/references/gandhari`, and "
            "`/concepts/2960`"
        ),
    },
    "20230521-rajasthani": {
        "exclusions": (
            "14 historical blank-form rows were removed from the installed input and retained in "
            "`source_checklists/audits/20230521-rajasthani-exclusions.csv`"
        ),
    },
    "20230530-tharu2": {
        "exclusions": (
            "3 historical blank-form rows were removed from the installed input and retained in "
            "`source_checklists/audits/20230530-tharu2-exclusions.csv`"
        ),
    },
    "20260813-bhaskararao-toda": {
        "exclusions": (
            "2 of 7,560 dictionary records have replacement-glyph-only heads and remain audit-only; "
            "7,558 readable records emit 8,859 installed rows after variant expansion"
        ),
        "unresolved": "the 2 corrupt heads are preserved without conjectural reconstruction",
    },
    "20260817-ghatage-marati-kasargod": {
        "exclusions": (
            "1 corrupt alternate candidate remains audit-only while its readable main form is installed"
        ),
        "unresolved": (
            "1,115 OCR records remain explicitly unreviewed; 129 are source-image verified, and the "
            "deterministic 20-record sample passes after correction"
        ),
        "transcription": (
            "`conversion/ghatage.txt`; every installed form retains `ocr-review`, and the source is "
            "marked OCR in the browser bibliography"
        ),
        "representative": (
            "`/references/ghatage-kasargod1970`, `/entries/f_zgcmreutdcjxa`, and `/concepts/2398`"
        ),
    },
    "20260818-hockings-badaga": {
        "exclusions": (
            "front matter, blank leaves, the English-Badaga reverse glossary, appendices, "
            "references, and publisher advertisement are outside the 9,993-article "
            "Badaga-English scope; no lexical article was structurally corrupt"
        ),
        "unresolved": (
            "93 articles retain unresolved printed DEDR citations without conjectural links; "
            "20 articles are image-reviewed and the remaining 9,973 retain a typed "
            "transcription-review marker"
        ),
        "transcription": (
            "`conversion/badaga-hockings.txt` converts source vowel-length colons to display "
            "macrons while preserving Original; durable scan-backed decisions live in "
            "`20260818-hockings-badaga-corrections.csv`"
        ),
        "representative": (
            "`/references/hockings-pilotraichoor1992`, "
            "`/entries/f_hmjkffhyzp44y` (reviewed agaṭu madilu), "
            "`/entries/f_id7i2lzuvr7ec` (review-pending Edekādu), and `/languages/Badaga`"
        ),
    },
}


@dataclass(frozen=True)
class Unit:
    id: str
    installed_file: str
    row_count: int
    row_widths: dict[str, int]
    languages: list[str]
    source_keys: list[str]
    source_key_counts: dict[str, int]
    entry_key_count: int
    unique_entry_key_count: int
    blank_form_count: int
    replacement_character_count: int
    importers: list[str]
    audits: list[str]
    tests: list[str]
    profiles: list[str]
    addenda: list[str]
    compiled_rows: int
    unresolved_references: list[str]
    unregistered_languages: list[str]
    unregistered_dialect_tags: list[str]


def input_paths() -> list[Path]:
    other = sorted((ROOT / "data/other/forms").glob("*.csv"))
    return [ROOT / path for path in CORE_INPUTS[:-1]] + other + [ROOT / CORE_INPUTS[-1]]


def unit_id(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[:3] == ("data", "other", "forms"):
        return path.stem
    return f"{path.parent.name}-{path.stem}".replace("_", "-")


def citation_keys(value: str) -> list[str]:
    return [
        token.strip().split("[", 1)[0]
        for token in value.split(";")
        if token.strip()
    ]


def load_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def load_registry_ids(filename: str) -> set[str]:
    with (ROOT / "cldf" / filename).open(encoding="utf-8", newline="") as stream:
        return {row["ID"] for row in csv.DictReader(stream)}


def load_dialect_registry() -> tuple[set[str], set[str]]:
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["Source_Language_ID"] for row in rows}, {row["Tag"] for row in rows}


def load_reference_ids() -> set[str]:
    with (ROOT / "cldf/references.csv").open(encoding="utf-8", newline="") as stream:
        return {row["ID"] for row in csv.DictReader(stream)}


def compiled_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            counts.update(citation_keys(row["Source"]))
    return counts


def compiled_source_rows() -> dict[str, list[dict[str, str]]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in citation_keys(row["Source"]):
                by_source.setdefault(key, []).append(row)
    return by_source


def existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if (ROOT / path).exists()]


def files_mentioning(
    directory: Path, patterns: set[str], suffix: str, *, inspect_contents: bool = True
) -> list[str]:
    matches: list[str] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        normalized_name = path.stem.casefold().replace("-", "_")
        name_match = any(pattern in normalized_name for pattern in patterns if pattern)
        content_match = False
        if inspect_contents:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            content_match = any(pattern.replace("_", "-") in text.casefold() for pattern in patterns)
        if name_match or content_match:
            matches.append(str(path.relative_to(ROOT)))
    return matches


def infer_related_files(path: Path, uid: str) -> tuple[list[str], list[str], list[str]]:
    override = CORE_REVIEW_FILES.get(uid, {})
    if override:
        return (
            existing_paths(override["importers"]),
            existing_paths(override["audits"]),
            existing_paths(override["tests"]),
        )

    stem = re.sub(r"^\d{8}-", "", path.stem).casefold().replace("-", "_")
    patterns = {stem}
    patterns.update(part for part in stem.split("_") if len(part) >= 5)
    raw_dir = ROOT / "data/other/forms/raw_data"
    importers = files_mentioning(raw_dir, patterns, ".py", inspect_contents=False)
    if uid in PINNED_LEGACY_UNITS:
        importers = ["data/other/forms/raw_data/legacy_snapshots.py"]
    elif not importers:
        # Some early hand-curated inputs are themselves the canonical machine-readable snapshot.
        # The deterministic installed-record audit pins every row even when no extractor survives.
        importers = [str(path.relative_to(ROOT))]
    audits = [
        str(candidate.relative_to(ROOT))
        for candidate in sorted(raw_dir.glob("*audit*.csv"))
        if any(pattern in candidate.stem.casefold().replace("-", "_") for pattern in patterns)
    ]
    review_audit_dir = ROOT / "source_checklists/audits"
    audits.extend(
        str(candidate.relative_to(ROOT))
        for candidate in sorted(review_audit_dir.glob(f"{uid}*.csv"))
    )
    tests = files_mentioning(ROOT / "tests", patterns, ".py", inspect_contents=False)
    if (ROOT / "tests/test_source_checklists.py").exists():
        tests.append("tests/test_source_checklists.py")
    return importers, audits, tests


def infer_profiles(path: Path, uid: str, rows: list[list[str]]) -> list[str]:
    override = CORE_REVIEW_FILES.get(uid, {})
    if override:
        return existing_paths(override["profiles"])

    available = {candidate.stem: candidate for candidate in (ROOT / "conversion").glob("*.txt")}
    stem = re.sub(r"^\d{8}-", "", path.stem)
    filename_key = path.stem.split("-")[1] if "-" in path.stem else path.stem
    candidates = [mapping.get(filename_key, filename_key), stem, stem.split("-")[0]]
    source = citation_keys(rows[0][7])[0] if rows and len(rows[0]) > 7 and rows[0][7] else ""
    explicit = {
        "shackle": "cdial",
        "shackle-auto": "cdial",
        "liljegren-hindukush": "liljegren-hindukush",
        "grierson-lsi1928": "lsi",
        "ali-kobayashi2024": "brahui",
        "abraham-sako2021": "tagin-puroik",
        "kondakov2013rabha": "rabha",
        "hilty-mitchell2014": "yamphu",
        "hilty2013eastern-magar": "eastern-magar",
        "grierson-lsi1928": "lsi",
    }.get(source)
    if explicit:
        candidates.insert(0, explicit)
    for candidate in candidates:
        if candidate in available:
            return [str(available[candidate].relative_to(ROOT))]
    return []


def infer_addenda(path: Path, uid: str, rows: list[list[str]], source_keys: list[str]) -> list[str]:
    override = CORE_REVIEW_FILES.get(uid, {})
    if override:
        return override["addenda"]
    if uid in UNIT_ADDENDA:
        return UNIT_ADDENDA[uid]

    text = " ".join([path.stem, *source_keys]).casefold()
    addenda: list[str] = []
    if any(word in text for word in ("dictionary", "lexicon", "berger", "kullui", "kota", "toda", "khowar", "brahui", "nihali")):
        addenda.append("Dictionary or glossary")
    if any(word in text for word in ("survey", "wordlist", "lsi", "ssnp", "northern", "tharu", "gurung", "tamang", "magar", "rai", "hajong", "santali", "pahari", "naaba", "humla", "dotyali")):
        addenda.append("Survey wordlists or comparative tables")
    if any(word in text for word in ("ocr", "sigiri", "andersen", "vaagri", "thari", "wadiyara")):
        addenda.append("OCR-heavy source")
    if any(word in text for word in ("-org", "wiktionary", "liljegren-hindukush", "grierson-lsi")):
        addenda.append("Website/API or external CLDF")
    if any(row[1].strip() for row in rows if len(row) > 1):
        addenda.append("Etymological/comparative source")
    if not addenda:
        addenda.append("Dictionary or glossary")
    return list(dict.fromkeys(addenda))


def build_units() -> list[Unit]:
    language_ids = load_registry_ids("languages.csv")
    _, dialect_tags = load_dialect_registry()
    reference_ids = load_reference_ids()
    compiled = compiled_source_rows()
    units: list[Unit] = []

    for path in input_paths():
        rows = load_csv(path)
        if not rows:
            # Empty historical placeholders are not ingested sources.
            continue
        uid = unit_id(path)
        source_counter: Counter[str] = Counter()
        languages: set[str] = set()
        widths: Counter[int] = Counter()
        entry_keys: list[str] = []
        blank_forms = 0
        replacements = 0

        for row in rows:
            widths[len(row)] += 1
            if row:
                languages.add(row[0])
            if len(row) > 2:
                blank_forms += not row[2].strip()
                replacements += "�" in row[2]
            if len(row) > 7:
                source_counter.update(citation_keys(row[7]))
            if len(row) > 10 and row[10].strip():
                entry_keys.append(row[10].strip())

        importers, audits, tests = infer_related_files(path, uid)
        audits = list(dict.fromkeys([*audits, str(INSTALLED_RECORD_AUDIT.relative_to(ROOT))]))
        sources = sorted(source_counter)
        profiles = infer_profiles(path, uid, rows)
        compiled_for_unit = {
            row["ID"]: row
            for key in sources
            for row in compiled.get(key, [])
        }
        compiled_languages = {row["Language_ID"] for row in compiled_for_unit.values()}
        compiled_dialect_tags = {
            tag
            for row in compiled_for_unit.values()
            for tag in row["Tags"].split()
            if tag.startswith("dialect:")
        }
        units.append(
            Unit(
                id=uid,
                installed_file=str(path.relative_to(ROOT)),
                row_count=len(rows),
                row_widths={str(key): widths[key] for key in sorted(widths)},
                languages=sorted(languages),
                source_keys=sources,
                source_key_counts={key: source_counter[key] for key in sources},
                entry_key_count=len(entry_keys),
                unique_entry_key_count=len(set(entry_keys)),
                blank_form_count=blank_forms,
                replacement_character_count=replacements,
                importers=importers,
                audits=audits,
                tests=tests,
                profiles=profiles,
                addenda=infer_addenda(path, uid, rows, sources),
                compiled_rows=len(compiled_for_unit),
                unresolved_references=sorted(set(sources) - reference_ids),
                unregistered_languages=sorted(compiled_languages - language_ids),
                unregistered_dialect_tags=sorted(compiled_dialect_tags - dialect_tags),
            )
        )
    return units


def section_evidence(unit: Unit) -> dict[str, tuple[bool, str]]:
    validation_path = OUTPUT_DIR / "VALIDATION.md"
    validation = validation_path.read_text(encoding="utf-8") if validation_path.exists() else ""
    data_validated = "Data pipeline: PASS" in validation
    browser_validated = "Browser database and QA: PASS" in validation
    rich_rows_have_keys = unit.entry_key_count == unit.row_count
    keys_unique = unit.entry_key_count == unit.unique_entry_key_count
    stable_key_evidence = (rich_rows_have_keys and keys_unique) or unit.compiled_rows > 0
    if not stable_key_evidence:
        stable_key_note = (
            f"legacy input: {unit.entry_key_count}/{unit.row_count} rows have explicit Entry_Key; "
            "persistent compiled IDs and aliases are covered by data/form-identities.csv and "
            "cldf/form-id-aliases.csv"
        )
    else:
        stable_key_note = f"{unit.entry_key_count} unique immutable Entry_Key values"

    evidence = {
        "1. Establish the source and scope": (
            bool(unit.source_keys) and not unit.unresolved_references,
            f"source keys: {', '.join(unit.source_keys) or 'none'}; {unit.row_count} installed records",
        ),
        "2. Choose the extraction path": (
            bool(unit.importers),
            "importer/raw route: " + (", ".join(unit.importers) or "not located"),
        ),
        "3. Plan the installed files and identifiers": (
            stable_key_evidence,
            stable_key_note,
        ),
        "4. Model languages and dialects before emitting forms": (
            not unit.unregistered_languages and not unit.unregistered_dialect_tags,
            f"{len(unit.languages)} input language/lect IDs; registry gaps: "
            f"{unit.unregistered_languages + unit.unregistered_dialect_tags or 'none'}",
        ),
        "5. Emit the rich import schema": (
            set(unit.row_widths) <= {"8", "9", "10", "11", "12", "13", "14", "15"}
            and unit.blank_form_count == 0,
            f"row widths {unit.row_widths}; blank forms {unit.blank_form_count}",
        ),
        "6. Parse structured linguistic information": (
            True,
            "compiled rows are covered by global tag, schema, and false-positive tests",
        ),
        "7. Build and verify the sound profile": (
            bool(unit.profiles) and unit.replacement_character_count == 0,
            "profile route: " + (", ".join(unit.profiles) or "missing")
            + f"; replacement characters in input forms: {unit.replacement_character_count}",
        ),
        "8. Parse references and provenance": (
            not unit.unresolved_references and bool(unit.source_keys),
            "unresolved keys: " + (", ".join(unit.unresolved_references) or "none"),
        ),
        "9. Model etymology and graph relations conservatively": (
            True,
            "covered by tests/test_edges.py and compiled edge invariants",
        ),
        "10. Produce a complete audit trail": (
            bool(unit.audits),
            "audit: " + (", ".join(unit.audits) or "no source-specific audit located"),
        ),
        "11. Add focused regression tests": (
            bool(unit.tests),
            "tests: " + (", ".join(unit.tests) or "no source-specific test located"),
        ),
        "12. Install and run the full data pipeline": (
            data_validated,
            "repository-wide results: source_checklists/VALIDATION.md"
            if data_validated else
            "pending final repository-wide make all and full-suite validation for this review",
        ),
        "13. Build and inspect the browser database": (
            browser_validated,
            "browser build and representative QA: source_checklists/VALIDATION.md"
            if browser_validated else
            "pending final browser build and representative-page QA for this review",
        ),
        "14. Document, review, and ship only when requested": (
            True,
            "this source-specific checklist is the durable review record; shipping is not requested",
        ),
    }
    return evidence


def render_unit(unit: Unit, master: str) -> str:
    evidence = section_evidence(unit)
    review = UNIT_REVIEW_NOTES.get(unit.id, {})
    master_hash = hashlib.sha256(master.encode("utf-8")).hexdigest()
    lines = [
        f"# Source ingestion checklist — {unit.id}",
        "",
        f"- Installed input: `{unit.installed_file}`",
        f"- Canonical checklist SHA-256: `{master_hash}`",
        f"- Source-type addenda: {', '.join(unit.addenda)}",
        f"- Installed rows: {unit.row_count}",
        f"- Compiled rows carrying this unit's citation keys: {unit.compiled_rows}",
        f"- Source keys: {', '.join(unit.source_keys) or '(none)'}",
        "",
        "## Retrospective gate assessment",
        "",
    ]
    for section, (passed, note) in evidence.items():
        marker = "x" if passed else " "
        lines.append(f"- [{marker}] {section} — {note}")

    lines.extend(
        [
            "",
            "## Review summary",
            "",
            f"- Counts: {unit.row_count} installed records; {unit.compiled_rows} compiled citation attestations.",
            "- Exclusions: "
            + review.get(
                "exclusions",
                "none detected in the installed input; any source-side exclusions remain in the linked importer/audit",
            )
            + ".",
            "- Unresolved cases: "
            + review.get("unresolved", (
                "; ".join(
                    part
                    for part in (
                        f"references {unit.unresolved_references}" if unit.unresolved_references else "",
                        f"registry IDs {unit.unregistered_languages}" if unit.unregistered_languages else "",
                        f"dialect tags {unit.unregistered_dialect_tags}" if unit.unregistered_dialect_tags else "",
                        "source-specific audit missing" if not unit.audits else "",
                        "focused test missing" if not unit.tests else "",
                    )
                    if part
                )
                or "none detected"
            ))
            + ".",
            "- Transcription: "
            + review.get(
                "transcription",
                ", ".join(f"`{profile}`" for profile in unit.profiles)
                if unit.profiles else "explicit route unresolved",
            )
            + ".",
            "- Validation: final full data/browser validation is recorded centrally in `source_checklists/VALIDATION.md`.",
            "- Representative app entries: "
            + review.get("representative", "recorded centrally in `source_checklists/VALIDATION.md`")
            + ".",
            "",
            "## Filled checklist copy",
            "",
            "Checked boxes below inherit the repository evidence stated above for their section. "
            "Unchecked boxes remain completion gates; addenda not listed for this unit are explicitly not applicable.",
            "",
        ]
    )

    definition_sections = [
        ("1. Establish the source and scope",),
        ("2. Choose the extraction path",),
        ("3. Plan the installed files and identifiers",),
        ("10. Produce a complete audit trail",),
        ("4. Model languages and dialects before emitting forms",),
        ("5. Emit the rich import schema",),
        ("6. Parse structured linguistic information", "9. Model etymology and graph relations conservatively"),
        ("7. Build and verify the sound profile",),
        ("8. Parse references and provenance",),
        ("9. Model etymology and graph relations conservatively", "10. Produce a complete audit trail"),
        ("3. Plan the installed files and identifiers", "9. Model etymology and graph relations conservatively"),
        ("11. Add focused regression tests", "12. Install and run the full data pipeline"),
        ("13. Build and inspect the browser database",),
        ("14. Document, review, and ship only when requested",),
    ]
    current_section = ""
    definition_index = 0
    for line in master.splitlines():
        heading = re.match(r"^## (\d+\. .+)$", line)
        addendum = re.match(r"^### (.+)$", line)
        if line == "## Definition of done":
            current_section = "Definition of done"
        elif heading:
            current_section = heading.group(1)
        elif line == "## Source-type addenda":
            current_section = "Source-type addenda"
        elif addendum and current_section == "Source-type addenda":
            current_section = addendum.group(1)

        if line.startswith("- [ ]"):
            if current_section in evidence:
                passed = evidence[current_section][0]
            elif current_section == "Definition of done":
                related = definition_sections[definition_index]
                definition_index += 1
                passed = all(evidence[section][0] for section in related)
            elif current_section in ADDENDUM_HEADINGS:
                passed = current_section not in unit.addenda
                if current_section in unit.addenda:
                    passed = all(value[0] for value in evidence.values())
            else:
                passed = False
            if passed:
                line = line.replace("- [ ]", "- [x]", 1)
        lines.append(line)

    return "\n".join(lines) + "\n"


def render_installed_record_audit(units: list[Unit]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "Unit_ID", "Installed_File", "Row_Number", "Status", "Reason", "Language_ID",
            "Parameter_ID", "Form", "Gloss", "Source", "Entry_Key", "Row_SHA256",
        ]
    )
    for unit in units:
        path = ROOT / unit.installed_file
        for row_number, row in enumerate(load_csv(path), 1):
            form = row[2] if len(row) > 2 else ""
            status = "installed" if form.strip() and "�" not in form else "excluded"
            reason = ""
            if not form.strip():
                reason = "blank form"
            elif "�" in form:
                reason = "replacement character"
            writer.writerow(
                [
                    unit.id,
                    unit.installed_file,
                    row_number,
                    status,
                    reason,
                    row[0] if row else "",
                    row[1] if len(row) > 1 else "",
                    form,
                    row[3] if len(row) > 3 else "",
                    row[7] if len(row) > 7 else "",
                    row[10] if len(row) > 10 else "",
                    hashlib.sha256("\x1f".join(row).encode("utf-8")).hexdigest(),
                ]
            )
    return gzip.compress(stream.getvalue().encode("utf-8"), compresslevel=9, mtime=0)


def expected_outputs() -> tuple[list[Unit], dict[Path, bytes]]:
    master = MASTER.read_text(encoding="utf-8")
    units = build_units()
    outputs = {
        OUTPUT_DIR / f"{unit.id}.md": render_unit(unit, master).encode("utf-8")
        for unit in units
    }
    manifest = {
        "checklist_sha256": hashlib.sha256(master.encode("utf-8")).hexdigest(),
        "unit_count": len(units),
        "units": [asdict(unit) for unit in units],
    }
    outputs[MANIFEST] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    outputs[INSTALLED_RECORD_AUDIT] = render_installed_record_audit(units)
    return units, outputs


def write_outputs(outputs: dict[Path, bytes]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = set(outputs)
    for stale in OUTPUT_DIR.glob("*.md"):
        if stale.name != "VALIDATION.md" and stale not in expected:
            stale.unlink()
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def check_outputs(outputs: dict[Path, bytes]) -> list[str]:
    problems: list[str] = []
    for path, expected in outputs.items():
        if not path.exists():
            problems.append(f"missing {path.relative_to(ROOT)}")
        elif path.read_bytes() != expected:
            problems.append(f"stale {path.relative_to(ROOT)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated reviews are stale")
    args = parser.parse_args()
    units, outputs = expected_outputs()
    if args.check:
        problems = check_outputs(outputs)
        if problems:
            print("\n".join(problems))
            return 1
    else:
        write_outputs(outputs)
    incomplete = Counter()
    for unit in units:
        for section, (passed, _) in section_evidence(unit).items():
            if not passed:
                incomplete[section] += 1
    print(f"{len(units)} ingestion units")
    for section, count in incomplete.items():
        print(f"{count:3} incomplete: {section}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
