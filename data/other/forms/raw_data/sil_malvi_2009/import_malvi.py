#!/usr/bin/env python3
"""Extract Appendix B of SIL ESR 2009-011 from its embedded SAG-IPA font.

The source PDF's ordinary text is searchable, but its main phonetic font is a
CID-keyed legacy SAG-IPA font without a ToUnicode map.  Extraction therefore
uses a checked CID-to-Unicode table, page geometry, and the separately embedded
WinAnsi SAG-IPA font used for ordinary ASCII and raised aspiration symbols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
WORKSPACE = DATA_ROOT.parent
PDF = WORKSPACE / "tmp/pdfs/malvi/silesr2009_011.pdf"
SNAPSHOT = HERE / "wordlist_snapshot.tsv"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-malvi.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-malvi-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-malvi-manifest.json"

SOURCE_KEY = "varghese-john-samuel2009malvi"
PDF_SHA256 = "e67e314974ab10eb8244b08dba56d08d1ce8cbf16eaef1be022071d49032a2dd"
PDF_URL = "http://www-01.sil.org/silesr/2009/silesr2009-011.pdf"
ARCHIVE_URL = (
    "https://web.archive.org/web/20150930085157id_/"
    "http://www-01.sil.org/silesr/2009/silesr2009-011.pdf"
)
OMITTED_PROMPTS = {11: "breast", 23: "urine", 24: "feces"}

TARGET_LECTS = [
    "Ujjaini-Malvi-Harsodan", "Ujjaini-Malvi-Chandukhedi",
    "Ujjaini-Malvi-Nain", "Ujjaini-Malvi-Koyal", "Ujjaini-Malvi-Rojdi",
    "Ujjaini-Malvi-Thillorkhurd", "Ujjaini-Malvi-Kumardi",
    "Ujjaini-Malvi-Jokhar", "Ujjaini-Malvi-Bercha",
    "Ujjaini-Malvi-Moondikhedi", "Rajwadi-Malvi-Lojithara",
    "Rajwadi-Malvi-Bhimakhedi", "Rajwadi-Malvi-Kishorpura",
    "Rajwadi-Malvi-Bhunyakhedi", "Rajwadi-Malvi-Jesingpura",
    "Rajwadi-Malvi-Bhandia", "Umadwadi-Malvi-Jhadmu",
    "Umadwadi-Malvi-Semlikakad", "Umadwadi-Malvi-Sagpur",
    "Umadwadi-Malvi-Paldyabana", "Umadwadi-Malvi-Mungavali",
    "Sondhwadi-Malvi-Harnauda", "Sondhwadi-Malvi-Narana",
    "Sondhwadi-Malvi-Adakhedi", "Sondhwadi-Malvi-Era",
    "Sondhwadi-Malvi-Deevdi", "Sondhwadi-Malvi-Jamli",
    "Gond-Malvi-Kalwar", "Gond-Malvi-Samapura", "Bhil-Malvi-Bhandikhali",
]
CONTROL_LECTS = [
    "Bhili-Kankaria", "Bhili-Athwaniya", "Nimadi-Jajumkhedi",
    "Nimadi-Sonipura", "Bhopali-Rpura", "Hindi-standard",
    "Gujarati-Standard", "Marathi-Standard",
]
LECTS = TARGET_LECTS + CONTROL_LECTS

LECT_METADATA = {
    "Ujjaini-Malvi-Harsodan": ("Harsodan-Ujjaini", "Harsodan, Ujjain tahsil and district, Madhya Pradesh; Ujjaini Malvi; source code 3"),
    "Ujjaini-Malvi-Chandukhedi": ("Chandukhedi-Ujjaini", "Chandukhedi, Ujjain tahsil and district, Madhya Pradesh; Ujjaini Malvi; source code C"),
    "Ujjaini-Malvi-Nain": ("Nain-Ujjaini", "Nain, Nagda tahsil, Ujjain district, Madhya Pradesh; Ujjaini Malvi; source code N"),
    "Ujjaini-Malvi-Koyal": ("Koyal-Ujjaini", "Koyal, Mahidpur tahsil, Ujjain district, Madhya Pradesh; Ujjaini Malvi; source code K"),
    "Ujjaini-Malvi-Rojdi": ("Rojdi-Ujjaini", "Rojdi, Indore tahsil and district, Madhya Pradesh; Ujjaini Malvi; source code R"),
    "Ujjaini-Malvi-Thillorkhurd": ("Thillorkhurd-Ujjaini", "Thillorkhurd, Indore tahsil and district, Madhya Pradesh; Ujjaini Malvi; wordlist reused from the Bhil-country report; source code T"),
    "Ujjaini-Malvi-Kumardi": ("Kumardi-Ujjaini", "Kumardi, Sonkatch tahsil, Dewas district, Madhya Pradesh; Ujjaini Malvi; source code I"),
    "Ujjaini-Malvi-Jokhar": ("Jokhar-Ujjaini", "Jhokher, Maksi tahsil, Shajapur district, Madhya Pradesh; Ujjaini Malvi; source code V"),
    "Ujjaini-Malvi-Bercha": ("Bercha-Ujjaini", "Bercha respondent resident in Dewas, Madhya Pradesh; Ujjaini Malvi; source code Y"),
    "Ujjaini-Malvi-Moondikhedi": ("Moondikhedi-Ujjaini", "Moondikhedi, Ashta tahsil, Sehore district, Madhya Pradesh; Ujjaini Malvi; source code 1"),
    "Rajwadi-Malvi-Lojithara": ("Lojithara-Rajwadi", "Logithara, Ratlam tahsil and district, Madhya Pradesh; Rajwadi Malvi; source code L"),
    "Rajwadi-Malvi-Bhimakhedi": ("Bhimakhedi-Rajwadi", "Bhimakhedi, Jaora tahsil, Ratlam district, Madhya Pradesh; Rajwadi Malvi; source code Q"),
    "Rajwadi-Malvi-Kishorpura": ("Kishorpura-Rajwadi", "Kishorepura, Sitamau tahsil, Mandsaur district, Madhya Pradesh; Rajwadi Malvi; source code O"),
    "Rajwadi-Malvi-Bhunyakhedi": ("Bhunyakhedi-Rajwadi", "Bhunyakhedi, Mandsaur tahsil and district, Madhya Pradesh; Rajwadi Malvi; source code 4"),
    "Rajwadi-Malvi-Jesingpura": ("Jesingpura-Rajwadi", "Jesingapura, Neemuch tahsil and district, Madhya Pradesh; Rajwadi Malvi; source code J"),
    "Rajwadi-Malvi-Bhandia": ("Bhandia-Rajwadi", "Bhandia, Manasa tahsil, Neemuch district, Madhya Pradesh; Rajwadi Malvi; source code B"),
    "Umadwadi-Malvi-Jhadmu": ("Jhadmu-Umadwadi", "Jhadmu, Zirapur tahsil, Rajgarh district, Madhya Pradesh; Umadwadi Malvi; source code X"),
    "Umadwadi-Malvi-Semlikakad": ("Semlikakad-Umadwadi", "Semlikakhad, Khilchipur tahsil, Rajgarh district, Madhya Pradesh; Umadwadi Malvi; source code Z"),
    "Umadwadi-Malvi-Sagpur": ("Sagpur-Umadwadi", "Sagpur, Narsinghgarh tahsil, Rajgarh district, Madhya Pradesh; Umadwadi Malvi; source code G"),
    "Umadwadi-Malvi-Paldyabana": ("Paldyabana-Umadwadi", "Paldyabana, Narsinghgarh tahsil, Rajgarh district, Madhya Pradesh; Umadwadi Malvi; source code 2"),
    "Umadwadi-Malvi-Mungavali": ("Mungavali-Umadwadi", "Mungawali, Sehore tahsil and district, Madhya Pradesh; Umadwadi Malvi; source code M"),
    "Sondhwadi-Malvi-Harnauda": ("Harnauda-Sondhwadi", "Harnauda, Gangdhar tahsil, Jhalawar district, Rajasthan; Sondhwadi Malvi; source code 5"),
    "Sondhwadi-Malvi-Narana": ("Narana-Sondhwadi", "Narana, Pirawa tahsil, Jhalawar district, Rajasthan; Sondhwadi Malvi; source code W"),
    "Sondhwadi-Malvi-Adakhedi": ("Adakhedi-Sondhwadi", "Adakhedi, Pirawa tahsil, Jhalawar district, Rajasthan; Sondhwadi Malvi; source code A"),
    "Sondhwadi-Malvi-Era": ("Era-Sondhwadi", "Era, Pachpahar tahsil, Jhalawar district, Rajasthan; Sondhwadi Malvi; source code E"),
    "Sondhwadi-Malvi-Deevdi": ("Deevdi-Sondhwadi", "Deevdi, Jhalrapatan tahsil, Jhalawar district, Rajasthan; Sondhwadi Malvi; source code D"),
    "Sondhwadi-Malvi-Jamli": ("Jamli-Sondhwadi", "Jamli, Agar tahsil, historical Shajapur district, Madhya Pradesh; Sondhwadi Malvi; source code U"),
    "Gond-Malvi-Kalwar": ("Kalwar-Gond", "Kalwar, Kannod tahsil, Dewas district, Madhya Pradesh; Malvi-speaking Gond respondent; source code P"),
    "Gond-Malvi-Samapura": ("Samapura-Gond", "Samapura, Ichhawar tahsil, Sehore district, Madhya Pradesh; Malvi-speaking Gond respondent; source code S"),
    "Bhil-Malvi-Bhandikhali": ("Bhandikhali-Bhil", "Bandikhali, Sardarpur tahsil, Dhar district, Madhya Pradesh; Bhil-Malvi respondent; source code F"),
}

LABEL_ALIASES = {
    "Ujjain-Malvi-Thillorkhurd": "Ujjaini-Malvi-Thillorkhurd",
    "Rajwadi-Malvi-Bandia": "Rajwadi-Malvi-Bhandia",
    "Bhpali-Rpura": "Bhopali-Rpura",
    "Hindi-sotandard": "Hindi-standard",
}

# CIDs used on physical pages 59--177.  All but the last three are printed in
# the source's IPA chart on p.45.  CID 45 is byte 0x4B in SIL's SAGIPA2Uni map;
# CID 64 is byte 0x5E.  CID 197 follows CID 196 (combining bridge below) and is
# byte 0xE7, combining square below.  Each was also checked in rendered pages.
CID_MAP = {
    3: "ː", 30: "ɸ", 35: "ɑ", 36: "β", 37: "ç",
    39: "ɛ", 43: "ɪ", 45: "ɠ", 52: "ɾ", 53: "ʃ",
    55: "ʊ", 56: "ʋ", 60: "ʒ", 64: "^", 73: "g",
    95: "ɽ", 98: "ɨ", 101: "ə", 102: "ɜ", 103: "ɐ",
    106: "ʌ", 107: "ɔ", 118: "ɳ", 119: "ʂ", 121: "ʈ",
    122: "ɖ", 124: "ɭ", 126: "ɲ", 132: "ŋ", 137: "ɦ",
    139: "ɕ", 174: "\N{COMBINING TILDE}",
    196: "\N{COMBINING BRIDGE BELOW}",
    197: "\N{COMBINING SQUARE BELOW}",
}

SNAPSHOT_FIELDS = [
    "PDF_Page", "Printed_Page", "Concept", "Gloss", "Lect", "Category",
    "Response_Index", "Form", "Raw_Fragments", "Notes", "Source_Status",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Concept",
    "Gloss", "Lect", "Scope", "Category", "Response_Index", "Raw_Form",
    "Form", "Notes", "Status", "Reason", "Language_ID", "Dialect_ID",
    "Citation", "Entry_Key",
]

CID_RE = re.compile(r"^\(cid:(\d+)\)$")
CATEGORY_RE = re.compile(r"^(?:\d+(?:,\d+)*,?|[a-z])$")
HEADING_RE = re.compile(r"^(\d{1,3})\.$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def dialect_id(lect: str) -> str:
    return f"sil-malvi-2009-{slug(LECT_METADATA[lect][0])}"


def normalize(value: str) -> str:
    value = re.sub(r"\s+([,.;:\u0300-\u036f])", r"\1", value.strip())
    return unicodedata.normalize("NFC", value)


def decode_char(char: dict[str, object]) -> str:
    text = str(char["text"])
    match = CID_RE.fullmatch(text)
    if match:
        cid = int(match.group(1))
        if cid not in CID_MAP:
            raise ValueError(f"Unmapped SAG-IPA CID {cid}")
        text = CID_MAP[cid]
    if float(char.get("size", 99)) < 7:
        modifiers = {
            "h": "ʰ", "ɦ": "ʱ", "i": "ⁱ", "u": "ᵘ",
            "ɪ": "ᶦ", "ə": "ᵊ", "j": "ʲ", "w": "ʷ",
        }
        text = modifiers.get(text, text)
    return text


def reconstruct_chars(chars: list[tuple[int, dict[str, object]]]) -> tuple[str, str]:
    ordered = sorted(chars, key=lambda item: (round(float(item[1]["x0"]), 3), item[0]))
    output: list[str] = []
    raw: list[str] = []
    previous_x1: float | None = None
    for _, char in ordered:
        x0, x1 = float(char["x0"]), float(char["x1"])
        text = str(char["text"])
        decoded = decode_char(char)
        # The PDF positions real inter-word gaps at roughly 3.4pt; glyph and
        # diacritic overlaps can be negative, so only positive gaps add space.
        if previous_x1 is not None and x0 - previous_x1 > 2.1:
            output.append(" ")
            raw.append(" ")
        output.append(decoded)
        raw.append(text)
        previous_x1 = max(previous_x1 or x1, x1)
    return normalize("".join(output)), "".join(raw).strip()


def heading_for_column(
    words: list[dict[str, object]], x0: float, x1: float,
    fallback: tuple[int, str] | None = None,
) -> tuple[int, str]:
    candidates = [
        word for word in words
        if x0 <= float(word["x0"]) < x1 and HEADING_RE.fullmatch(str(word["text"]))
        and 60 <= float(word["top"]) <= 190
    ]
    if not candidates and fallback is not None:
        return fallback
    if len(candidates) != 1:
        raise ValueError(f"Expected one heading in x={x0}:{x1}, found {candidates}")
    heading = candidates[0]
    top = float(heading["top"])
    gloss_words = [
        word for word in words
        if float(word["x0"]) > float(heading["x1"])
        and float(word["x0"]) < x1 and abs(float(word["top"]) - top) < 1.2
    ]
    gloss = " ".join(str(word["text"]) for word in sorted(gloss_words, key=lambda w: float(w["x0"])))
    return int(str(heading["text"])[:-1]), gloss.strip()


def parse_column(
    page, physical_page: int, bounds: tuple[float, float, float, float],
    fallback_heading: tuple[int, str] | None = None,
) -> list[dict[str, object]]:
    x0, x1, category_x, category_x1 = bounds
    words = page.extract_words(x_tolerance=2, y_tolerance=2, extra_attrs=["fontname"])
    concept, gloss = heading_for_column(words, x0, x1, fallback_heading)
    categories = [
        word for word in words
        if category_x <= float(word["x0"]) < category_x1
        and CATEGORY_RE.fullmatch(str(word["text"]))
        and "Times" in str(word.get("fontname", ""))
        and 80 <= float(word["top"]) <= 740
    ]
    indexed_chars = list(enumerate(page.chars))
    records: list[dict[str, object]] = []
    current_lect: str | None = None
    for category in sorted(categories, key=lambda word: float(word["top"])):
        row_top = float(category["top"])
        label_words = [
            word for word in words
            if x0 <= float(word["x0"]) < category_x
            and abs(float(word["top"]) - row_top) < 1.2
        ]
        raw_label = " ".join(str(word["text"]) for word in sorted(label_words, key=lambda w: float(w["x0"]))).strip()
        if raw_label:
            normalized_label = raw_label.replace("-Hindi-", "-Malvi-")
            lect = LABEL_ALIASES.get(normalized_label, normalized_label)
            if lect not in LECTS:
                raise ValueError(
                    f"Unknown lect label on PDF page {physical_page}, concept {concept}: {raw_label!r}"
                )
            current_lect = lect
        elif current_lect is None:
            raise ValueError(f"Orphan alternate on PDF page {physical_page}, concept {concept}")

        form_x = float(category["x0"]) + 11.5
        form_chars = [
            (index, char) for index, char in indexed_chars
            if form_x <= float(char["x0"]) < x1
            and row_top - 1.3 <= float(char["top"]) <= row_top + 6.5
            and "SAG-IPA" in str(char.get("fontname", ""))
        ]
        form, raw = reconstruct_chars(form_chars)
        if not form:
            normal_form_words = [
                word for word in words
                if float(category["x0"]) + 10.0 <= float(word["x0"]) < x1
                and abs(float(word["top"]) - row_top) < 1.2
            ]
            normal_text = " ".join(
                str(word["text"]) for word in sorted(normal_form_words, key=lambda w: float(w["x0"]))
            ).strip()
            if normal_text.casefold() not in {"no entry", "by name"}:
                raise ValueError(
                    f"No response on PDF page {physical_page}, concept {concept}, "
                    f"lect {current_lect}; normal text={normal_text!r}"
                )
            raw = normal_text
            source_status = normal_text.casefold().replace(" ", "_")
        else:
            source_status = "response"
        records.append({
            "PDF_Page": physical_page,
            "Printed_Page": physical_page,
            "Concept": concept,
            "Gloss": gloss,
            "Lect": current_lect,
            "Category": str(category["text"]),
            "Form": form,
            "Raw_Fragments": raw,
            "Notes": f"source label {raw_label!r}" if raw_label and raw_label != current_lect else "",
            "Source_Status": source_status,
        })
    return records


def extract_pdf(path: Path) -> list[dict[str, str]]:
    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover
        raise SystemExit("--extract requires pdfplumber") from error
    if sha256(path) != PDF_SHA256:
        raise ValueError(f"Unexpected PDF SHA-256 for {path}")

    records: list[dict[str, object]] = []
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 280:
            raise ValueError(f"Expected 280 PDF pages, got {len(pdf.pages)}")
        for physical_page in range(59, 178):
            page = pdf.pages[physical_page - 1]
            if (round(page.width, 1), round(page.height, 1)) != (612.0, 792.0):
                raise ValueError(f"Unexpected page geometry on physical page {physical_page}")
            columns = (
                [(80.0, 306.0, 190.0, 217.0), (322.0, 612.0, 440.0, 458.0)]
                if physical_page <= 148 else
                [(80.0, 612.0, 190.0, 240.0)]
            )
            page_heading: tuple[int, str] | None = None
            for bounds in columns:
                column_records = parse_column(page, physical_page, bounds, page_heading)
                if column_records and page_heading is None:
                    page_heading = (
                        int(column_records[0]["Concept"]), str(column_records[0]["Gloss"])
                    )
                records.extend(column_records)

    concepts = {int(record["Concept"]): str(record["Gloss"]) for record in records}
    expected = set(range(1, 211)) - set(OMITTED_PROMPTS)
    if set(concepts) != expected:
        raise ValueError(f"Unexpected concept inventory: missing={sorted(expected-set(concepts))}, extra={sorted(set(concepts)-expected)}")

    per_pair: Counter[tuple[int, str]] = Counter(
        (int(record["Concept"]), str(record["Lect"])) for record in records
    )
    missing_pairs = sorted(
        (concept, lect) for concept in expected for lect in LECTS
        if not per_pair[(concept, lect)]
    )
    if missing_pairs:
        raise ValueError(f"Missing concept/lect responses: {missing_pairs[:20]} ({len(missing_pairs)} total)")

    for concept, gloss in OMITTED_PROMPTS.items():
        for lect in LECTS:
            records.append({
                "PDF_Page": "", "Printed_Page": "", "Concept": concept,
                "Gloss": gloss, "Lect": lect, "Category": "", "Form": "",
                "Raw_Fragments": "", "Notes": "prompt disqualified and absent from Appendix B",
                "Source_Status": "omitted_prompt",
            })

    indexed_records = list(enumerate(records))
    indexed_records.sort(key=lambda item: (
        int(item[1]["Concept"]), LECTS.index(str(item[1]["Lect"])), item[0]
    ))
    indices: Counter[tuple[int, str]] = Counter()
    output: list[dict[str, str]] = []
    for _, record in indexed_records:
        key = (int(record["Concept"]), str(record["Lect"]))
        indices[key] += 1
        record["Response_Index"] = indices[key]
        output.append({field: str(record.get(field, "")) for field in SNAPSHOT_FIELDS})
    if len(output) != 8_912:
        raise ValueError(f"Expected 8,912 audited cells/responses, got {len(output)}")
    return output


def write_snapshot(records: list[dict[str, str]]) -> None:
    with SNAPSHOT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SNAPSHOT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def load_snapshot() -> list[dict[str, str]]:
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 8_912:
        raise ValueError(f"Expected 8,912 snapshot rows, got {len(rows)}")
    return rows


def install(records: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in records:
        lect = row["Lect"]
        target = lect in TARGET_LECTS
        source_status = row["Source_Status"]
        status, reason = "installed", ""
        if not target:
            status, reason = "excluded", "borrowed or standard comparison list"
        elif source_status == "by_name":
            status, reason = "excluded", "source records a by-name response, not a lexical form"
        elif source_status == "omitted_prompt":
            status, reason = "excluded", "prompt disqualified and absent from the published appendix"
        elif source_status == "no_entry":
            status, reason = "excluded", "source explicitly says no entry"

        concept = int(row["Concept"])
        response_index = int(row["Response_Index"])
        site_id = dialect_id(lect) if target else ""
        page = f"printed p. {row['Printed_Page']}, " if row["Printed_Page"] else ""
        citation = f"{SOURCE_KEY}[Appendix B, {page}item {concept}, {lect}]"
        entry_key = (
            f"silmalvi2009:g{concept:03d}:{site_id}:i{response_index}"
            if status == "installed" else ""
        )
        notes = "; ".join(part for part in (
            f"Appendix B lexical-similarity category {row['Category']}" if row["Category"] else "",
            row["Notes"],
        ) if part)
        if status == "installed":
            display = LECT_METADATA[lect][0]
            tag = f"dialect:mewari_basad:{site_id}:{quote(display)}"
            forms.append([
                "mewari_basad", "", row["Form"], row["Gloss"], "", row["Form"], notes,
                citation, "", "", entry_key, "", "", "", tag,
            ])
        audit.append(dict(zip(AUDIT_FIELDS, [
            "wordlist response" if source_status == "response" else "wordlist matrix cell",
            SOURCE_KEY, row["PDF_Page"], row["Printed_Page"], row["Concept"],
            row["Gloss"], lect, "Malvi target list" if target else "comparison list",
            row["Category"], row["Response_Index"], row["Raw_Fragments"], row["Form"],
            row["Notes"], status, reason, "mewari_basad" if target else "", site_id,
            citation, entry_key,
        ])))
    return forms, audit


def write_outputs(records: list[dict[str, str]]) -> None:
    forms, audit = install(records)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)

    excluded = Counter(row["Reason"] for row in audit if row["Status"] == "excluded")
    manifest = {
        "source": SOURCE_KEY,
        "title": "The Malvi-speaking people of Madhya Pradesh and Rajasthan: A sociolinguistic profile",
        "source_pdf_url": PDF_URL,
        "source_pdf_archive_url": ARCHIVE_URL,
        "source_pdf_sha256": PDF_SHA256,
        "source_pdf_pages": 280,
        "appendix": "Appendix B, source metadata on printed pp. 49-58 and comparative wordlists on printed pp. 59-177",
        "extraction": "Geometric parser over the embedded legacy SAG-IPA CID font and companion WinAnsi font; checked CID-to-Unicode table; no OCR",
        "snapshot_sha256": sha256(SNAPSHOT),
        "counts": {
            "standard_prompts": 210,
            "printed_prompts": 207,
            "target_lists": 30,
            "comparison_lists": 8,
            "snapshot_and_audit_records": len(audit),
            "printed_response_records": sum(row["Source_Status"] != "omitted_prompt" for row in records),
            "additional_response_lines": sum(int(row["Response_Index"]) > 1 for row in records),
            "synthetic_omitted_prompt_cells": sum(row["Source_Status"] == "omitted_prompt" for row in records),
            "installed_malvi_forms": len(forms),
            "excluded_records": sum(row["Status"] == "excluded" for row in audit),
            "explicit_no_entry_records": sum(row["Source_Status"] == "no_entry" for row in records),
            "by_name_records": sum(row["Source_Status"] == "by_name" for row in records),
            "comparison_records": sum(row["Lect"] in CONTROL_LECTS for row in records),
        },
        "lect_counts": dict(sorted(Counter(
            row[14].split(":", 3)[3] for row in forms
        ).items())),
        "font_recovery": {
            "used_cids": len(CID_MAP),
            "validation": "31 used CIDs identified from the source IPA chart; three additional used CIDs identified from SIL's SAGIPA2Uni map and rendered pages",
            "cross_source_check": "the Thillorkhurd list was compared with the later Unicode Malvi comparison list in SIL ESR 2012-002; 132 response forms agree exactly within concept, validating 126 concepts and nearly the full used symbol inventory",
        },
        "editorial_policy": {
            "scope": "thirty Malvi target lists installed; two Bhili, two Nimadi, Bhopali, Hindi, Gujarati and Marathi controls retained only in audit",
            "categories": "source lexical-similarity category numbers and letters retained in Notes, never interpreted as etymologies",
            "missing_prompts": "items 11, 23 and 24 are described as disqualified, absent from Appendix B, and represented as explicit audit-only matrix cells",
            "nonlexical_responses": "printed 'By Name' cells are retained in the audit and excluded from installed lexical forms",
            "transcription": "legacy SAG-IPA glyphs converted diplomatically to Unicode, including raised modifier letters and combining diacritics; source-interchangeable central-vowel symbols remain distinct in Phonemic and are coalesced only by the display profile",
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"installed={len(forms)} comparisons={excluded['borrowed or standard comparison list']} "
        f"by_name={excluded['source records a by-name response, not a lexical form']} "
        f"omitted={excluded['prompt disqualified and absent from the published appendix']} "
        f"audit={len(audit)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="refresh the TSV from the pinned PDF")
    args = parser.parse_args()
    if args.extract:
        records = extract_pdf(PDF)
        write_snapshot(records)
        print(f"wrote {len(records)} rows to {SNAPSHOT}")
    write_outputs(load_snapshot())


if __name__ == "__main__":
    main()
