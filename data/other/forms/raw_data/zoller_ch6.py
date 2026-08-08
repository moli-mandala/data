#!/usr/bin/env python3
"""Extract Zoller's OIA -> Indus Kohistani index and link it to CDIAL.

The chapter PDF uses an old Type 3 phonetic font without a Unicode map.  Its
printable character codes are not their visible glyphs (for example, byte 57,
which extractors report as ``9``, draws ``z``).  This script decodes that font
directly, reconstructs the two index columns, resolves the OIA heads against
the local CDIAL data, and writes the repository's eight-column manual-import
CSV.  The neighbouring Bhatise/Batera (B) and Gabar/Gowro (G) forms are kept
as Bhateri and Gowro rows alongside the three Indus Kohistani varieties.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pdfplumber


# Character-code -> visible glyph for Zoller's embedded Type 3 font.
TYPE3 = {
    0: "\u0300", 1: "\u0301", 2: "\u0303", 3: "\u0308", 4: "\u030a",
    5: "\u030c", 6: "\u0306", 7: "\u0304", 8: "\u0307", 9: "\u0302",
    10: "ı", 11: "ȷ", 12: "*", 13: ",", 14: "-", 15: "ʌ", 16: "ʒ",
    17: ":", 18: "\u0323", 19: "=", 20: "ə", 21: "ɑ", 22: "ɕ",
    23: "ð", 24: "ɛ", 25: "ɣ", 26: "ŋ", 27: "ɔ", 28: "ʔ", 29: "θ",
    30: "ʋ", 31: "ɯ", 32: "'", 33: "a", 34: "b", 35: "c", 36: "d",
    37: "e", 38: "f", 39: "g", 40: "h", 41: "i", 42: "j", 43: "k",
    44: "l", 45: "m", 46: "n", 47: "o", 48: "p", 49: "q", 50: "r",
    51: "s", 52: "t", 53: "u", 54: "v", 55: "x", 56: "y", 57: "z",
    58: "æ",
}

DIALECTS = {"J", "Š", "S", "B", "G"}
LANGUAGE_BY_DIALECT = {
    "J": "Mai-Jijal",  # Jijālī Indus Kohistani
    "Š": "Mai-Shatot",  # Šāṭōṭī Indus Kohistani
    "S": "Mai-Seo",  # Seo Indus Kohistani
    "B": "bhatr",  # Bhaṭīse/Baṭerā (Bhateri)
    "G": "Gowro",  # Gabār/Gowro
}


def decode_char(char: dict) -> str:
    text = char["text"]
    if char["fontname"] != "unknown":
        return text
    match = re.fullmatch(r"\(cid:(\d+)\)", text)
    code = int(match.group(1)) if match else ord(text) if len(text) == 1 else None
    return TYPE3.get(code, text)


def canonicalize(text: str) -> str:
    """Turn the PDF's TeX-era transliteration into NFC Unicode."""
    for old, new in {
        "´": "\u0301", "`": "\u0300", "¯": "\u0304", "˜": "\u0303",
        "¨": "\u0308", "˙": "\u0307", "ˇ": "\u030c",
    }.items():
        text = text.replace(old, new)

    # Dot-below letters are encoded either as .t or t. in the legacy layer.
    text = re.sub(r"([tdnrscz])\.", lambda m: m.group(1) + "\u0323", text)
    text = re.sub(r"\.([tdnrs])", lambda m: m.group(1) + "\u0323", text)
    text = re.sub(r"([ṭḍṇṛṣ])\.", r"\1", text)
    text = text.replace("ı", "i")
    text = re.sub(r"\s+", " ", text).strip()
    decomposed = unicodedata.normalize("NFD", text)
    # In strings such as TeX ``.t.t`` the middle dot is simultaneously after
    # the first t and before the second. If geometry attached both dots to the
    # first base, distribute the second one to the following base.
    decomposed = re.sub(
        r"([tdnrs])\u0323\u0323([tdnrs])",
        lambda m: m.group(1) + "\u0323" + m.group(2) + "\u0323",
        decomposed,
    )
    decomposed = re.sub(r"\u0323{2,}", "\u0323", decomposed)
    return unicodedata.normalize("NFC", decomposed)


def rebuild(chars: list[dict]) -> str:
    """Deduplicate CID artifacts, place accents geometrically, restore spaces."""
    unique: list[tuple[int, dict, str]] = []
    previous = None
    for index, char in enumerate(chars):
        identity = (
            char["text"], round(char["x0"], 3), round(char["top"], 3),
            round(char["x1"], 3), round(char["bottom"], 3),
        )
        if identity == previous:
            continue
        previous = identity
        unique.append((index, char, decode_char(char)))

    spacing_marks = {
        "´": "\u0301", "`": "\u0300", "¯": "\u0304", "˜": "\u0303",
        "¨": "\u0308", "˙": "\u0307", "ˇ": "\u030c",
    }
    bases = []
    marks = []
    for index, char, text in unique:
        mark = spacing_marks.get(text, text if len(text) == 1 and unicodedata.combining(text) else "")
        if mark:
            marks.append((char, mark))
        else:
            bases.append({"index": index, "char": char, "text": text, "marks": []})

    if not bases:
        return ""

    # TeX draws accents as separate glyphs. Attach each to the horizontally
    # nearest base glyph; content-stream order alone puts some marks on the
    # preceding letter (e.g. extracted ``ʌ~y/`` visibly reads ``ʌỹ``).
    for mark_char, mark in marks:
        mark_center = (mark_char["x0"] + mark_char["x1"]) / 2
        target = min(
            bases,
            key=lambda base: abs(
                (base["char"]["x0"] + base["char"]["x1"]) / 2 - mark_center
            ),
        )
        target["marks"].append(mark)

    output: list[str] = []
    rightmost = None
    for base in sorted(bases, key=lambda item: (item["char"]["x0"], item["index"])):
        char = base["char"]
        if rightmost is not None and char["x0"] - rightmost > 1.6:
            output.append(" ")
        output.append(base["text"] + "".join(base["marks"]))
        rightmost = max(rightmost if rightmost is not None else char["x1"], char["x1"])
    return canonicalize("".join(output))


def extract_mappings(pdf: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    with pdfplumber.open(pdf) as document:
        for page_number, page in enumerate(document.pages, 1):
            columns: list[list[tuple[float, float, str]]] = [[], []]
            # The media box is offset; x=300 is the stable absolute gutter.
            for line in page.extract_text_lines(layout=False, return_chars=True):
                for column in (0, 1):
                    chars = [
                        char for char in line["chars"]
                        if (130 <= char["x0"] < 300) if column == 0
                    ] if column == 0 else [
                        char for char in line["chars"] if 300 <= char["x0"] < 530
                    ]
                    text = rebuild(chars)
                    if text:
                        columns[column].append(
                            (line["top"], min(char["x0"] for char in chars), text)
                        )

            for column, lines in enumerate(columns):
                # A raised homonym number can make pdfplumber divide a single
                # printed mapping into three lines (source, superscript, RHS).
                # Their vertical span is much smaller than the interline gap;
                # cluster them and restore their left-to-right order.
                clustered: list[list[tuple[float, float, str]]] = []
                for item in sorted(lines):
                    if clustered and item[0] - min(part[0] for part in clustered[-1]) <= 6:
                        clustered[-1].append(item)
                    else:
                        clustered.append([item])
                rebuilt_lines = [
                    (min(part[0] for part in group), " ".join(
                        part[2] for part in sorted(group, key=lambda part: part[1])
                    ))
                    for group in clustered
                ]
                previous = None
                for _, text in rebuilt_lines:
                    if ">" in text:
                        previous = {
                            "pdf_page": page_number,
                            "printed_page": 476 + page_number,
                            "column": column,
                            "text": text,
                        }
                        records.append(previous)
                    elif previous and (
                        re.fullmatch(r"\(?[JBGŠS](?:, ?[JBGŠS])*\)?", text)
                        or text.startswith((",", ";"))
                    ):
                        previous["text"] = f"{previous['text']} {text}"
    return records


def match_key(text: str, *, loose: bool = False) -> str:
    text = canonicalize(text).lower()
    decomposed = unicodedata.normalize("NFD", text)
    # The strict key preserves both segmental marks and Vedic accent, which
    # disambiguates a number of CDIAL homonyms. The loose fallback removes all
    # marks for entries whose index spelling differs slightly from the head.
    ignored: set[str] = set()
    if loose:
        ignored = {chr(code) for code in range(0x300, 0x370)}
    text = "".join(char for char in decomposed if char not in ignored)
    text = unicodedata.normalize("NFC", text).replace("ʌ", "a").replace("ɑ", "a")
    return "".join(char for char in text if char.isalpha())


def load_cdial(repo: Path):
    redirects: dict[str, str] = {}
    merge_path = repo / "cldf/merges.csv"
    if merge_path.exists():
        with merge_path.open(encoding="utf-8") as stream:
            redirects = {row["Addendum_ID"]: row["Main_ID"] for row in csv.DictReader(stream)}

    strict: dict[str, list[str]] = defaultdict(list)
    loose: dict[str, list[str]] = defaultdict(list)
    glosses: dict[str, str] = {}
    head_forms: dict[str, list[str]] = defaultdict(list)
    homonyms: dict[str, str] = {}

    with (repo / "data/cdial/params.csv").open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            entry_id, forms, description = row[0], row[1], row[3]
            entry_id = redirects.get(entry_id, entry_id)
            for form in re.split(r"[,;/]", forms):
                strict[match_key(form)].append(entry_id)
                loose[match_key(form, loose=True)].append(entry_id)
                head_forms[entry_id].append(form)
            number = re.search(r"</b>\s*([¹²³⁴⁵⁶⁷⁸⁹])", description)
            if number:
                homonyms[entry_id] = str("¹²³⁴⁵⁶⁷⁸⁹".index(number.group(1)) + 1)

    # Unified forms contains promoted numbered subheads (ID-n) and internal
    # OIA variants. Add direct etyma first. Variant spellings are a fallback
    # and point to their CDIAL Origin_ID, never to generated ``0-N`` row IDs.
    variant_rows = []
    with (repo / "cldf/forms.csv").open(encoding="utf-8") as stream:
        _form_rows = list(csv.DictReader(stream))
    import sys as _sys
    _sys.path.insert(0, str(repo))
    from edges_util import attach_legacy_graph
    attach_legacy_graph(_form_rows, str(repo / "cldf/edges.csv"))
    if True:
        for row in _form_rows:
            if row["Language_ID"] != "Indo-Aryan" or "CDIAL" not in row["Source"]:
                continue
            direct = not row["Origin_ID"]
            entry_id = row["ID"] if direct else row["Origin_ID"]
            entry_id = redirects.get(entry_id, row["Redirect"] or entry_id)
            glosses[entry_id] = row["Gloss"]
            original = row["Original"] or row["Form"]
            if direct:
                for form in re.split(r"[,;/]", original):
                    strict[match_key(form)].append(entry_id)
                    loose[match_key(form, loose=True)].append(entry_id)
            else:
                variant_rows.append((entry_id, original))

    # Only use a variant spelling when no CDIAL head/subhead has that key.
    for entry_id, original in variant_rows:
        for form in re.split(r"[,;/]", original):
            strict_key = match_key(form)
            loose_key = match_key(form, loose=True)
            if strict_key not in strict:
                strict[strict_key].append(entry_id)
            if loose_key not in loose:
                loose[loose_key].append(entry_id)

    def dedupe(mapping):
        return {key: list(dict.fromkeys(values)) for key, values in mapping.items() if key}

    return dedupe(strict), dedupe(loose), glosses, head_forms, homonyms


def parse_oia(text: str) -> tuple[str, str | None]:
    # Compounds are linked to their first stated CDIAL element.  A parenthetic
    # "see X" explicitly redirects an author reconstruction to Turner's X.
    text = re.split(r"\bplus\b", text, 1)[0]
    see = re.search(r"\bsee\s+([^\)]+)", text)
    if see:
        text = see.group(1)
    section = re.search(r"-?(\d+)\s*$", text)
    text = re.sub(r"-?\d*\s*$", "", text).strip(" *†̊")
    return text, section.group(1) if section else None


def choose_cdial(
    oia: str,
    section: str | None,
    strict: dict[str, list[str]],
    loose: dict[str, list[str]],
    head_forms: dict[str, list[str]],
    homonyms: dict[str, str],
) -> tuple[str, str, list[str]]:
    candidates = strict.get(match_key(oia), [])
    status = "exact"
    if not candidates:
        candidates = loose.get(match_key(oia, loose=True), [])
        status = "loose"

    if section:
        numbered_heads = [item for item in candidates if homonyms.get(item) == section]
        section_ids = [item for item in candidates if item.endswith(f"-{section}")]
        if numbered_heads:
            candidates = numbered_heads
        elif section_ids:
            candidates = section_ids
        elif len(candidates) >= int(section):
            # Turner's superscript homonym number is printed as ``-N`` in the
            # index. When several CDIAL heads have the same normalized form,
            # select the Nth head in dictionary order.
            candidates = [candidates[int(section) - 1]]
        else:
            base_ids = [item for item in candidates if "-" not in item]
            candidates = base_ids or candidates

    # Source-specific resolutions checked against Turner's numbered heads and
    # meanings. These cover homographs that spelling alone cannot separate.
    source_exact = canonicalize(oia).lower().strip(" *-")
    manual = {
        "aho": "996",
        "anda": "1111",       # āṇḍá 'egg', not *anda 'binding'
        "kunda": "3265",      # kuṇḍa²
        "khadda": "3790",     # *khaḍḍa 'hole, pit'
        "gadda": "3982",      # *gaḍḍa² 'bundle, sheaf'
        "gudda": "4189",      # *guḍḍa 'doll' (cf. guṛī)
        "tapti": "5683",
        "tarayati": "5796",
        "na": "6906",
        "pastana": "8014",
        "pilla": "8214",      # 'small/young', not pilla² 'blear-eyed'
        "peda": "8377a",      # 'tree', reflected by B peṛ
        "manda": "9754-5",    # manda 'dull, slow, weak'
        "luda": "11076-3",
        "lunda": "11076-7",
        "lettha": "11054-6",
        "vasati": "11435",
        "sammukha": "12982",
        "saras": "13254",     # sáras 'lake', not śáras 'cream'
    }.get(match_key(oia, loose=True))
    if source_exact.startswith("ṣaṇḍh"):
        manual = "12270-5"      # ṣaṇḍhá 'eunuch'
    elif source_exact.startswith(("sāṇḍ", "sɑ̄́ṇḍ", "sɑ̄́nḍ")):
        manual = "13331"        # sā́ṇḍa 'uncastrated bull'
    if manual:
        return manual, "manual", candidates

    # Identical promoted spellings can occur in two numbered sections of the
    # same CDIAL entry. The chapter does not always distinguish them; linking
    # to their common base is accurate and stable for ingestion.
    roots = {candidate.split("-", 1)[0] for candidate in candidates}
    if len(candidates) > 1 and len(roots) == 1:
        return roots.pop(), "common-base", candidates

    if len(candidates) > 1:
        # Some index spellings omit a macron or use nḍ for CDIAL ṇḍ. Rank the
        # still-ambiguous CDIAL heads by Unicode-aware edit similarity, while
        # requiring a clear margin to avoid inventing a match.
        source = canonicalize(oia).strip(" *-").lower()
        ranked = []
        for candidate in candidates:
            score = max(
                difflib.SequenceMatcher(
                    None, source, canonicalize(form).strip(" *-").lower()
                ).ratio()
                for form in head_forms.get(candidate, [candidate])
            )
            ranked.append((score, candidate))
        ranked.sort(reverse=True)
        if ranked[0][0] >= 0.72 and (
            len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.06
        ):
            return ranked[0][1], "disambiguated", candidates

    # Prefer a unique base entry over internal/promoted duplicates.
    base_ids = [item for item in candidates if "-" not in item]
    if len(base_ids) == 1:
        return base_ids[0], status, candidates
    if len(candidates) == 1:
        return candidates[0], status, candidates
    return "", "ambiguous" if candidates else "unmatched", candidates


def top_level_split(text: str, separators=(";", ",")) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char in separators and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def rhs_forms(rhs: str) -> list[tuple[str, set[str]]]:
    output: list[tuple[str, set[str]]] = []
    for segment in top_level_split(rhs):
        dialect_match = re.search(r"\(([JBGŠS](?:, ?[JBGŠS])*)\)\s*$", segment)
        dialects = (
            {item.strip() for item in dialect_match.group(1).split(",")}
            if dialect_match else {"J"}
        )
        if dialect_match:
            segment = segment[:dialect_match.start()].strip()
        # The index's "in X" is metalinguistic context, not part of X.
        segment = re.split(r"\.\s+(?:See|Cf\.)\b", segment, 1)[0]
        if re.match(r"^(?:connected with|same as)\b", segment, re.IGNORECASE):
            continue
        variants = re.split(r"\s+and\s+in\s+|\s+(?:or|and)\s+", segment)
        for form in variants:
            form = re.sub(r"^in\s+", "", form.strip())
            form = re.sub(r"\s*\(see(?: also)?[^)]*\)\s*", "", form)
            form = re.sub(r"(?<=\D)\d+(?!\d)", "", form)  # homonym subscripts
            form = form.strip(" ;,")
            if form:
                output.append((form, dialects))
    return output


def run(pdf: Path, repo: Path, output: Path, audit: Path) -> None:
    strict, loose, glosses, head_forms, homonyms = load_cdial(repo)
    rows: list[list[str]] = []
    audit_rows: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str]] = set()

    for record in extract_mappings(pdf):
        raw_oia, rhs = re.split(r"\s*-?\s*>\s*", str(record["text"]), 1)
        oia, section = parse_oia(raw_oia)
        entry_id, status, candidates = choose_cdial(
            oia, section, strict, loose, head_forms, homonyms
        )
        forms = rhs_forms(rhs)

        audit_rows.append({
            "PDF_Page": record["pdf_page"],
            "Printed_Page": record["printed_page"],
            "OIA_Source": raw_oia,
            "OIA_Match_Form": oia,
            "CDIAL_ID": entry_id,
            "Match_Status": status,
            "Candidates": ";".join(candidates),
            "RHS_Source": rhs,
            "Indus_Kohistani_Forms": ";".join(form for form, _ in forms),
        })
        if not entry_id:
            continue

        for form, dialects in forms:
            # A parenthetical list can name multiple varieties. Indus
            # Kohistani varieties use dialect Language_IDs, which the static
            # DB converts into structured ``dialect:...`` tags. Bhateri and
            # Gowro remain neighboring base languages.
            languages: dict[str, set[str]] = defaultdict(set)
            for dialect in dialects:
                languages[LANGUAGE_BY_DIALECT[dialect]].add(dialect)
            for language in languages:
                key = (language, entry_id, form)
                if key in seen:
                    continue
                seen.add(key)
                notes = (
                    f"Zoller 2005 ch. 6, p. {record['printed_page']}; "
                    f"OIA index head: {raw_oia}"
                )
                # The index supplies no gloss. Do not project Turner's OIA
                # meaning onto the modern reflex.
                rows.append([language, entry_id, form, "", "", "", notes, "zoller2005"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)

    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"wrote {len(rows)} ingestion rows to {output}")
    print(f"wrote {len(audit_rows)} source mappings to {audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "20260724-zoller-indus-kohistani.csv",
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path(__file__).with_name("20260724-zoller-ch6-audit.csv"),
    )
    args = parser.parse_args()
    run(args.pdf, args.repo, args.output, args.audit)


if __name__ == "__main__":
    main()
