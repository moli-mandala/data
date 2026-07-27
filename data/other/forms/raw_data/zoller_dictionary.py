#!/usr/bin/env python3
"""Extract Zoller's complete Indus Kohistani-English dictionary.

The book's phonetic font has no Unicode map, so this reuses the glyph decoder
from ``zoller_ch6.py``.  Printed dictionary definitions are retained, but a
CDIAL gloss is never projected onto a modern lemma.  Entries without a secure
CDIAL link are emitted with a blank Param_ID; ``make_cldf.py`` treats these as
standalone (unetymologised) lemmas.

The generated ingestion CSV deliberately replaces the chapter-6-only import.
Chapter 6 remains useful as a second linking source and as an extraction audit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pdfplumber


HERE = Path(__file__).resolve().parent
CH6_SCRIPT = HERE / "zoller_ch6.py"
_SPEC = importlib.util.spec_from_file_location("zoller_ch6", CH6_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import {CH6_SCRIPT}")
_CH6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CH6)
canonicalize = _CH6.canonicalize
rebuild = _CH6.rebuild


LANGUAGE_BY_DIALECT = {
    "J": "Mai-Jijal",
    "Š": "Mai-Shatot",
    "S": "Mai-Seo",
    "B": "bhatr",
    "G": "Gowro",
}

# Zero-based PDF pages 70..425 are printed dictionary pages 61..416.
FIRST_PDF_INDEX = 70
LAST_PDF_INDEX = 425
ENGLISH_FIRST_PDF_INDEX = 426
ENGLISH_LAST_PDF_INDEX = 485
BODY_TOP = 115
BODY_BOTTOM = 700
COLUMNS = ((130, 300, 138.75), (300, 535, 311.0))

POS_PATTERN = re.compile(
    r"(?<![\w.])(?P<pos>"
    r"n\.[mf]\.|adj(?:\.[mf])?\.?|adv\.?|"
    r"v\.(?:aux\.|hab\.|i\.|t\.|i\./v\.t\.|t\./v\.i\.)|"
    r"interj\.?|excl\.?|onom\.?|pron(?:\.(?:int|adj|adv))?\.?|"
    r"refl\.(?:adv|pron)\.?|obl\.sg\.?|indef\.pron\.?|"
    r"num\.?|part\.?|postp\.?|prep\.?|"
    r"conj\.?|suff(?:ix)?|pref(?:ix)?|particle"
    r")",
    re.IGNORECASE,
)
DIALECT_GROUP = re.compile(r"\(([JBGŠS](?:\??)(?:\s*,\s*[JBGŠS]\??)*)\)")
LEADING_DIALECT = re.compile(r"^(?P<dialect>[JBGŠS])\s+(?=\S)")
PARADIGM_OR_PROSE = re.compile(
    r"^(?:Pres|Preṣ|Fut|Fuṭ|Perf|Plup|Aor|Aoṛ|Cont|Conṭ|Part|Parṭ|"
    r"Conv|Cond|Conḍ|Imp|Subj|Adh|Acc|Acc̣|Ex|Cf|See|Same|Note|"
    r"Only|Other|There|The|This|No passive|Singular|Plural)\b",
    re.IGNORECASE,
)


def clean_dictionary_text(text: str) -> str:
    """Undo punctuation that the TeX-era dot-below heuristic can overread."""
    replacements = {
        "ṇm.": "n.m.",
        "ṇf.": "n.f.",
        "ṣth.": "s.th.",
        "ṣo.": "s.o.",
        "Preṣ": "Pres.",
        "preṣ": "pres.",
        "Cauṣ": "Caus.",
        "cauṣ": "caus.",
        "vṭ": "v.t.",
        "inṭparṭ": "interj.",
        "intenṣ": "intens.",
        "dimiṇ": "dimin.",
        "proṇinṭ": "pron.int.",
        "ṇfḍimiṇ": "n.f. dimin.",
        "ṇmḍimiṇ": "n.m. dimin.",
        "an̈d": "and",
        "preﬁx": "prefix",
        "reﬂ.": "refl.",
        "oblṣg.": "obl.sg.",
        "proṇadj.": "pron.adj.",
        "proṇadvṛeﬂ.": "pron.adv.refl.",
        "indef.proṇ": "indef.pron.",
        "attṛm.": "attr.m.",
        "attṛf.": "attr.f.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clustered_column_lines(page) -> list[list[dict]]:
    """Return layout-preserving text lines for the left and right columns."""
    columns: list[list[tuple[float, float, str]]] = [[], []]
    for line in page.extract_text_lines(layout=False, return_chars=True):
        if not BODY_TOP <= line["top"] <= BODY_BOTTOM:
            continue
        for column, (left, right, _) in enumerate(COLUMNS):
            chars = [char for char in line["chars"] if left <= char["x0"] < right]
            if not chars:
                continue
            text = clean_dictionary_text(rebuild(chars))
            if text:
                columns[column].append(
                    (line["top"], min(char["x0"] for char in chars), text)
                )

    output: list[list[dict]] = [[], []]
    for column, lines in enumerate(columns):
        groups: list[list[tuple[float, float, str]]] = []
        for item in sorted(lines):
            if groups and item[0] - min(part[0] for part in groups[-1]) <= 6:
                groups[-1].append(item)
            else:
                groups.append([item])
        for group in groups:
            output[column].append({
                "top": min(part[0] for part in group),
                "x0": min(part[1] for part in group),
                "text": " ".join(part[2] for part in sorted(group, key=lambda p: p[1])),
            })
    return output


def extract_english_index(pdf: Path) -> list[dict]:
    """Extract the chapter-5 reverse index using its Roman/italic boundary."""
    output: list[dict] = []
    with pdfplumber.open(pdf) as document:
        for page_index in range(ENGLISH_FIRST_PDF_INDEX, ENGLISH_LAST_PDF_INDEX + 1):
            page = document.pages[page_index]
            for column, (left, right, _) in enumerate(COLUMNS):
                fragments: list[tuple[float, list[dict]]] = []
                for line in page.extract_text_lines(layout=False, return_chars=True):
                    if not BODY_TOP <= line["top"] <= BODY_BOTTOM:
                        continue
                    chars = [char for char in line["chars"] if left <= char["x0"] < right]
                    if chars:
                        fragments.append((line["top"], chars))
                groups: list[list[tuple[float, list[dict]]]] = []
                for fragment in sorted(fragments, key=lambda item: item[0]):
                    if groups and fragment[0] - min(part[0] for part in groups[-1]) <= 6:
                        groups[-1].append(fragment)
                    else:
                        groups.append([fragment])

                previous: dict | None = None
                for group in groups:
                    chars = [char for _, part in group for char in part]
                    italic = [
                        char for char in chars
                        if char["fontname"] == "unknown" or "Times-Italic" in char["fontname"]
                    ]
                    if not italic:
                        continue
                    split_x = min(char["x0"] for char in italic)
                    gloss_chars = [char for char in chars if char["x0"] < split_x]
                    form_chars = [char for char in chars if char["x0"] >= split_x]
                    gloss = clean_dictionary_text(rebuild(gloss_chars))
                    form_text = clean_dictionary_text(rebuild(form_chars))
                    if gloss and form_text:
                        previous = {
                            "pdf_page": page_index + 1,
                            "printed_page": page_index - ENGLISH_FIRST_PDF_INDEX + 417,
                            "column": column + 1,
                            "gloss": gloss.strip(" ,"),
                            "form_source": form_text,
                        }
                        output.append(previous)
                    elif previous and form_text:
                        previous["form_source"] += f" {form_text}"
    return output


def is_entry_start(line: dict, base_x: float) -> tuple[bool, bool]:
    """Return (starts entry, is + subentry) from the dictionary indentation."""
    text = line["text"]
    x0 = line["x0"]
    # The + itself sometimes sits at the main headword indent and sometimes at
    # the nominal subentry indent, depending on the surrounding glyph widths.
    if text.startswith("+") and abs(x0 - base_x) <= 6.8:
        return True, True
    if abs(x0 - base_x) <= 1.0:
        # Paradigm lines can be outdented at a column/page break. Everything
        # else at the dictionary's exact headword indent starts an entry,
        # including heads whose POS/definition wraps to the following line.
        if (
            PARADIGM_OR_PROSE.match(text)
            or re.match(r"^(?:[A-ZŠ]|<|←)", text)
            or len(text.strip()) == 1
        ):
            return False, False
        return True, False
    if abs(x0 - (base_x + 5.6)) <= 1.1 and text.startswith("+"):
        return True, True
    return False, False


def extract_entries(pdf: Path) -> list[dict]:
    entries: list[dict] = []
    current: dict | None = None
    parent_head = ""
    parent_dialects = {"J"}

    with pdfplumber.open(pdf) as document:
        for page_index in range(FIRST_PDF_INDEX, LAST_PDF_INDEX + 1):
            page = document.pages[page_index]
            for column, lines in enumerate(clustered_column_lines(page)):
                base_x = COLUMNS[column][2]
                for line in lines:
                    starts, is_subentry = is_entry_start(line, base_x)
                    if starts:
                        if current:
                            current["text"] = clean_dictionary_text(" ".join(current["lines"]))
                            entries.append(current)
                        current = {
                            "pdf_page": page_index + 1,
                            "printed_page": page_index - FIRST_PDF_INDEX + 61,
                            "column": column + 1,
                            "is_subentry": is_subentry,
                            "parent_head": parent_head if is_subentry else "",
                            "parent_dialects": sorted(parent_dialects) if is_subentry else [],
                            "lines": [line["text"]],
                        }
                        if not is_subentry:
                            provisional = parse_entry(current, valid_cdial=set())
                            if provisional["forms"]:
                                parent_head = provisional["forms"][0][0]
                                parent_dialects = provisional["forms"][0][1]
                    elif current:
                        current["lines"].append(line["text"])

    if current:
        current["text"] = clean_dictionary_text(" ".join(current["lines"]))
        entries.append(current)
    return entries


def normalize_pos_text(text: str) -> str:
    return clean_dictionary_text(text).rstrip(".")


def structured_pos_tag(pos: str) -> str:
    pos = pos.lower()
    if pos.startswith("n.m"):
        return "m"
    if pos.startswith("n.f"):
        return "f"
    if pos.startswith("v."):
        return "verb"
    for tag in ("adj", "adv", "pron", "num", "interj", "postp", "prep", "conj", "part"):
        if pos.startswith(tag):
            return tag
    return ""


def parse_dialect_codes(text: str) -> set[str]:
    output: set[str] = set()
    for match in DIALECT_GROUP.finditer(text):
        output.update(code.replace("?", "").strip() for code in match.group(1).split(","))
    leading = LEADING_DIALECT.match(text.strip())
    if leading:
        output.add(leading.group("dialect"))
    return output


def clean_form(text: str) -> str:
    text = DIALECT_GROUP.sub("", text)
    text = LEADING_DIALECT.sub("", text.strip())
    text = re.sub(r"\((?:n|m|f|v)?\d+[a-z]?\)", "", text)
    text = re.sub(r"\((?:note|cf\.|see)\b[^)]*\)", "", text, flags=re.IGNORECASE)
    # Parentheses left at this point contain usage/phonetic commentary rather
    # than part of the citation form. Alternative heads are recovered from the
    # reverse index when the bold Type3 font obscures them here.
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b(?:G|J|Š|S|B) has also\b.*$", "", text)
    text = re.sub(r"\bbut\b.*$", "", text)
    text = re.split(
        r"\s+(?:not an independent form|corresponds to|intens?\.|Note:|"
        r"apparently|due to|the ultrashort vowel|to play|an oblique|obl\. base|"
        r"call (?:produced|of|for)|sound for|the inarticulate|this word|"
        r"only in|prob\. only|however|echo formation|passive meaning|is only|"
        r"used in|a prefix|a derivation|a child|the meaning|Cf\.|Buṛ|Bng\.|"
        r"the squawking|shout (?:of|uttered)|bleating of|fem\. of|dimin\. of|"
        r"allomorph of|said to|cannot be|causative of|obl\. of|reﬂ\.pron|"
        r"usually in|to produce|attr\.[mf]\.|"
        r"the\s+(?:put-put|Khāndia)|echo formatioṇ|reﬂ\.proṇ|attṛ[mf]\.|"
        r"indef\.proṇ|caus\. verbs|"
        r"←|<)",
        text,
        1,
        flags=re.IGNORECASE,
    )[0]
    text = text.strip(" ,;:")
    # Comma-following forms beginning with a hyphen are inflectional endings,
    # not separate dictionary headwords.
    text = re.split(r",\s*-", text, 1)[0].strip()
    # Zoller's ASCII digits distinguish homonyms; they are not phonological.
    text = re.sub(r"(?<=\D)\d+(?!\d)", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return canonicalize(text)


def plausible_form(form: str) -> bool:
    if not form or form.startswith(("(", "<", "←")) or len(form) > 80:
        return False
    if form.count("(") != form.count(")") or form in LANGUAGE_BY_DIALECT:
        return False
    if re.match(
        r"^(?:[A-Z]|a child\b|all the\b|give\b|try\b|type\b|connected\b|"
        r"corresponds\b|precedence\b|"
        r"Š(?:\.|pl\b|erg\b)|[JGBS](?:\.|pl\b|erg\b)|"
        r"(?:pl|erg|nom|obl|gen|dat|acc)\.)",
        form,
    ):
        return False
    if "‘" in form or "’" in form or PARADIGM_OR_PROSE.match(form):
        return False
    if re.search(
        r"\b(?:has|also|and|form|word|with|from|meaning|plural|singular|"
        r"preceding|following|entry|entries|base of|shout|bleating|dimin|"
        r"allomorph|causative|cannot|usually|produce|used|only|prec|type)\b",
        form,
        flags=re.IGNORECASE,
    ):
        return False
    if any(mark in form for mark in ("←", "<", ";")):
        return False
    accentless = "".join(
        char for char in unicodedata.normalize("NFD", form)
        if not unicodedata.combining(char)
    ).lower()
    if re.match(r"^(?:p?e|t?ype) of\b", accentless):
        return False
    if re.search(
        r"\b(?:preﬁx|goodnesṣ|verbṣ|drumbeaṭ|pulsating|perṣproṇ)\b",
        form,
        flags=re.IGNORECASE,
    ):
        return False
    tokens = form.split()
    stripped = [
        "".join(char for char in unicodedata.normalize("NFD", token) if not unicodedata.combining(char))
        for token in tokens
    ]
    if len(tokens) > 1 and stripped[-1].strip("-") in {"v", "m", "f"}:
        return False
    letters = sum(char.isalpha() for char in unicodedata.normalize("NFD", form))
    return letters >= 1


def index_forms(text: str) -> list[tuple[str, set[str]]]:
    output: list[tuple[str, set[str]]] = []
    segments = _CH6.top_level_split(text, separators=(",",))
    for segment in segments:
        variants = re.split(r"\s+or\s+", segment)
        dialects = parse_dialect_codes(segment) or {"J"}
        for variant in variants:
            form = clean_form(variant)
            if plausible_form(form):
                output.append((form, dialects))
    return output


def head_forms(prefix: str, parent_head: str, parent_dialects: set[str]) -> list[tuple[str, set[str]]]:
    is_subentry = prefix.lstrip().startswith("+")
    if is_subentry:
        prefix = prefix.lstrip()[1:].strip()
    dialect_alternatives = re.fullmatch(
        r"([JBGŠS])\s+(.+?)\s+and\s+([JBGŠS])\s+(.+)", prefix
    )
    if dialect_alternatives:
        output = []
        for dialect, raw_form in (
            (dialect_alternatives.group(1), dialect_alternatives.group(2)),
            (dialect_alternatives.group(3), dialect_alternatives.group(4)),
        ):
            form = clean_form(raw_form)
            if is_subentry and parent_head:
                form = parent_head + form[1:] if form.startswith("-") else f"{parent_head} {form}"
            if plausible_form(form):
                output.append((form, {dialect}))
        return output
    segments = [segment.strip() for segment in prefix.split(";") if segment.strip()]
    any_explicit = any(parse_dialect_codes(segment) for segment in segments)
    output: list[tuple[str, set[str]]] = []
    for segment in segments:
        dialects = parse_dialect_codes(segment)
        if not dialects:
            dialects = set(parent_dialects) if is_subentry else ({"J"} if not any_explicit else set())
        if not dialects:
            continue
        comma_parts = _CH6.top_level_split(segment, separators=(",",))
        for variant in (
            item for part in comma_parts for item in re.split(r"\s+(?:or|and)\s+", part)
        ):
            form = clean_form(variant)
            if (
                not form
                or (form.startswith("-") and not is_subentry)
                or re.match(r"^(?:pl|f|m)\.", form)
                or PARADIGM_OR_PROSE.match(form)
            ):
                continue
            if is_subentry and parent_head:
                if form.startswith("-"):
                    form = parent_head + form[1:]
                else:
                    form = f"{parent_head} {form}"
            output.append((form, dialects))
    deduped: list[tuple[str, set[str]]] = []
    seen = set()
    for form, dialects in output:
        key = (form, tuple(sorted(dialects)))
        if key not in seen:
            seen.add(key)
            deduped.append((form, dialects))
    return deduped


def direct_cdial_ids(text: str, valid_cdial: set[str]) -> list[str]:
    candidates = []
    for match in re.finditer(r"(?:<|←)\s*[^.‘’]{0,180}?\((\d{1,5}[a-z]?)\)", text):
        candidate = match.group(1)
        if candidate in valid_cdial and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def derivation_tags(text: str) -> list[str]:
    """Structured labels for analyses Zoller explicitly marks in an entry."""
    tags: list[str] = []
    if "←" in text:
        tags.append("derived")
    if re.search(r"\blw\.|\bloanword\b", text, re.IGNORECASE):
        tags.append("loanword")
    if re.search(r"\bdimin\.", text, re.IGNORECASE):
        tags.append("diminutive")
    if re.search(r"\bcaus\.", text, re.IGNORECASE):
        tags.append("caus")
    if re.search(r"\bintens\.", text, re.IGNORECASE):
        tags.append("intensive")
    if re.search(r"\bcomp\.|(?:←|\bderiv\w*)[^.;]{0,100}\bplus\b", text, re.IGNORECASE):
        tags.append("compound")
    if tags and re.search(r"\b(?:Prob|Perh)\.", text, re.IGNORECASE):
        tags.append("uncertain")
    return list(dict.fromkeys(tags))


def extract_etymology(text: str) -> str:
    """Retain Zoller's own etymological/derivational tail without reconstructing it."""
    starts = []
    for pattern in (
        r"←", r"(?<!<)<(?=\s|\*)",
        r"\b(?:dimin|caus|intens|ext)\.\s+(?:of|from)\b",
        r"\blw\.",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            starts.append(match.start())
    if not starts:
        return ""
    snippet = clean_dictionary_text(text[min(starts):])
    return snippet[:1000].rstrip()


def parse_entry(entry: dict, valid_cdial: set[str]) -> dict:
    text = entry.get("text") or clean_dictionary_text(" ".join(entry["lines"]))
    before_gloss = text.split("‘", 1)[0].strip()
    pos_match = POS_PATTERN.search(before_gloss)
    if pos_match:
        prefix = before_gloss[:pos_match.start()].strip()
        pos = normalize_pos_text(pos_match.group("pos"))
    else:
        prefix = re.split(
            r"\b(?:see|same as|same meaning as|a suffix|a prefix|a shout|"
            r"sound(?: produced)?|the fem\.|dimin\.|intens\.)\b",
            before_gloss,
            1,
            flags=re.IGNORECASE,
        )[0].strip()
        pos = ""
    gloss_match = re.search(r"‘([^’]+)’", text)
    gloss = clean_dictionary_text(gloss_match.group(1)) if gloss_match else ""
    parent_dialects = set(entry.get("parent_dialects") or ["J"])
    forms = head_forms(prefix, entry.get("parent_head", ""), parent_dialects)
    cdial = direct_cdial_ids(text, valid_cdial)
    return {
        **entry,
        "head_source": prefix,
        "forms": forms,
        "pos": pos,
        "pos_tag": structured_pos_tag(pos),
        "gloss": gloss,
        "cdial_candidates": cdial,
        "cdial_id": cdial[0] if len(cdial) == 1 else "",
        "derivation_tags": derivation_tags(text),
        "etymology": extract_etymology(text),
        "text": text,
    }


def load_valid_cdial(repo: Path) -> set[str]:
    with (repo / "cldf/forms.csv").open(encoding="utf-8") as stream:
        return {
            row["ID"] for row in csv.DictReader(stream)
            if row["Language_ID"] == "Indo-Aryan" and "CDIAL" in row["Source"]
        }


def form_key(text: str) -> str:
    text = canonicalize(text).lower().strip(" -")
    return re.sub(r"\s+", " ", text)


def load_ch6_links(path: Path) -> tuple[dict[tuple[str, str], list[str]], list[list[str]]]:
    links: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows: list[list[str]] = []
    if not path.exists():
        return links, rows
    with path.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != 8:
                continue
            rows.append(row)
            key = (row[0], form_key(row[2]))
            if row[1] and row[1] not in links[key]:
                links[key].append(row[1])
    return links, rows


def merge_rows(rows: Iterable[list[str]]) -> list[list[str]]:
    """Merge repeated attestations while retaining all direct senses/pages."""
    merged: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        key = (row[0], row[1], row[2])
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        for index in (3, 6):
            values = [value.strip() for value in (existing[index], row[index]) if value.strip()]
            existing[index] = "; ".join(dict.fromkeys(values))
        # Etymological tails may themselves contain many semicolons. Repeated
        # attestations of the same lemma should not recursively concatenate
        # the growing composite string; prefer the first direct analysis and
        # fill it only when that attestation had none.
        existing[9] = existing[9] or row[9]
    return list(merged.values())


def run(pdf: Path, repo: Path, output: Path, audit: Path, chapter6_csv: Path) -> None:
    valid_cdial = load_valid_cdial(repo)
    ch6_links, ch6_rows = load_ch6_links(chapter6_csv)
    parsed = [parse_entry(entry, valid_cdial) for entry in extract_entries(pdf)]
    english_index = extract_english_index(pdf)

    rows: list[list[str]] = []
    audit_rows: list[dict] = []
    covered_ch6: set[tuple[str, str, str]] = set()
    for entry in parsed:
        direct_id = entry["cdial_id"]
        for form, dialects in entry["forms"]:
            if not plausible_form(form):
                continue
            for dialect in sorted(dialects):
                language = LANGUAGE_BY_DIALECT[dialect]
                index_candidates = ch6_links.get((language, form_key(form)), [])
                if direct_id:
                    entry_id = direct_id
                    link_source = "dictionary"
                elif len(index_candidates) == 1:
                    entry_id = index_candidates[0]
                    link_source = "chapter-6"
                else:
                    entry_id = ""
                    link_source = "unlinked"
                if entry_id:
                    covered_ch6.add((language, entry_id, form_key(form)))
                notes = []
                if entry["pos_tag"]:
                    notes.append(entry["pos_tag"])
                notes.extend(entry["derivation_tags"])
                notes.append(
                    f"Zoller 2005 ch. 4, p. {entry['printed_page']}; "
                    f"dictionary head: {entry['head_source']}"
                )
                if entry["pos"]:
                    notes.append(f"dictionary POS: {entry['pos']}")
                rows.append([
                    language, entry_id, form, entry["gloss"], "", "",
                    "; ".join(notes), "zoller2005", "", entry["etymology"],
                ])
                audit_rows.append({
                    "PDF_Page": entry["pdf_page"],
                    "Printed_Page": entry["printed_page"],
                    "Column": entry["column"],
                    "Head_Source": entry["head_source"],
                    "Form": form,
                    "Dialect": dialect,
                    "Language_ID": language,
                    "POS": entry["pos"],
                    "Gloss": entry["gloss"],
                    "CDIAL_ID": entry_id,
                    "Link_Source": link_source,
                    "CDIAL_Candidates": ";".join(entry["cdial_candidates"] or index_candidates),
                    "Entry_Text": entry["text"],
                })

    # Chapter 5 recovers headwords whose bold glyphs in chapter 4 have no text
    # operators. Its inverted English headings (e.g. ``squirt, to``) are index
    # lookup labels, not dictionary definitions, so retain them only as page
    # provenance and leave the gloss blank.
    main_form_keys = {(row[0], form_key(row[2])) for row in rows}
    secure_links: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row[1] and row[1] not in secure_links[(row[0], form_key(row[2]))]:
            secure_links[(row[0], form_key(row[2]))].append(row[1])
    for item in english_index:
        for form, dialects in index_forms(item["form_source"]):
            for dialect in sorted(dialects):
                language = LANGUAGE_BY_DIALECT[dialect]
                form_lookup = (language, form_key(form))
                if form_lookup in main_form_keys:
                    continue
                candidates = secure_links.get(form_lookup, []) or ch6_links.get(form_lookup, [])
                entry_id = candidates[0] if len(candidates) == 1 else ""
                notes = (
                    f"Zoller 2005 ch. 5, p. {item['printed_page']}; "
                    f"English index head: {item['gloss']}"
                )
                rows.append([language, entry_id, form, "", "", "", notes, "zoller2005", "", ""])
                main_form_keys.add(form_lookup)

    # Retain chapter-6 forms missed by layout/head parsing. Its gloss column is
    # intentionally ignored because it was a CDIAL proxy, not Zoller's gloss.
    for row in ch6_rows:
        key = (row[0], row[1], form_key(row[2]))
        if key in covered_ch6:
            continue
        notes = re.sub(r";?\s*gloss from mapped CDIAL entry", "", row[6])
        notes = re.sub(r";?\s*dialect:\s*[^;]+", "", notes)
        rows.append([row[0], row[1], row[2], "", row[4], row[5], notes, row[7], "", ""])

    rows = merge_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)

    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    linked = sum(bool(row[1]) for row in rows)
    standalone = len(rows) - linked
    print(f"wrote {len(rows)} rows to {output}: {linked} linked, {standalone} standalone")
    print(f"wrote {len(audit_rows)} parsed attestations to {audit}")


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
        default=Path(__file__).with_name("20260725-zoller-dictionary-audit.csv"),
    )
    parser.add_argument(
        "--chapter6-csv", type=Path,
        default=Path(__file__).with_name("20260724-zoller-ch6-ingestion.csv"),
    )
    args = parser.parse_args()
    run(args.pdf, args.repo, args.output, args.audit, args.chapter6_csv)


if __name__ == "__main__":
    main()
