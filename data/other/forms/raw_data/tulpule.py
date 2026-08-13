#!/usr/bin/env python3
"""Import securely matched Old Marathi entries from Tulpule and Feldhaus.

The Digital Dictionaries of South Asia site exposes one HTML result page per
printed dictionary page.  This importer caches those pages, parses the
Devanagari/roman headword pairs, grammatical label, English definition, and
bracketed etymology, then emits a Jambu row only when the primary Sanskrit or
explicit CDIAL etymon has one unique existing Jambu match.

Run from ``data/``::

    uv run --with beautifulsoup4 --with lxml python \
        data/other/forms/raw_data/tulpule.py

Unmatched and ambiguous records remain in the audit CSV.  Cached pages make
the crawl resumable and avoid requesting the same public page twice.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


BASE_URL = "https://dsal.uchicago.edu/cgi-bin/app/tulpule_query.py?page={}"
USER_AGENT = "Jambu dictionary importer/1.0 (https://github.com/moli-mandala)"
SOURCE_ID = "tulpule1999"
FIRST_PAGE = 1
LAST_PAGE = 807
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "data/other/forms/20260810-tulpule-old-marathi.csv"
DEFAULT_AUDIT = ROOT / "tmp/tulpule-old-marathi-audit.csv"
DEFAULT_CACHE = ROOT / "tmp/tulpule-old-marathi-cache"
DEFAULT_CDIAL = ROOT / "data/cdial/params.csv"
DEFAULT_FORMS = ROOT / "cldf/forms.csv"
DEFAULT_ALIASES = ROOT / "cldf/form-id-aliases.csv"


@dataclass(frozen=True)
class Entry:
    page: int
    ordinal: int
    forms: tuple[tuple[str, str], ...]
    grammar: str
    tags: str
    etymology: str
    gloss: str


def plain(value: str | Tag | None) -> str:
    if value is None:
        return ""
    if isinstance(value, Tag):
        value = value.get_text(" ", strip=True)
    value = html.unescape(str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_match(value: str) -> str:
    """Strict Sanskrit lookup key, folding notation and accents but not segments."""
    value = html.unescape(value).casefold().strip()
    value = value.replace("ṃ", "ṁ").replace("m̐", "ṁ")
    value = re.sub(r"^(?:sk\.|skt\.|sanskrit|cf\.)\s*", "", value)
    value = re.sub(r"^[*†‡]+|[-‐‑–—*†‡?.,;:¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", value)
    accents = {"\u0300", "\u0301", "\u0302", "\u0340", "\u0341", "\u0951", "\u0952"}
    chars = [c for c in unicodedata.normalize("NFD", value) if c not in accents]
    value = unicodedata.normalize("NFC", "".join(chars))
    return re.sub(r"[\s.·'’(){}\[\]]+", "", value)


def normalize_form(native: str, roman: str) -> str:
    """Normalize the source romanization to Jambu's CDIAL-style conventions."""
    value = unicodedata.normalize("NFC", plain(roman)).replace("ṃ", "ṁ")
    # The site's transliteration occasionally renders word-final anusvāra as
    # plain m even though the paired Devanagari ends in U+0902.
    if native.rstrip().endswith("ं") and value.endswith("m"):
        value = value[:-1] + "ṁ"
    return value


def grammatical_tags(label: str) -> str:
    """Map Tulpule's compact grammar labels to canonical Jambu tags."""
    text = plain(label).casefold()
    tags: list[str] = []
    gender = [(r"(?<!\w)m\.", "m"), (r"(?<!\w)f\.", "f"), (r"(?<!\w)n\.", "n")]
    for pattern, tag in gender:
        if re.search(pattern, text):
            tags.append(tag)
    mappings = [
        (r"\b(?:vi|v\.\s*i)\.", ("intr", "verb")),
        (r"\b(?:vt|v\.\s*t)\.", ("tr", "verb")),
        (r"\bv\.", ("verb",)),
        (r"\badj\.", ("adj",)),
        (r"\badv\.", ("adv",)),
        (r"\bpron\.", ("pron",)),
        (r"\bnum\.", ("num",)),
        (r"\bprep\.", ("prep",)),
        (r"\bpostp(?:ost)?\.", ("postp",)),
        (r"\bpostpos\.", ("postp",)),
        (r"\bconj\.", ("conj",)),
        (r"\binterj\.", ("interj",)),
        (r"\bindecl\.", ("indecl",)),
        (r"\bind\.", ("indecl",)),
        (r"\bpart\.", ("part",)),
        (r"\bsuff(?:ix)?\.", ("suffix",)),
        (r"\bprefix\b", ("prefix",)),
        (r"\bpl\.", ("pl",)),
        (r"\bdat\.", ("dat",)),
        (r"\bloc\.", ("loc",)),
        (r"\babl\.", ("abl",)),
        (r"\binstr\.", ("instr",)),
        (r"\binterrog\.", ("interr",)),
    ]
    for pattern, values in mappings:
        if re.search(pattern, text):
            tags.extend(values)
    numbered_noun = bool(re.search(r"(?:^|\b\d+\s+)(?:m|f|n)\.", text))
    if any(tag in tags for tag in ("m", "f", "n")) and (
        numbered_noun or not any(tag in tags for tag in ("verb", "adj", "adv", "pron", "num"))
    ):
        tags.append("noun")
    return " ".join(dict.fromkeys(tags))


def _prefix_after_hw(container: Tag, hw: Tag) -> str:
    pieces = []
    for node in hw.next_siblings:
        if isinstance(node, Tag) and node.name == "d":
            break
        pieces.append(plain(node))
    return plain(" ".join(pieces))


def _etymology(prefix: str) -> str:
    matches = re.findall(r"\[([^]]+)\]", prefix)
    return next((m.strip() for m in matches if re.search(r"\b(?:Sk|Skt|CDIAL)\.", m)), "")


def _grammar(prefix: str, container: Tag) -> str:
    values = [plain(re.sub(r"\[[^]]+\]", "", prefix))]
    for marker in container.find_all("b"):
        if not plain(marker).isdigit():
            continue
        pieces = []
        for node in marker.next_siblings:
            if isinstance(node, Tag) and node.name == "d":
                break
            pieces.append(plain(node))
        values.append(plain(f"{plain(marker)} {' '.join(pieces)}"))
    return plain(" ".join(dict.fromkeys(value for value in values if value)))


def _definition_nodes(container: Tag, hw: Tag) -> list[Tag]:
    descendants = list(container.descendants)
    hw_pos = descendants.index(hw)
    ds = [d for d in container.find_all("d") if descendants.index(d) > hw_pos]
    if not ds:
        return []
    result = [ds[0]]
    for d in ds[1:]:
        node = d.previous_sibling
        while node is not None:
            if isinstance(node, Tag) and node.name == "d":
                break
            if isinstance(node, Tag) and node.name == "b" and plain(node).isdigit():
                result.append(d)
                break
            node = node.previous_sibling
    return result


def _english_after(d: Tag) -> str:
    pieces = []
    for node in d.next_siblings:
        if isinstance(node, Tag) and (node.name == "d" or (node.name == "b" and plain(node).isdigit())):
            break
        pieces.append(plain(node))
    value = plain(" ".join(pieces))
    value = re.sub(r"^[;,:.\s()]+|[;\s]+$", "", value)
    if value.endswith(")") and "(" not in value:
        value = value[:-1]
    if value.endswith(").") and "(" not in value:
        value = value[:-2] + "."
    if not re.search(r"\b[a-z]{2,}\b", value):
        return ""
    return value


def parse_page(page: int, source: str) -> list[Entry]:
    soup = BeautifulSoup(source, "lxml")
    entries = []
    previous_tags = ""
    for ordinal, hw in enumerate(soup.find_all("hw"), 1):
        # Result pages wrap every entry in its immediate div; the surrounding
        # page block also has ``px-4``, so a class-based ancestor lookup would
        # accidentally merge the definitions of every entry on that page.
        container = hw.parent
        bold = [plain(x) for x in hw.find_all("b", recursive=False)]
        if len(bold) < 2 or len(bold) % 2:
            continue
        forms = tuple(
            (bold[i], normalize_form(bold[i], bold[i + 1])) for i in range(0, len(bold), 2)
        )
        prefix = _prefix_after_hw(container, hw)
        grammar = _grammar(prefix, container)
        definitions = [_english_after(d) for d in _definition_nodes(container, hw)]
        definitions = [d for d in definitions if d]
        tags = grammatical_tags(grammar)
        if grammar.casefold().strip() == "id." and previous_tags:
            tags = previous_tags
        if tags:
            previous_tags = tags
        entries.append(
            Entry(
                page=page,
                ordinal=ordinal,
                forms=forms,
                grammar=grammar,
                tags=tags,
                etymology=_etymology(prefix),
                gloss="; ".join(dict.fromkeys(definitions)),
            )
        )
    return entries


def jambu_index(
    cdial_path: Path = DEFAULT_CDIAL,
    forms_path: Path = DEFAULT_FORMS,
    aliases_path: Path = DEFAULT_ALIASES,
) -> dict[str, list[tuple[str, str]]]:
    """Index numeric CDIAL heads and Sanskrit-root aliases already in Jambu."""
    index: dict[str, list[tuple[str, str]]] = {}

    def add(key: str, ident: str, form: str) -> None:
        if key and (ident, form) not in index.setdefault(key, []):
            index[key].append((ident, form))

    with cdial_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 2:
                continue
            for form in re.split(r"\s*,\s*", row[1]):
                add(normalize_match(form), row[0], form)

    if forms_path.exists() and aliases_path.exists():
        with aliases_path.open(encoding="utf-8", newline="") as stream:
            aliases = list(csv.DictReader(stream))
        root_alias = {
            row.get("Canonical_ID", row.get("Form_ID", "")):
            row.get("Alias_ID", row.get("Legacy_ID", ""))
            for row in aliases
            if re.fullmatch(r"r\d+", row.get("Alias_ID", row.get("Legacy_ID", "")))
        }
        with forms_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                form = row.get("Form", "")
                alias = root_alias.get(row.get("ID", ""))
                if alias and row.get("Status") == "entry" and form.startswith("√"):
                    add(normalize_match(form), alias, form)
    return index


def etymon_candidates(etymology: str, form: str) -> tuple[list[str], list[str]]:
    """Return explicit CDIAL IDs and primary Sanskrit lookup forms."""
    direct = re.findall(r"(?:CDIAL|Turner)\s*(?:no\.?\s*)?(\d+[a-z]?)", etymology, flags=re.I)
    sanskrit = []
    markers = list(re.finditer(r"\b(?:Sk|Skt)\.\s*", etymology, flags=re.I))
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(etymology)
        value = etymology[marker.end() : end]
        value = re.split(r"/\s*cf\.", value, maxsplit=1, flags=re.I)[0]
        value = value.strip(" /;,:()")
        if not value or value.casefold() == "cf.":
            value = form
        if value:
            sanskrit.append(value)
    return direct, sanskrit


def match_etymon(entry: Entry, index: dict[str, list[tuple[str, str]]]):
    if not entry.etymology:
        return "", [], "no-etymology"
    direct, forms = etymon_candidates(entry.etymology, entry.forms[0][1])
    matches = {(ident, head) for value in forms for ident, head in index.get(normalize_match(value), ())}
    matches.update((ident, f"CDIAL {ident}") for ident in direct if any(ident == p[0] for v in index.values() for p in v))
    ids = {ident for ident, _ in matches}
    if len(ids) == 1:
        ident = next(iter(ids))
        return ident, sorted(head for match_id, head in matches if match_id == ident), "matched"
    if ids:
        return "", sorted(f"{ident}:{head}" for ident, head in matches), "ambiguous"
    return "", [], "unmatched"


def fetch_page(page: int, cache_dir: Path, attempts: int = 5) -> str:
    path = cache_dir / f"{page:03d}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    request = urllib.request.Request(BASE_URL.format(page), headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                source = response.read().decode("utf-8")
            path.write_text(source, encoding="utf-8")
            return source
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_all(pages: list[int], cache_dir: Path, workers: int) -> list[Entry]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_page, page, cache_dir): page for page in pages}
        for done, future in enumerate(as_completed(futures), 1):
            page = futures[future]
            entries.extend(parse_page(page, future.result()))
            if done % 50 == 0 or done == len(pages):
                print(f"fetched {done}/{len(pages)} pages", flush=True)
    return sorted(entries, key=lambda entry: (entry.page, entry.ordinal))


def build(entries: list[Entry], output: Path, audit: Path, index=None):
    index = index or jambu_index()
    rows, audit_rows = [], []
    counts: dict[str, int] = {}
    for entry in entries:
        etymon_id, candidates, status = match_etymon(entry, index)
        if status == "matched" and not entry.gloss:
            status = "missing-definition"
        elif status == "matched" and not entry.tags:
            status = "missing-grammar"
        counts[status] = counts.get(status, 0) + 1
        if status == "matched":
            primary_key = f"tulpule:p{entry.page}:e{entry.ordinal}:v1"
            for variant, (native, roman) in enumerate(entry.forms, 1):
                key = f"tulpule:p{entry.page}:e{entry.ordinal}:v{variant}"
                rows.append([
                    "OM", etymon_id, roman, entry.gloss, native, "", "",
                    f"{SOURCE_ID}[p. {entry.page}, entry {entry.ordinal}]", "",
                    entry.etymology, key, primary_key if variant > 1 else "", "", "",
                    entry.tags,
                ])
        audit_rows.append([
            entry.page, entry.ordinal, " | ".join(x[0] for x in entry.forms),
            " | ".join(x[1] for x in entry.forms), entry.grammar, entry.tags,
            entry.etymology, entry.gloss, status, etymon_id, " | ".join(candidates),
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    with audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "Page", "Entry", "Native", "Roman", "Grammar", "Tags", "Etymology",
            "Gloss", "Status", "Jambu_ID", "Candidates",
        ])
        writer.writerows(audit_rows)
    print(f"wrote {len(rows)} rows to {output}; audit counts: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--first-page", type=int, default=FIRST_PAGE)
    parser.add_argument("--last-page", type=int, default=LAST_PAGE)
    args = parser.parse_args()
    pages = list(range(args.first_page, args.last_page + 1))
    build(fetch_all(pages, args.cache, max(1, args.workers)), args.output, args.audit)


if __name__ == "__main__":
    main()
