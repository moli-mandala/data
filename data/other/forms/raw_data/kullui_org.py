#!/usr/bin/env python3
"""Snapshot the public Kullui dictionary and link its OIA etyma to CDIAL.

The browser at https://kullui.org/ uses two public JSON endpoints.  Search
results enumerate the lexemes and ``article.php`` returns the complete record.
The site is newer and substantially richer than the July 2023 PDF export, so
the API is the canonical input; the PDF is useful only as a historical check.

Typical refresh (run from ``data/``)::

    uv run python data/other/forms/raw_data/kullui_org.py

Responses are cached under ``tmp/kullui-org-cache``.  The checked-in import has
one row per source article.  A CDIAL link is made only when an explicitly
labelled Old Indo-Aryan/Sanskrit protoform has one exact normalized match.
Ambiguous, unmatched, and non-IA analyses remain visible in the audit CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://kullui.org/php3"
SEARCH_URL = f"{BASE_URL}/search.php"
ARTICLE_URL = f"{BASE_URL}/article.php"
HELLO_URL = f"{BASE_URL}/hello.php"
USER_AGENT = "Jambu dictionary importer/1.0 (https://github.com/moli-mandala)"

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "data/other/forms/20260813-kullui-org.csv"
DEFAULT_AUDIT = ROOT / "tmp/kullui-org-audit.csv"
DEFAULT_CACHE = ROOT / "tmp/kullui-org-cache"
DEFAULT_CDIAL = ROOT / "data/cdial/params.csv"

# Searching every character in the site's transcription alphabet makes the
# enumeration independent of an undocumented "list all" request (which the
# API rejects).  ASCII letters already cover the current data; the additional
# IPA characters make future snapshots robust to a headword written entirely
# with non-ASCII phonetic symbols.
SEARCH_CHARACTERS = tuple(dict.fromkeys(
    "abcdefghijklmnopqrstuvwxyz"
    "əɐɛɔɪʊæɑɒɘɵɜɞɤɯɳɖɽɭʃʒʰŋɲʈɟɡɣχʁɦʔθðʂʐɕʑçʝβɸʋɱ"
))
PAGE_SIZE = 100

IA_SOURCE = re.compile(r"\b(?:old\s+indo-?aryan|oia|sanskrit|skr\.?|skt\.?)\b", re.I)
IA_MARKER = re.compile(
    r"\b(?:old\s+indo-?aryan|oia|sanskrit|skr\.?|skt\.?)\s*[:*]?\s*",
    re.I,
)


def api_get(url: str, params: list[tuple[str, str]] | None = None, attempts: int = 5) -> dict:
    query = urllib.parse.urlencode(params or [])
    request = urllib.request.Request(
        f"{url}?{query}" if query else url,
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                packet = json.load(response)
            if packet.get("result") != "ok":
                raise RuntimeError(f"Kullui API error: {packet!r}")
            return packet["data"]
        except Exception:  # transient network/API errors are retried, then surfaced
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def search_page(character: str, skip: int, take: int = PAGE_SIZE) -> dict:
    return api_get(
        SEARCH_URL,
        [
            ("search", character),
            ("mode[]", "lexeme"),
            ("skip", str(skip)),
            ("take", str(take)),
            ("with_total_count", "1"),
        ],
    )


def enumerate_lexemes() -> list[str]:
    lexemes: set[str] = set()
    for index, character in enumerate(SEARCH_CHARACTERS, 1):
        skip = 0
        while True:
            result = search_page(character, skip)
            items = result.get("items", [])
            lexemes.update(item["lexeme"] for item in items)
            skip += len(items)
            if not items or skip >= int(result.get("total_count", skip)):
                break
        if index % 10 == 0 or index == len(SEARCH_CHARACTERS):
            print(f"enumerated {len(lexemes)} unique lexemes", flush=True)
    return sorted(lexemes, key=lambda value: unicodedata.normalize("NFD", value).casefold())


def _cache_name(lexeme: str) -> str:
    return hashlib.sha256(lexeme.encode("utf-8")).hexdigest() + ".json"


def fetch_article(lexeme: str, cache_dir: Path) -> dict:
    cache_file = cache_dir / "articles" / _cache_name(lexeme)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    article = api_get(ARTICLE_URL, [("lexeme", lexeme)])
    if article.get("lexeme") != lexeme or not isinstance(article.get("id"), int):
        raise ValueError(f"invalid article response for {lexeme!r}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(article, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(cache_file)
    return article


def fetch_all(lexemes: list[str], cache_dir: Path, workers: int) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_article, lexeme, cache_dir): lexeme for lexeme in lexemes}
        for done, future in enumerate(as_completed(futures), 1):
            entries.append(future.result())
            if done % 250 == 0 or done == len(lexemes):
                print(f"fetched {done}/{len(lexemes)} articles", flush=True)
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Kullui API returned duplicate article IDs")
    return sorted(entries, key=lambda entry: entry["id"])


def normalize_oia(value: str) -> str:
    """Normalize notation and accent while preserving segmental distinctions."""
    value = unicodedata.normalize("NFC", value).casefold().strip()
    value = value.replace("ṁ", "ṃ").replace("m̐", "ṃ").replace("r̥", "ṛ")
    value = re.sub(r"^[*†‡√?]+", "", value)
    value = re.sub(r"[?¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", value)
    value = value.replace("ˊ", "").replace("ˋ", "")
    chars = []
    accents = {"\u0300", "\u0301", "\u0302", "\u0340", "\u0341", "\u0951", "\u0952"}
    for char in unicodedata.normalize("NFD", value):
        if char not in accents:
            chars.append(char)
    value = unicodedata.normalize("NFC", "".join(chars))
    return re.sub(r"[\s\-‐‑–—.,;:'’ʻ‘\"(){}\[\]<>]+", "", value)


def _clean_candidate(value: str) -> str:
    value = value.strip()
    # English meanings are frequently quoted immediately after the form.
    value = re.split(r"\s+[\"'ʻ‘]", value, maxsplit=1)[0]
    value = re.split(r"\s+(?:meaning|lit\.)\b", value, maxsplit=1, flags=re.I)[0]
    return value.strip(" *†‡?:;,.()[]{}")


def extract_oia_etyma(article: dict) -> list[str]:
    """Return explicitly identified OIA/Sanskrit protoforms from an article."""
    protoform = (article.get("protoform") or "").strip()
    source = ((article.get("source") or {}).get("english") or "").strip()
    values: list[str] = []

    if IA_SOURCE.search(source) and protoform:
        # A direct OIA/Sanskrit source makes the complete protoform relevant.
        segment = re.split(r"\s+[<←]\s+", protoform, maxsplit=1)[0]
        values.extend(re.split(r"\s+or\s+|\s*[,;/]\s*|\s+(?=[*†])", segment))

    markers = list(IA_MARKER.finditer(protoform))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(protoform)
        segment = protoform[marker.end():end]
        segment = re.split(r"\s+[;<>←]\s+|\s+\|\s+", segment, maxsplit=1)[0]
        values.extend(re.split(r"\s+or\s+|\s*[,/]\s*|\s+(?=[*†])", segment))

    result = []
    for value in values:
        value = _clean_candidate(value)
        if normalize_oia(value) and value not in result:
            result.append(value)
    return result


def normalize_meaning(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value or "")).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def quoted_meanings(description: str) -> tuple[str, ...]:
    return tuple(
        key for value in re.findall(r"['‘](.*?)['’]", description or "")
        if (key := normalize_meaning(value))
    )


def cdial_index(path: Path = DEFAULT_CDIAL) -> dict[str, list[tuple[str, str, tuple[str, ...]]]]:
    raw_rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            raw_rows.append(row)

    # CDIAL addenda (14190--14845) are folded into a main entry downstream.
    # Collapse an addendum here only when its normalized head has one main
    # candidate; true main-volume homographs remain separate candidates.
    mains: dict[str, list[str]] = {}
    for row in raw_rows:
        cdial_id, heads = row[:2]
        if not (cdial_id.isdigit() and 14190 <= int(cdial_id) <= 14845):
            for head in re.split(r"\s*,\s*", heads):
                if key := normalize_oia(head):
                    mains.setdefault(key, []).append(cdial_id)
    redirects = {}
    for row in raw_rows:
        cdial_id, heads = row[:2]
        if cdial_id.isdigit() and 14190 <= int(cdial_id) <= 14845:
            candidates = {
                candidate
                for head in re.split(r"\s*,\s*", heads)
                for candidate in mains.get(normalize_oia(head), ())
            }
            if len(candidates) == 1:
                redirects[cdial_id] = next(iter(candidates))

    index: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
    for row in raw_rows:
        cdial_id, heads = row[:2]
        cdial_id = redirects.get(cdial_id, cdial_id)
        meanings = quoted_meanings(row[3] if len(row) > 3 else "")
        for head in re.split(r"\s*,\s*", heads):
            key = normalize_oia(head)
            pair = (cdial_id, head, meanings)
            if key and pair not in index.setdefault(key, []):
                index[key].append(pair)
    return index


def match_cdial(
    etyma: list[str],
    index: dict[str, list[tuple[str, str, tuple[str, ...]]]],
    proto_meaning: str = "",
):
    matches = {
        pair
        for etymon in etyma
        for pair in index.get(normalize_oia(etymon), ())
    }
    ids = {pair[0] for pair in matches}
    if len(ids) > 1 and proto_meaning:
        source_meanings = {
            key for value in re.findall(r"['‘](.*?)['’]", proto_meaning)
            if (key := normalize_meaning(value))
        }
        semantic_ids = {
            ident for ident, _head, meanings in matches
            if source_meanings.intersection(meanings)
        }
        if len(semantic_ids) == 1:
            ids = semantic_ids
    if len(ids) == 1:
        cdial_id = next(iter(ids))
        return cdial_id, sorted(head for ident, head, _ in matches if ident == cdial_id), "matched"
    if ids:
        return "", sorted(f"{ident}:{head}" for ident, head, _ in matches), "ambiguous"
    return "", [], "unmatched" if etyma else "not-applicable"


def english(value) -> str:
    return ((value or {}).get("english") or "").strip()


def grammar_tags(article: dict) -> list[str]:
    tags: list[str] = []
    grammars = [
        (translation.get("grammar_info") or "").casefold().strip()
        for translation in article.get("translations", [])
    ]
    grammar = " ".join(grammars)
    source = english(article.get("source"))
    origin = english(article.get("origin"))
    if any(re.match(r"n(?:[1-7]|prop|\b)", item) for item in grammars):
        tags.append("noun")
    if "nprop" in grammar:
        tags.append("proper-noun")
    if "adj" in grammar:
        tags.append("adj")
    if "adv" in grammar:
        tags.append("adv")
    if any(item.startswith(("cardnum", "ordnum")) for item in grammars):
        tags.append("num")
    if any(item.startswith("ordnum") for item in grammars):
        tags.append("ord")
    if any(item.startswith(("indfpro", "interrog", "poss", "relpro")) for item in grammars):
        tags.append("pron")
    if any(item.startswith("indfpro") for item in grammars):
        tags.append("indef")
    if any(item.startswith("interrog") for item in grammars):
        tags.append("interr")
    if any(item.startswith("poss") for item in grammars):
        tags.append("poss")
    if any(item.startswith("post") for item in grammars):
        tags.append("postp")
    if any(item.startswith("prt") for item in grammars):
        tags.append("part")
    if any(item.startswith("conj") for item in grammars):
        tags.append("conj")
    if any(item.startswith("interj") for item in grammars):
        tags.append("interj")
    if any(item.startswith("aux") for item in grammars):
        tags.extend(["verb", "auxiliary"])
    if any(re.match(r"v(?:t|i|\b)", item) for item in grammars):
        tags.append("verb")
    if "vt" in grammar:
        tags.append("tr")
    if "vi" in grammar:
        tags.append("intr")
    if re.search(r"(?:^|_)m(?:-|\b)", grammar):
        tags.append("m")
    if re.search(r"(?:^|_)f(?:-|\b)", grammar):
        tags.append("f")
    if "sg" in grammar:
        tags.append("sg")
    if "pl" in grammar:
        tags.append("pl")
    if "loan" in origin.casefold():
        tags.append("loanword")
    elif IA_SOURCE.search(source):
        tags.append("inherited")
    return list(dict.fromkeys(tags))


def combined_gloss(article: dict) -> str:
    glosses = [english(item.get("translation_text")) for item in article.get("translations", [])]
    glosses = [gloss for gloss in glosses if gloss]
    if len(glosses) < 2:
        return glosses[0] if glosses else ""
    return "; ".join(f"{index}) {gloss}" for index, gloss in enumerate(glosses, 1))


def etymology_text(article: dict) -> str:
    fields = [
        ("Origin", english(article.get("origin"))),
        ("Source", english(article.get("source"))),
        ("Protoform", article.get("protoform") or ""),
        ("Proto-meaning", english(article.get("proto_meaning"))),
        ("Analysis", english(article.get("etymology"))),
        ("Ethnocultural note", english(article.get("ethnocultural"))),
    ]
    return "; ".join(
        f"{label}: {value.strip(' ;')}" for label, value in fields if value.strip(" ;")
    )


def source_row(article: dict, cdial_id: str) -> list[str]:
    article_id = str(article["id"])
    lexeme = article["lexeme"]
    origin = english(article.get("origin")).casefold()
    parameter = f">{cdial_id}" if cdial_id and "loan" in origin else cdial_id
    return [
        "kul",
        parameter,
        lexeme,
        combined_gloss(article),
        article.get("orthography") or "",
        lexeme,
        "",
        f"kullui-org[article {article_id}]",
        "",
        etymology_text(article),
        f"kullui:{article_id}",
        "",
        "",
        "",
        " ".join(grammar_tags(article)),
    ]


def build(entries: list[dict], output: Path, audit: Path, cdial_path: Path = DEFAULT_CDIAL):
    index = cdial_index(cdial_path)
    rows, audit_rows = [], []
    counts = {"matched": 0, "ambiguous": 0, "unmatched": 0, "not-applicable": 0}
    for entry in entries:
        etyma = extract_oia_etyma(entry)
        cdial_id, candidates, status = match_cdial(
            etyma, index, english(entry.get("proto_meaning"))
        )
        counts[status] += 1
        rows.append(source_row(entry, cdial_id))
        audit_rows.append([
            entry["id"], entry["lexeme"], " | ".join(etyma), status,
            cdial_id, " | ".join(candidates), english(entry.get("source")),
            entry.get("protoform") or "", combined_gloss(entry),
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Article_ID", "Lexeme", "OIA_Etyma", "Status", "CDIAL_ID",
            "Candidates", "Source", "Protoform", "Gloss",
        ])
        writer.writerows(audit_rows)
    print(f"wrote {len(rows)} rows to {output}; audit counts: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cdial", type=Path, default=DEFAULT_CDIAL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lexemes", help="comma-separated lexemes (for testing)")
    parser.add_argument(
        "--offline", action="store_true",
        help="rebuild from the existing article cache without contacting the API",
    )
    args = parser.parse_args()

    if args.offline:
        entries = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((args.cache / "articles").glob("*.json"))
        ]
        if not entries:
            raise FileNotFoundError(f"no cached articles under {args.cache / 'articles'}")
        version = json.loads((args.cache / "version.json").read_text(encoding="utf-8"))
    else:
        lexemes = args.lexemes.split(",") if args.lexemes else enumerate_lexemes()
        entries = fetch_all(lexemes, args.cache, max(1, args.workers))
        version = api_get(HELLO_URL)
        (args.cache / "version.json").write_text(
            json.dumps(version, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    print(f"Kullui database version {version.get('database_version', 'unknown')}")
    build(entries, args.output, args.audit, args.cdial)


if __name__ == "__main__":
    main()
