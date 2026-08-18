#!/usr/bin/env python3
"""Snapshot NurED and attach revisioned article prose to Jambu CDIAL/PNur entries.

NurED is a live MediaWiki dictionary.  Namespace-0 page IDs are stable while titles and content
can change, so this importer inventories every page, stores exact revision IDs and raw wikitext in
the audit, and caches the rendered response for each ``(page ID, revision ID)`` pair.  Only the two
article classes that have a conservative Jambu home are installed:

* ``Category:Middle Indo-Aryan loanwords`` -> a CDIAL head;
* ``Category:Proto-Nuristani`` -> an existing Strand Proto-Nuristani head.

Printed Turner IDs take precedence for CDIAL articles.  Otherwise matching is exact after removing
accent marks and source notation, while retaining segmental and length distinctions.  PNur articles
are routed through Jambu's reviewed CDIAL/PNur correspondences or an exact reconstructed-head match.
Ambiguous and unmatched pages remain audit-only.  Reviewed exceptional mappings live in the small
``20260818-nured-org-targets.csv`` overlay rather than in parser code.

The installed artifact is an entry-text sidecar, not a forms wordlist.  It preserves the complete
rendered source article as attributed HTML on the corresponding headword and therefore introduces
no new transcription or graph claim.

Preview the live source (run from the data repository root)::

    uv run python data/other/forms/raw_data/nured_org.py --refresh

Install a reviewed snapshot::

    uv run python data/other/forms/raw_data/nured_org.py --refresh --install

Rebuild from the checked-in audit without network access::

    uv run python data/other/forms/raw_data/nured_org.py --offline --install
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, Comment


API_URL = "https://nured.org/w/api.php"
ARTICLE_BASE = "https://nured.org/wiki/"
USER_AGENT = "Jambu dictionary importer/1.0 (https://github.com/moli-mandala/data)"
SOURCE_ID = "nured"
SNAPSHOT_DATE = datetime.now(timezone.utc).date().isoformat()

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
DEFAULT_CACHE = ROOT / "tmp/nured-org-cache"
DEFAULT_AUDIT = RAW_DIR / "20260818-nured-org-audit.csv"
DEFAULT_MANIFEST = RAW_DIR / "20260818-nured-org-manifest.json"
DEFAULT_TARGETS = RAW_DIR / "20260818-nured-org-targets.csv"
DEFAULT_OUTPUT = ROOT / "data/other/entry_texts/20260818-nured-org.csv"
PREVIEW_DIR = ROOT / "tmp/nured-org-preview"
DEFAULT_CDIAL = ROOT / "data/cdial/params.csv"
DEFAULT_MERGES = ROOT / "cldf/merges.csv"
DEFAULT_STRAND = ROOT / "data/other/params/strand3.csv"
DEFAULT_COGNATES = ROOT / "data/nuristani_cognates.csv"

AUDIT_FIELDS = [
    "Snapshot_Date", "Page_ID", "Revision_ID", "Revision_Timestamp", "Entry_Key", "Title",
    "Article_Type", "Categories", "Status", "Reason", "Accepted_Targets",
    "Target_Candidates", "Candidate_Evidence", "Source_Citation", "Article_URL",
    "Wikitext_SHA256", "Raw_Wikitext", "Rendered_HTML", "Output_Blocks",
]
TEXT_FIELDS = ["Form_ID", "Position", "Kind", "Format", "Content", "Source"]

ARTICLE_CATEGORIES = {
    "Middle Indo-Aryan loanwords": "cdial",
    "Proto-Nuristani": "pnur",
}

# Vedic accents and generic combining stress marks are irrelevant to exact dictionary-head
# comparison.  Macron/breve, underdots, nasality, and all other segmental marks are retained.
ACCENTS = {
    "\u0300", "\u0301", "\u0302", "\u0304",  # macron is removed from this set below
    "\u030f", "\u0311", "\u0340", "\u0341", "\u0951", "\u0952",
}
ACCENTS.remove("\u0304")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, packet: object) -> None:
    atomic_text(path, json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def api_get(params: dict[str, str], attempts: int = 5) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    url = f"{API_URL}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                packet = json.load(response)
            if "error" in packet:
                raise RuntimeError(f"NurED API error: {packet['error']!r}")
            return packet
        except urllib.error.HTTPError as error:
            # Miraheze's Cloudflare layer intermittently rejects urllib's TLS fingerprint for
            # ``action=parse`` while serving the identical public request to curl.  The argv-only
            # fallback is portable on CI/macOS and avoids any shell interpolation.
            if error.code == 403:
                raw = subprocess.check_output(
                    [
                        "curl", "-L", "--fail", "--max-time", "60", "-sS",
                        "-A", USER_AGENT, url,
                    ],
                    timeout=75,
                )
                packet = json.loads(raw)
                if "error" in packet:
                    raise RuntimeError(f"NurED API error: {packet['error']!r}")
                return packet
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def all_namespace_pages() -> list[dict]:
    """Enumerate namespace 0 deterministically, including hard redirects for completeness."""
    pages: list[dict] = []
    # MediaWiki does not include a redirect flag in ordinary ``allpages`` output. Query the two
    # disjoint sets explicitly so redirects are neither fetched as articles nor miscounted.
    for filter_name in ("nonredirects", "redirects"):
        continuation = ""
        while True:
            params = {
                "action": "query", "list": "allpages", "apnamespace": "0", "aplimit": "max",
                "apfilterredir": filter_name,
            }
            if continuation:
                params["apcontinue"] = continuation
            packet = api_get(params)
            chunk = packet["query"]["allpages"]
            if filter_name == "redirects":
                chunk = [{**page, "redirect": True} for page in chunk]
            pages.extend(chunk)
            continuation = packet.get("continue", {}).get("apcontinue", "")
            if not continuation:
                break
    return sorted(pages, key=lambda page: int(page["pageid"]))


def latest_pages(catalog: list[dict]) -> list[dict]:
    """Fetch the current raw wikitext/category state of every non-hard-redirect page."""
    page_ids = [str(page["pageid"]) for page in catalog if not page.get("redirect")]
    pages: list[dict] = []
    for start in range(0, len(page_ids), 50):
        packet = api_get({
            "action": "query",
            "pageids": "|".join(page_ids[start : start + 50]),
            "prop": "revisions|categories",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "cllimit": "max",
        })
        for page in packet["query"]["pages"]:
            revision = page["revisions"][0]
            wikitext = revision["slots"]["main"].get("content", "")
            # ``prop=categories`` paginates across the whole 50-page batch.  The literal category
            # declarations are the authoritative snapshot and ensure late pages are not lost to a
            # module continuation token.
            literal_categories = re.findall(
                r"\[\[\s*Category\s*:\s*([^\]|]+)", wikitext, flags=re.I
            )
            pages.append({
                "pageid": int(page["pageid"]),
                "title": page["title"],
                "revid": int(revision["revid"]),
                "timestamp": revision["timestamp"],
                "wikitext": wikitext,
                "categories": sorted(set(literal_categories) | {
                    category["title"].removeprefix("Category:")
                    for category in page.get("categories", [])
                }),
            })
    return sorted(pages, key=lambda page: page["pageid"])


def cache_file(cache: Path, page: dict) -> Path:
    return cache / f"page-{page['pageid']}-revision-{page['revid']}.json"


def rendered_packet(page: dict, cache: Path, offline: bool = False) -> dict:
    path = cache_file(cache, page)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"missing cached NurED render {path}")
    packet = api_get({
        "action": "parse",
        "oldid": str(page["revid"]),
        "prop": "text|sections|categories|displaytitle",
    })
    parsed = packet.get("parse", {})
    if int(parsed.get("revid", 0)) != page["revid"]:
        raise ValueError(f"NurED returned the wrong revision for page {page['pageid']}")
    atomic_json(path, packet)
    return packet


def article_type(categories: list[str]) -> str:
    kinds = {ARTICLE_CATEGORIES[category] for category in categories if category in ARTICLE_CATEGORIES}
    if len(kinds) > 1:
        raise ValueError(f"conflicting NurED article categories: {categories}")
    return next(iter(kinds), "")


def normalize_head(value: str) -> str:
    """Conservative comparison key: remove notation/accents, preserve actual sounds and length."""
    value = html.unescape(value or "").casefold().strip()
    value = value.replace("ṁ", "ṃ").replace("m̐", "ṃ").replace("r̥", "ṛ")
    value = re.sub(r"^[*†‡√?]+", "", value)
    value = re.sub(r"[?¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", value)
    chars = [char for char in unicodedata.normalize("NFD", value) if char not in ACCENTS]
    value = unicodedata.normalize("NFC", "".join(chars))
    return re.sub(r"[\s.·'’\-–—_{}\[\]()]", "", value)


def head_variants(value: str) -> set[str]:
    """Return exact keys for source parentheses meaning an explicitly optional sequence."""
    variants = {value}
    for match in list(re.finditer(r"\(([^()]*)\)", value)):
        variants |= {
            candidate[: match.start()] + optional + candidate[match.end() :]
            for candidate in list(variants)
            for optional in ("", match.group(1))
            if match.start() <= len(candidate)
        }
    return {normalize_head(value) for value in variants if normalize_head(value)}


def cdial_merges(path: Path) -> dict[str, str]:
    """Return transitive addendum-to-main redirects from the compiled CDIAL review table."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        direct = {
            row["Addendum_ID"].strip(): row["Main_ID"].strip()
            for row in csv.DictReader(handle)
            if row.get("Addendum_ID", "").strip() and row.get("Main_ID", "").strip()
        }
    result = {}
    for source in direct:
        target = direct[source]
        seen = {source}
        while target in direct:
            if target in seen:
                raise ValueError(f"cyclic CDIAL merge involving {source}")
            seen.add(target)
            target = direct[target]
        result[source] = target
    return result


def cdial_index(
    path: Path, merges_path: Path
) -> tuple[dict[str, list[tuple[str, str]]], set[str], dict[str, str]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    ids: set[str] = set()
    merges = cdial_merges(merges_path)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            source_id, heads = row[:2]
            entry_id = merges.get(source_id, source_id)
            ids.add(source_id)
            ids.add(entry_id)
            for head in re.split(r"\s*,\s*", heads):
                for key in head_variants(head):
                    pair = (entry_id, head)
                    if pair not in index[key]:
                        index[key].append(pair)
    return dict(index), ids, merges


def pnur_index(path: Path) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[1] != "PNur":
                continue
            entry_id, head = row[0], row[2]
            ids.add(entry_id)
            for key in head_variants(head):
                pair = (entry_id, head)
                if pair not in index[key]:
                    index[key].append(pair)
    return dict(index), ids


def pnur_by_cdial(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pnur, cdial = row["Proto_Nuristani_ID"], row["Indo_Aryan_ID"]
            if pnur not in result[cdial]:
                result[cdial].append(pnur)
    return {key: sorted(value) for key, value in result.items()}


def printed_turner_ids(wikitext: str, valid_ids: set[str]) -> list[str]:
    ids = re.findall(r"\bT\.\s*(\d+[a-z]?)\b", wikitext, flags=re.I)
    return list(dict.fromkeys(entry_id for entry_id in ids if entry_id in valid_ids))


def extract_section(wikitext: str, title: str) -> str:
    match = re.search(
        rf"^==\s*{re.escape(title)}\s*==\s*$([\s\S]*?)(?=^==[^=]|\Z)",
        wikitext,
        flags=re.M | re.I,
    )
    return match.group(1).strip() if match else ""


def first_positional_argument(arguments: str) -> str:
    """Read the first positional value while ignoring MediaWiki named arguments."""
    for value in arguments.split("|"):
        value = value.strip()
        if value and "=" not in value:
            return value
    return ""


def source_heads(wikitext: str) -> list[str]:
    section = extract_section(wikitext, "Etymology of Source")
    old_ia = re.search(r"'''Old Indo-Aryan'''([\s\S]*?)(?:\n\*|\Z)", section, flags=re.I)
    segment = old_ia.group(1) if old_ia else section
    values = []
    # Keep source order: the first attested OIA form is the etymon's primary source head, while
    # later forms are often derivational components or comparanda rather than attachment targets.
    pattern = re.compile(
        r"\{\{\s*form\s*\|([^{}]*)\}\}|<big>\s*([^<]+?)\s*</big>", flags=re.I
    )
    for match in pattern.finditer(segment):
        value = (
            first_positional_argument(match.group(1))
            if match.group(1) is not None
            else match.group(2).strip()
        )
        for alternative in re.split(r"\s*~\s*", value):
            alternative = alternative.strip()
            if normalize_head(alternative) and alternative not in values:
                values.append(alternative)
    return values


def source_lemma(wikitext: str) -> str:
    match = re.search(r"\{\{\s*lemma\s*\|([^{}]*)\}\}", wikitext, flags=re.I)
    return first_positional_argument(match.group(1)) if match else ""


def load_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        page_id = row.get("Page_ID", "").strip()
        if not page_id or page_id in result:
            raise ValueError(f"bad or duplicate NurED target override page ID {page_id!r}")
        result[page_id] = row
    return result


def match_target(
    page: dict,
    kind: str,
    cdial: dict[str, list[tuple[str, str]]],
    cdial_ids: set[str],
    pnur: dict[str, list[tuple[str, str]]],
    pnur_ids: set[str],
    correspondences: dict[str, list[str]],
    merges: dict[str, str],
    overrides: dict[str, dict[str, str]],
) -> tuple[list[str], list[str], str, str]:
    printed_raw = printed_turner_ids(page["wikitext"], cdial_ids)
    printed = list(dict.fromkeys(merges.get(value, value) for value in printed_raw))
    heads = source_heads(page["wikitext"])
    lemma = source_lemma(page["wikitext"])
    exact_by_head = [
        sorted({pair[0] for key in head_variants(head) for pair in cdial.get(key, [])})
        for head in heads
    ]
    exact_cdial = next((values for values in exact_by_head if values), [])
    exact_pnur = sorted({
        pair[0]
        for key in head_variants(lemma)
        for pair in pnur.get(key, [])
    })
    evidence = (
        f"printed Turner raw={printed_raw or '-'} canonical={printed or '-'}; "
        f"source heads={heads or '-'}; first matching CDIAL={exact_cdial or '-'}; "
        f"lemma={lemma or '-'}; exact PNur={exact_pnur or '-'}"
    )

    candidates: list[str]
    reason: str
    if kind == "cdial":
        printed_set, exact_set = set(printed), set(exact_cdial)
        if len(printed_set) == 1 and (not exact_set or printed_set == exact_set):
            candidates = sorted(printed_set)
            reason = "unique printed Turner ID"
        elif len(exact_set) == 1 and not printed_set:
            candidates = sorted(exact_set)
            reason = "unique exact accent-normalized Old Indo-Aryan head"
        else:
            candidates = sorted(printed_set | exact_set)
            reason = "conflicting or non-unique CDIAL evidence" if candidates else "no CDIAL evidence"
    else:
        via_cdial = sorted({pnur_id for cdial_id in printed for pnur_id in correspondences.get(cdial_id, [])})
        combined = sorted(set(via_cdial) | set(exact_pnur))
        if len(exact_pnur) == 1 and (not via_cdial or exact_pnur[0] in via_cdial):
            candidates = exact_pnur
            reason = "unique exact PNur head corroborated by reviewed correspondence"
        elif len(via_cdial) == 1:
            candidates = via_cdial
            reason = "unique reviewed CDIAL-to-PNur correspondence"
        elif len(combined) == 1:
            candidates = combined
            reason = "unique conservative PNur candidate"
        else:
            candidates = combined
            reason = "non-unique PNur evidence" if candidates else "no PNur evidence"

    override = overrides.get(str(page["pageid"]))
    if override:
        accepted = [value for value in override.get("Target_IDs", "").split("|") if value]
        valid = cdial_ids if kind == "cdial" else pnur_ids
        missing = sorted(set(accepted) - valid)
        if missing:
            raise ValueError(f"NurED override page {page['pageid']} has invalid {kind} targets {missing}")
        return accepted, candidates, evidence, override.get("Reason", "reviewed override")

    accepted = candidates if len(candidates) == 1 else []
    return accepted, candidates, evidence, reason


def absolute_source_url(title: str) -> str:
    return ARTICLE_BASE + urllib.parse.quote(title.replace(" ", "_"), safe="()*,~-._")


def revision_url(page: dict) -> str:
    query = urllib.parse.urlencode({"title": page["title"], "oldid": page["revid"]})
    return f"https://nured.org/w/index.php?{query}"


def sanitize_article(raw_html: str, page: dict) -> str:
    """Keep source HTML, remove MediaWiki controls/reports, and make links collision-safe."""
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for node in root.select(
        ".mw-editsection, meta, script, style, iframe, object, embed, form, input, button, "
        "textarea, select, option, link, base"
    ):
        node.decompose()
    for comment in root.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    prefix = f"nured-{page['pageid']}-"
    id_map = {}
    for node in root.find_all(id=True):
        old = node["id"]
        id_map[old] = prefix + old
        node["id"] = id_map[old]
    for node in root.find_all(href=True):
        href = node["href"]
        if href.startswith("#"):
            node["href"] = "#" + id_map.get(href[1:], prefix + href[1:])
        elif href.startswith("/"):
            node["href"] = urllib.parse.urljoin("https://nured.org", href)
        elif not urllib.parse.urlparse(href).scheme:
            node["href"] = urllib.parse.urljoin(absolute_source_url(page["title"]), href)
        elif urllib.parse.urlparse(href).scheme not in {"http", "https", "mailto"}:
            del node["href"]
    for node in root.find_all(src=True):
        src = urllib.parse.urljoin("https://nured.org", node["src"])
        if urllib.parse.urlparse(src).scheme in {"http", "https"}:
            node["src"] = src
        else:
            del node["src"]
    for node in root.find_all(True):
        for attribute in list(node.attrs):
            if attribute.casefold().startswith("on") or attribute.casefold() in {"style", "srcset"}:
                del node[attribute]

    body = "".join(str(child) for child in root.contents).strip()
    title = html.escape(page["title"])
    url = html.escape(revision_url(page), quote=True)
    return (
        f'<article class="nured-article" data-page-id="{page["pageid"]}" '
        f'data-revision-id="{page["revid"]}">'
        f'<h3>NurED article: <a href="{url}">{title}</a> '
        f'<small>(revision {page["revid"]})</small></h3>{body}</article>'
    )


def source_citation(page: dict) -> str:
    date = page["timestamp"].split("T", 1)[0]
    return f"{SOURCE_ID}[page {page['pageid']}, revision {page['revid']}, {date}]"


def retain_snapshot_date_when_unchanged(
    rows: list[dict[str, str]], manifest: dict, previous_path: Path = DEFAULT_MANIFEST
) -> None:
    """Avoid date-only weekly diffs when the live page/revision inventory is unchanged."""
    if not previous_path.exists():
        return
    try:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    stable_fields = (
        "pages_sha256", "namespace_0_pages", "hard_redirects_excluded",
        "nonredirect_pages_audited",
    )
    if not all(previous.get(field) == manifest.get(field) for field in stable_fields):
        return
    previous_date = previous.get("snapshot_date", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", previous_date):
        return
    manifest["snapshot_date"] = previous_date
    for row in rows:
        row["Snapshot_Date"] = previous_date


def audit_from_live(
    cache: Path,
    cdial_path: Path,
    merges_path: Path,
    strand_path: Path,
    cognates_path: Path,
    targets_path: Path,
) -> tuple[list[dict[str, str]], dict]:
    catalog = all_namespace_pages()
    live_pages = latest_pages(catalog)
    cdial, cdial_ids, merges = cdial_index(cdial_path, merges_path)
    pnur, pnur_ids = pnur_index(strand_path)
    correspondences = pnur_by_cdial(cognates_path)
    overrides = load_overrides(targets_path)

    rows: list[dict[str, str]] = []
    scoped = 0
    for page in live_pages:
        kind = article_type(page["categories"])
        rendered = ""
        accepted: list[str] = []
        candidates: list[str] = []
        evidence = ""
        if kind:
            scoped += 1
            accepted, candidates, evidence, reason = match_target(
                page, kind, cdial, cdial_ids, pnur, pnur_ids, correspondences, merges, overrides
            )
            packet = rendered_packet(page, cache)
            rendered = sanitize_article(packet["parse"]["text"], page)
            status = "ingested" if accepted else ("ambiguous" if candidates else "unmatched")
        else:
            status = "excluded"
            reason = "outside the CDIAL/PNur article scope"
        rows.append({
            "Snapshot_Date": SNAPSHOT_DATE,
            "Page_ID": str(page["pageid"]),
            "Revision_ID": str(page["revid"]),
            "Revision_Timestamp": page["timestamp"],
            "Entry_Key": f"nured:{page['pageid']}",
            "Title": page["title"],
            "Article_Type": kind,
            "Categories": " | ".join(page["categories"]),
            "Status": status,
            "Reason": reason,
            "Accepted_Targets": " | ".join(accepted),
            "Target_Candidates": " | ".join(candidates),
            "Candidate_Evidence": evidence,
            "Source_Citation": source_citation(page),
            "Article_URL": absolute_source_url(page["title"]),
            "Wikitext_SHA256": hashlib.sha256(page["wikitext"].encode()).hexdigest(),
            "Raw_Wikitext": page["wikitext"],
            "Rendered_HTML": rendered,
            "Output_Blocks": str(len(accepted)),
        })

    hard_redirects = sum(bool(page.get("redirect")) for page in catalog)
    article_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["Article_Type"]:
            article_counts[row["Article_Type"]] += 1
    manifest = {
        "source": "Nuristani Etymological Dictionary (NurED)",
        "source_url": "https://nured.org/wiki/Main_Page",
        "api_url": API_URL,
        "snapshot_date": SNAPSHOT_DATE,
        "license": "CC BY-SA 4.0",
        "namespace_0_pages": len(catalog),
        "hard_redirects_excluded": hard_redirects,
        "nonredirect_pages_audited": len(live_pages),
        "scoped_articles": scoped,
        "article_type_counts": dict(sorted(article_counts.items())),
        "max_revision_id": max(page["revid"] for page in live_pages),
        "max_revision_timestamp": max(page["timestamp"] for page in live_pages),
        "pages_sha256": hashlib.sha256(
            "\n".join(
                f"{page['pageid']}\t{page['revid']}\t{page['title']}"
                for page in live_pages
            ).encode()
        ).hexdigest(),
    }
    retain_snapshot_date_when_unchanged(rows, manifest)
    return rows, manifest


def read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(AUDIT_FIELDS) - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"NurED audit missing fields {sorted(missing)}")
    for row in rows:
        digest = hashlib.sha256(row["Raw_Wikitext"].encode()).hexdigest()
        if digest != row["Wikitext_SHA256"]:
            raise ValueError(f"NurED audit wikitext checksum mismatch for page {row['Page_ID']}")
    return rows


def text_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in audit_rows:
        if row["Status"] != "ingested":
            continue
        targets = [value.strip() for value in row["Accepted_Targets"].split("|") if value.strip()]
        if not row["Rendered_HTML"].strip():
            raise ValueError(f"ingested NurED page {row['Page_ID']} has no rendered HTML")
        for target in targets:
            key = (target, row["Page_ID"])
            if key in seen:
                raise ValueError(f"duplicate NurED text block key {key}")
            seen.add(key)
            output.append({
                "Form_ID": target,
                # A page ID is immutable, and this high range leaves room for hand-authored blocks.
                "Position": str(100000 + int(row["Page_ID"])),
                "Kind": "etymology",
                "Format": "html",
                "Content": row["Rendered_HTML"],
                "Source": row["Source_Citation"],
            })
    return sorted(output, key=lambda row: (row["Form_ID"], int(row["Position"])))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--cdial", type=Path, default=DEFAULT_CDIAL)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--strand", type=Path, default=DEFAULT_STRAND)
    parser.add_argument("--cognates", type=Path, default=DEFAULT_COGNATES)
    parser.add_argument("--refresh", action="store_true", help="query the current live wiki")
    parser.add_argument("--offline", action="store_true", help="rebuild from the checked-in audit")
    parser.add_argument("--install", action="store_true", help="replace canonical snapshot outputs")
    args = parser.parse_args()
    if args.refresh and args.offline:
        parser.error("--refresh and --offline are mutually exclusive")

    if args.install:
        audit_path = args.audit or DEFAULT_AUDIT
        manifest_path = args.manifest or DEFAULT_MANIFEST
        output_path = args.output or DEFAULT_OUTPUT
    else:
        audit_path = args.audit or PREVIEW_DIR / DEFAULT_AUDIT.name
        manifest_path = args.manifest or PREVIEW_DIR / DEFAULT_MANIFEST.name
        output_path = args.output or PREVIEW_DIR / DEFAULT_OUTPUT.name

    if args.offline:
        source_audit = DEFAULT_AUDIT if audit_path != DEFAULT_AUDIT else audit_path
        audit_rows = read_audit(source_audit)
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    else:
        audit_rows, manifest = audit_from_live(
            args.cache, args.cdial, args.merges, args.strand, args.cognates, args.targets
        )

    blocks = text_rows(audit_rows)
    counts: dict[str, int] = defaultdict(int)
    article_counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        counts[row["Status"]] += 1
        if row["Article_Type"]:
            article_counts[row["Article_Type"]] += 1
    manifest["nonredirect_pages_audited"] = len(audit_rows)
    manifest["scoped_articles"] = sum(article_counts.values())
    manifest["article_type_counts"] = dict(sorted(article_counts.items()))
    manifest["audit_status_counts"] = dict(sorted(counts.items()))
    manifest["installed_text_blocks"] = len(blocks)
    manifest["accepted_target_count"] = len({row["Form_ID"] for row in blocks})

    write_csv(audit_path, AUDIT_FIELDS, audit_rows)
    write_csv(output_path, TEXT_FIELDS, blocks)
    atomic_json(manifest_path, manifest)
    print(
        f"audited {len(audit_rows)} nonredirect pages; wrote {len(blocks)} entry-text blocks; "
        f"statuses: {dict(sorted(counts.items()))}; output: {output_path}"
    )


if __name__ == "__main__":
    main()
