#!/usr/bin/env python3
"""Audit Mayrhofer's KEWA article images and attach scans to conservative CDIAL matches.

The samskrtam.ru edition exposes a machine-readable index (stable integer ID, volume/page,
Devanagari and romanised heads) and one tightly cropped image for each of 9,587 articles.  The
article body itself is image-only.  This importer pins the site's 2021 version-1.0 index, caches
and checks every image, runs a fixed Tesseract pass, and records the complete raw OCR in a
per-article audit.

KEWA contributes source commentary, not new lexical attestations or graph edges.  Articles are
therefore installed through ``data/other/entry_texts``.  A match requires an exact Sanskrit head
after conservative notation/inflection normalisation.  Accent is used first; an accent-neutral
fallback is admitted only when unique.  Ambiguous CDIAL homographs and unmatched articles remain
audit-only.  When several main-dictionary KEWA articles compete for the same CDIAL head, only a
sole accented match may win; otherwise every competitor stays uninstalled.  OCR is never used as
matching evidence.  Supplement/correction articles may coexist with the matched main article.

Installed blocks embed the exact per-article source scan and link to its stable article anchor.
OCR never enters the database: it is retained only in the audit to document extraction quality.
The authoritative index head, raw OCR, image checksum, exact locator, candidate evidence, and
review state remain in the audit.

Run from the data repository root::

    uv run python data/other/forms/raw_data/kewa.py --refresh --install

Rebuild the sidecar from the checked-in audit without the network or image cache::

    uv run python data/other/forms/raw_data/kewa.py --offline --install
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image


SOURCE_ID = "mayrhofer-kewa"
INDEX_URL = "https://samskrtam.ru/sanskrit-lexicon/KEWA/"
SOURCE_VERSION = "2021 version 1.0"
SNAPSHOT_DATE = "2026-08-18"
PINNED_INDEX_SHA256 = "d0a3127ba237149713b706da0ce5b4380a1b8077ead07f854169113e4e5da234"
EXPECTED_ARTICLES = 9587
SAMPLE_SEED = 1953
REVIEW_SAMPLE_IDS = (
    313, 810, 1152, 1390, 2030, 3517, 4409, 4774, 4830, 5741,
    6035, 6642, 6823, 7872, 8195, 8759, 8879, 9062, 9142, 9264,
)
OCR_LANG = "script/Latin"
OCR_PSM = "6"
USER_AGENT = "Jambu dictionary importer/1.0 (https://github.com/moli-mandala/data)"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
DEFAULT_CACHE = ROOT / "tmp/kewa-cache"
DEFAULT_AUDIT = RAW_DIR / "20260818-kewa-audit.csv"
DEFAULT_MANIFEST = RAW_DIR / "20260818-kewa-manifest.json"
DEFAULT_SAMPLE = RAW_DIR / "20260818-kewa-sample.csv"
DEFAULT_OUTPUT = ROOT / "data/other/entry_texts/20260818-kewa.csv"
PREVIEW_DIR = ROOT / "tmp/kewa-preview"
DEFAULT_CDIAL = ROOT / "data/cdial/params.csv"
DEFAULT_MERGES = ROOT / "cldf/merges.csv"

AUDIT_FIELDS = [
    "Snapshot_Date", "Index_SHA256", "Upstream_ID", "Entry_Key", "Stable_Anchor",
    "Volume", "Printed_Pages", "Index_Page_Label", "Locator_Note", "Is_Supplement",
    "Devanagari", "Accented_Heads",
    "IAST_Heads", "Image_URL", "Image_SHA256", "Image_Width", "Image_Height", "Image_Bytes",
    "OCR_Engine", "OCR_Config", "Raw_OCR", "OCR_Review", "English_Gloss_OCR",
    "Status", "Reason", "Match_Method", "Target_Candidates", "Accepted_Targets",
    "Candidate_Evidence", "Source_Citation", "Output_Blocks", "Unresolved_Characters",
]
TEXT_FIELDS = ["Form_ID", "Position", "Kind", "Format", "Content", "Source"]
SAMPLE_FIELDS = [
    "Seed", "Upstream_ID", "Entry_Key", "Stable_Anchor", "Volume", "Printed_Pages",
    "IAST_Heads", "Accepted_Targets", "Match_Method", "Image_SHA256", "OCR_Compared",
    "Head_Match_Compared", "OCR_Character_Perfect", "Material_Structural_Error",
    "Review_Notes",
]

ACCENT_MARKS = {
    "\u0300", "\u0301", "\u0302", "\u030f", "\u0311", "\u0340", "\u0341",
    "\u0951", "\u0952",
}
@dataclass(frozen=True)
class Article:
    upstream_id: int
    stable_anchor: str
    volume: str
    printed_pages: str
    index_page_label: str
    locator_note: str
    is_supplement: bool
    devanagari: str
    accented_heads: tuple[str, ...]
    iast_heads: tuple[str, ...]
    image_url: str

    @property
    def entry_key(self) -> str:
        return f"kewa:{self.upstream_id}"

    @property
    def citation(self) -> str:
        page = self.printed_pages.replace("-", "--")
        return f"{SOURCE_ID}[vol. {self.volume}, p. {page}, article {self.upstream_id}]"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_text(path: Path, content: str) -> None:
    atomic_bytes(path, content.encode("utf-8"))


def atomic_json(path: Path, packet: object) -> None:
    atomic_text(path, json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def download(url: str, attempts: int = 5) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data:
                raise ValueError(f"empty response from {url}")
            return data
        except (OSError, urllib.error.URLError, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_index(data: bytes) -> list[Article]:
    digest = sha256_bytes(data)
    if digest != PINNED_INDEX_SHA256:
        raise ValueError(
            f"KEWA index changed: expected {PINNED_INDEX_SHA256}, found {digest}; "
            "review and pin the new snapshot before ingesting it"
        )
    soup = BeautifulSoup(data.decode("utf-8"), "html.parser")
    articles: list[Article] = []
    for tr in soup.select("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) != 4 or not cells[0].get_text(strip=True).isdigit():
            continue
        upstream_id = int(cells[0].get_text(strip=True))
        page_text = cells[1].get_text(" ", strip=True)
        anchor = cells[2].find("a", id=True)
        image = cells[3].find("img", src=True)
        if not anchor or not image:
            raise ValueError(f"KEWA article {upstream_id} lacks an anchor or image")
        page_match = re.fullmatch(r"(I|II|III):\s*(\d+(?:-\d+)?)", page_text)
        if not page_match:
            # Three version-1.0 page ranges were converted into Russian month labels by a
            # spreadsheet.  The stable image anchor retains the exact start/end pages.
            anchor_match = re.fullmatch(r"([123])-(\d{3})(?:-(\d{2,3}))?-(\d{2})", anchor["id"])
            if not anchor_match:
                raise ValueError(f"bad KEWA locator for article {upstream_id}: {page_text!r}")
            volume = {"1": "I", "2": "II", "3": "III"}[anchor_match.group(1)]
            pages = str(int(anchor_match.group(2)))
            if anchor_match.group(3):
                pages += "-" + str(int(anchor_match.group(3)))
            locator_note = (
                f"site label {page_text!r} mechanically repaired from stable anchor {anchor['id']}"
            )
        else:
            volume, pages = page_match.groups()
            locator_note = ""
        iast_text = " ".join(
            node.get_text(" ", strip=True) for node in cells[2].select("p.iast")
        )
        iast_heads = tuple(value.strip() for value in re.findall(r"/([^/]*)/", iast_text))
        accented_text = " ".join(
            node.get_text(" ", strip=True)
            for node in cells[2].select("p.mhbld, p.mhitl")
            if node.get_text(" ", strip=True)
        )
        accented_heads = tuple(
            value.strip(" ,") for value in accented_text.split(",") if value.strip(" ,")
        )
        if not iast_heads or not accented_heads:
            raise ValueError(f"KEWA article {upstream_id} lacks a romanised head")
        articles.append(Article(
            upstream_id=upstream_id,
            stable_anchor=anchor["id"],
            volume=volume,
            printed_pages=pages,
            index_page_label=page_text,
            locator_note=locator_note,
            is_supplement="addcorr" in (cells[0].get("class") or []),
            devanagari=" | ".join(
                node.get_text(" ", strip=True) for node in cells[2].select("p.sa")
            ),
            accented_heads=accented_heads,
            iast_heads=iast_heads,
            image_url=urllib.parse.urljoin(INDEX_URL, image["src"]),
        ))
    if len(articles) != EXPECTED_ARTICLES:
        raise ValueError(f"expected {EXPECTED_ARTICLES} KEWA articles, found {len(articles)}")
    if [article.upstream_id for article in articles] != list(range(1, EXPECTED_ARTICLES + 1)):
        raise ValueError("KEWA integer article IDs are not a complete ordered sequence")
    if len({article.stable_anchor for article in articles}) != len(articles):
        raise ValueError("duplicate KEWA stable anchors")
    if len({article.image_url for article in articles}) != len(articles):
        raise ValueError("duplicate KEWA article image URLs")
    return articles


def image_path(cache: Path, article: Article) -> Path:
    return cache / "images" / f"{article.stable_anchor}.jpg"


def ocr_path(cache: Path, article: Article) -> Path:
    return cache / "ocr" / f"{article.stable_anchor}.txt"


def ocr_config_path(cache: Path, article: Article) -> Path:
    return cache / "ocr" / f"{article.stable_anchor}.psm"


def validate_image(data: bytes, article: Article) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if image.format != "JPEG" or width < 100 or height < 20:
            raise ValueError(
                f"bad KEWA image for article {article.upstream_id}: {image.format} {image.size}"
            )
    return width, height


def run_ocr(path: Path, psm: str) -> str:
    environment = dict(os.environ)
    environment["OMP_THREAD_LIMIT"] = "1"
    result = subprocess.run(
        [
            "tesseract", str(path), "stdout", "-l", OCR_LANG,
            "--psm", psm, "--dpi", "300",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    text = unicodedata.normalize("NFC", result.stdout.decode("utf-8")).strip()
    return text


def materialize_article(
    article: Article, cache: Path, *, refresh: bool, reocr: bool
) -> dict[str, object]:
    image_file = image_path(cache, article)
    if not image_file.exists() or refresh:
        data = download(article.image_url)
        validate_image(data, article)
        atomic_bytes(image_file, data)
    else:
        data = image_file.read_bytes()
    width, height = validate_image(data, article)
    text_file = ocr_path(cache, article)
    config_file = ocr_config_path(cache, article)
    if not text_file.exists() or reocr or refresh:
        used_psm = OCR_PSM
        text = run_ocr(image_file, used_psm)
        if not text:
            # Sparse cross-reference slips can be rejected as an empty page by PSM 6. PSM 11
            # recovers isolated text without the hallucinated speckle seen under single-line PSM 13.
            used_psm = "11"
            text = run_ocr(image_file, used_psm)
        if not text:
            raise ValueError(f"OCR emitted no text for {image_file} under PSM 6 and 11")
        atomic_text(text_file, text + "\n")
        atomic_text(config_file, used_psm + "\n")
    else:
        text = unicodedata.normalize("NFC", text_file.read_text(encoding="utf-8")).strip()
        used_psm = config_file.read_text(encoding="utf-8").strip() if config_file.exists() else OCR_PSM
    if not text:
        raise ValueError(f"cached OCR is empty for article {article.upstream_id}")
    return {
        "article": article,
        "image_sha256": sha256_bytes(data),
        "image_width": width,
        "image_height": height,
        "raw_ocr": text,
        "ocr_psm": used_psm,
        "image_bytes": len(data),
    }


def materialize_all(
    articles: list[Article], cache: Path, *, workers: int, refresh: bool, reocr: bool
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                materialize_article, article, cache, refresh=refresh, reocr=reocr
            ): article
            for article in articles
        }
        for completed, future in enumerate(as_completed(futures), 1):
            article = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                raise RuntimeError(f"failed to materialize KEWA article {article.upstream_id}") from error
            if completed % 100 == 0 or completed == len(articles):
                print(f"materialized {completed}/{len(articles)} KEWA articles", flush=True)
    return sorted(results, key=lambda item: item["article"].upstream_id)


def normalize_head(value: str, *, preserve_accent: bool) -> str:
    value = value.casefold().strip()
    value = value.replace("ṁ", "ṃ").replace("m̐", "ṃ").replace("r̥", "ṛ").replace("l̥", "ḷ")
    value = re.sub(r"^[*†‡√?]+", "", value)
    value = re.sub(r"[?¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", value)
    chars = []
    for char in unicodedata.normalize("NFD", value):
        if not preserve_accent and char in ACCENT_MARKS:
            continue
        chars.append(char)
    value = unicodedata.normalize("NFC", "".join(chars))
    return re.sub(r"[\s.·'’\-–—_{}\[\]()]", "", value)


def optional_variants(value: str) -> set[str]:
    variants = {value}
    while True:
        expanded = set(variants)
        for candidate in variants:
            match = re.search(r"\(([^()]*)\)", candidate)
            if match:
                expanded.add(candidate[:match.start()] + candidate[match.end():])
                expanded.add(candidate[:match.start()] + match.group(1) + candidate[match.end():])
        if expanded == variants:
            return variants
        variants = expanded


def head_keys(value: str, *, preserve_accent: bool, source_inflection: bool) -> set[str]:
    output: set[str] = set()
    for variant in optional_variants(value):
        key = normalize_head(variant, preserve_accent=preserve_accent)
        if not key:
            continue
        output.add(key)
        if source_inflection and key.endswith("ḥ"):
            output.add(key[:-1])
        if source_inflection and key.endswith("am"):
            output.add(key[:-1])
    return {value for value in output if value}


def load_merges(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        direct = {
            row["Addendum_ID"].strip(): row["Main_ID"].strip()
            for row in csv.DictReader(handle)
            if row.get("Addendum_ID", "").strip() and row.get("Main_ID", "").strip()
        }
    output: dict[str, str] = {}
    for source in direct:
        target = direct[source]
        seen = {source}
        while target in direct:
            if target in seen:
                raise ValueError(f"cyclic CDIAL merge involving {source}")
            seen.add(target)
            target = direct[target]
        output[source] = target
    return output


def cdial_indexes(
    params_path: Path, merges_path: Path
) -> tuple[
    dict[str, list[tuple[str, str]]],
    dict[str, list[tuple[str, str]]],
    dict[str, str],
    set[str],
]:
    accented: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unaccented: dict[str, list[tuple[str, str]]] = defaultdict(list)
    descriptions: dict[str, str] = {}
    valid_ids: set[str] = set()
    merges = load_merges(merges_path)
    with params_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            source_id, heads, _, description = row[:4]
            target = merges.get(source_id, source_id)
            valid_ids.update((source_id, target))
            if target not in descriptions or source_id == target:
                descriptions[target] = description
            for head in re.split(r"\s*,\s*", heads):
                pair = (target, head)
                for key in head_keys(head, preserve_accent=True, source_inflection=False):
                    if pair not in accented[key]:
                        accented[key].append(pair)
                for key in head_keys(head, preserve_accent=False, source_inflection=False):
                    if pair not in unaccented[key]:
                        unaccented[key].append(pair)
    return dict(accented), dict(unaccented), descriptions, valid_ids


def candidate_sets(
    heads: tuple[str, ...],
    index: dict[str, list[tuple[str, str]]],
    *,
    preserve_accent: bool,
) -> list[tuple[str, set[str], list[tuple[str, str]]]]:
    output = []
    for head in heads:
        pairs = {
            pair
            for key in head_keys(head, preserve_accent=preserve_accent, source_inflection=True)
            for pair in index.get(key, [])
        }
        output.append((head, {pair[0] for pair in pairs}, sorted(pairs)))
    return output


def english_gloss(raw_ocr: str) -> str:
    first_paragraph = re.split(r"\n\s*\n", raw_ocr, maxsplit=1)[0]
    first_paragraph = re.sub(r"\s*\n\s*", " ", first_paragraph)
    if "/" not in first_paragraph:
        return ""
    gloss = first_paragraph.split("/", 1)[1]
    gloss = re.split(r"\s(?:=|\[|<|>)\s?", gloss, maxsplit=1)[0]
    return gloss.strip(" ,;.")


def match_records(
    materialized: list[dict[str, object]],
    cdial_path: Path,
    merges_path: Path,
) -> list[dict[str, object]]:
    accented, unaccented, _, _ = cdial_indexes(cdial_path, merges_path)
    records: list[dict[str, object]] = []
    for item in materialized:
        article: Article = item["article"]
        accented_sets = candidate_sets(
            article.accented_heads, accented, preserve_accent=True
        )
        if any(targets for _, targets, _ in accented_sets):
            method = "exact accented head"
            matches = accented_sets
        else:
            method = "unique exact accent-normalized head"
            matches = candidate_sets(article.iast_heads, unaccented, preserve_accent=False)
        matched = [(head, targets, pairs) for head, targets, pairs in matches if targets]
        candidates = sorted({target for _, targets, _ in matched for target in targets})
        ambiguous_head = any(len(targets) > 1 for _, targets, _ in matched)
        if candidates and not ambiguous_head:
            accepted = list(candidates)
            reason = (
                f"{method}; every matched printed head has one canonical CDIAL target"
                if len(candidates) > 1
                else method
            )
        elif len(candidates) == 1:
            accepted = list(candidates)
            reason = method
        elif candidates:
            accepted = []
            reason = "non-unique CDIAL homograph candidates"
        else:
            accepted = []
            reason = "no exact CDIAL head candidate"
        evidence = "; ".join(
            f"{head!r} -> " + (
                ", ".join(f"{target} ({cdial_head})" for target, cdial_head in pairs) or "-"
            )
            for head, _, pairs in matches
        )
        gloss = english_gloss(item["raw_ocr"])
        records.append({
            **item,
            "match_method": method if candidates else "",
            "candidates": candidates,
            "accepted": accepted,
            "reason": reason,
            "evidence": evidence,
            "english_gloss": gloss,
            "conflicts": [],
        })

    # A supplement is explicitly additional prose.  Main-dictionary homographs, however, must
    # not all pile onto one CDIAL sense merely because the head spelling is identical.
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        article: Article = record["article"]
        if article.is_supplement:
            continue
        for target in record["accepted"]:
            by_target[target].append(record)
    for target, competitors in by_target.items():
        if len(competitors) < 2:
            continue
        accented_competitors = [
            record for record in competitors if record["match_method"] == "exact accented head"
        ]
        winner = accented_competitors[0] if len(accented_competitors) == 1 else None
        decision = (
            f"CDIAL {target} collision resolved in favor of the sole accented index match"
            if winner else f"CDIAL {target} has unresolved KEWA sense competitors"
        )
        for record in competitors:
            record["conflicts"].append(decision)
            if record is not winner:
                record["accepted"] = [value for value in record["accepted"] if value != target]

    return records


def audit_rows(records: list[dict[str, object]], index_sha256: str) -> list[dict[str, str]]:
    version = subprocess.run(
        ["tesseract", "--version"], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.splitlines()[0]
    rows = []
    for record in records:
        article: Article = record["article"]
        accepted = sorted(record["accepted"])
        candidates = sorted(record["candidates"])
        if accepted:
            status = "ingested"
        elif candidates:
            status = "ambiguous"
        else:
            status = "unmatched"
        reason = record["reason"]
        if record["conflicts"]:
            reason += "; " + "; ".join(dict.fromkeys(record["conflicts"]))
        raw_ocr = record["raw_ocr"]
        unresolved = "�" if "�" in raw_ocr else ""
        rows.append({
            "Snapshot_Date": SNAPSHOT_DATE,
            "Index_SHA256": index_sha256,
            "Upstream_ID": str(article.upstream_id),
            "Entry_Key": article.entry_key,
            "Stable_Anchor": article.stable_anchor,
            "Volume": article.volume,
            "Printed_Pages": article.printed_pages,
            "Index_Page_Label": article.index_page_label,
            "Locator_Note": article.locator_note,
            "Is_Supplement": "yes" if article.is_supplement else "no",
            "Devanagari": article.devanagari,
            "Accented_Heads": " | ".join(article.accented_heads),
            "IAST_Heads": " | ".join(article.iast_heads),
            "Image_URL": article.image_url,
            "Image_SHA256": record["image_sha256"],
            "Image_Width": str(record["image_width"]),
            "Image_Height": str(record["image_height"]),
            "Image_Bytes": str(record["image_bytes"]),
            "OCR_Engine": version,
            "OCR_Config": f"-l {OCR_LANG} --psm {record['ocr_psm']} --dpi 300; NFC",
            "Raw_OCR": raw_ocr,
            "OCR_Review": "unreviewed_structurally_accepted",
            "English_Gloss_OCR": record["english_gloss"],
            "Status": status,
            "Reason": reason,
            "Match_Method": record["match_method"],
            "Target_Candidates": " | ".join(candidates),
            "Accepted_Targets": " | ".join(accepted),
            "Candidate_Evidence": record["evidence"],
            "Source_Citation": article.citation,
            "Output_Blocks": str(len(accepted)),
            "Unresolved_Characters": unresolved,
        })
    return rows


def text_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in rows:
        if row["Status"] != "ingested":
            continue
        targets = [value.strip() for value in row["Accepted_Targets"].split("|") if value.strip()]
        for target in targets:
            key = (target, row["Upstream_ID"])
            if key in seen:
                raise ValueError(f"duplicate KEWA text block {key}")
            seen.add(key)
            article_url = html.escape(f"{INDEX_URL}#{row['Stable_Anchor']}", quote=True)
            image_url = html.escape(row["Image_URL"], quote=True)
            alt = html.escape("Scanned KEWA article for " + row["IAST_Heads"], quote=True)
            output.append({
                "Form_ID": target,
                "Position": str(300000 + int(row["Upstream_ID"])),
                "Kind": "etymology",
                "Format": "html",
                "Content": (
                    '<figure class="source-scan">'
                    f'<a href="{article_url}">'
                    f'<img src="{image_url}" alt="{alt}" loading="lazy" '
                    'decoding="async"></a>'
                    '</figure>'
                ),
                "Source": row["Source_Citation"],
            })
    return sorted(output, key=lambda row: (row["Form_ID"], int(row["Position"])))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_audit(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    missing = set(AUDIT_FIELDS) - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"KEWA audit missing fields {sorted(missing)}")
    if len(rows) != EXPECTED_ARTICLES:
        raise ValueError(f"expected {EXPECTED_ARTICLES} audited KEWA articles, found {len(rows)}")
    if {row["Index_SHA256"] for row in rows} != {PINNED_INDEX_SHA256}:
        raise ValueError("KEWA audit does not use the pinned index")
    if any(not row["Raw_OCR"].strip() for row in rows):
        raise ValueError("KEWA audit contains empty OCR")
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sample_rows(rows: list[dict[str, str]], existing_path: Path) -> list[dict[str, str]]:
    by_id = {int(row["Upstream_ID"]): row for row in rows}
    selected = [by_id[upstream_id] for upstream_id in REVIEW_SAMPLE_IDS]
    if any(row["Status"] != "ingested" for row in selected):
        raise ValueError("a reviewed KEWA sample article is no longer ingested")
    prior = {
        row["Upstream_ID"]: row
        for row in read_csv(existing_path)
    } if existing_path.exists() else {}
    output = []
    for row in sorted(selected, key=lambda value: int(value["Upstream_ID"])):
        old = prior.get(row["Upstream_ID"], {})
        output.append({
            "Seed": str(SAMPLE_SEED),
            "Upstream_ID": row["Upstream_ID"],
            "Entry_Key": row["Entry_Key"],
            "Stable_Anchor": row["Stable_Anchor"],
            "Volume": row["Volume"],
            "Printed_Pages": row["Printed_Pages"],
            "IAST_Heads": row["IAST_Heads"],
            "Accepted_Targets": row["Accepted_Targets"],
            "Match_Method": row["Match_Method"],
            "Image_SHA256": row["Image_SHA256"],
            "OCR_Compared": old.get("OCR_Compared", ""),
            "Head_Match_Compared": old.get("Head_Match_Compared", ""),
            "OCR_Character_Perfect": old.get("OCR_Character_Perfect", ""),
            "Material_Structural_Error": old.get("Material_Structural_Error", ""),
            "Review_Notes": old.get("Review_Notes", ""),
        })
    return output


def build_manifest(rows: list[dict[str, str]], blocks: list[dict[str, str]]) -> dict:
    statuses = Counter(row["Status"] for row in rows)
    volumes = Counter(row["Volume"] for row in rows)
    methods = Counter(row["Match_Method"] for row in rows if row["Match_Method"])
    return {
        "source": "Manfred Mayrhofer, Kurzgefasstes etymologisches Wörterbuch des Altindischen",
        "source_url": INDEX_URL,
        "site_version": SOURCE_VERSION,
        "snapshot_date": SNAPSHOT_DATE,
        "index_sha256": PINNED_INDEX_SHA256,
        "license": (
            "Reuse terms not stated; the site says scanning was done with the author's permission "
            "and marks the web edition copyright 2021 version 1.0"
        ),
        "source_articles": len(rows),
        "volume_counts": dict(sorted(volumes.items())),
        "supplement_articles": sum(row["Is_Supplement"] == "yes" for row in rows),
        "repaired_index_page_labels": sum(bool(row["Locator_Note"]) for row in rows),
        "audit_status_counts": dict(sorted(statuses.items())),
        "match_method_counts": dict(sorted(methods.items())),
        "installed_text_blocks": len(blocks),
        "accepted_target_count": len({row["Form_ID"] for row in blocks}),
        "image_bytes": sum(int(row["Image_Bytes"]) for row in rows),
        "ocr_characters": sum(len(row["Raw_OCR"]) for row in rows),
        "ocr_engine": rows[0]["OCR_Engine"] if rows else "",
        "ocr_config_counts": dict(sorted(Counter(row["OCR_Config"] for row in rows).items())),
        "reader_display": "exact per-article source scan linked to its stable article anchor",
        "ocr_database_policy": "audit only; OCR text is not installed in the database",
        "match_evidence": "authoritative index heads only; OCR is not used for matching",
        "sound_profile": "not applicable: the source contributes entry prose, not forms",
        "languages_dialects": "not applicable: no lexical attestation is introduced",
        "graph_relations": "none: head matching only selects owners for attributed prose blocks",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cdial", type=Path, default=DEFAULT_CDIAL)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--refresh", action="store_true", help="download the pinned index/images")
    parser.add_argument("--reocr", action="store_true", help="rerun OCR even when cached")
    parser.add_argument("--offline", action="store_true", help="rebuild from the checked-in audit")
    parser.add_argument("--install", action="store_true", help="replace canonical snapshot outputs")
    args = parser.parse_args()
    if args.offline and (args.refresh or args.reocr):
        parser.error("--offline cannot be combined with --refresh or --reocr")
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.install:
        audit_path = args.audit or DEFAULT_AUDIT
        manifest_path = args.manifest or DEFAULT_MANIFEST
        sample_path = args.sample or DEFAULT_SAMPLE
        output_path = args.output or DEFAULT_OUTPUT
    else:
        audit_path = args.audit or PREVIEW_DIR / DEFAULT_AUDIT.name
        manifest_path = args.manifest or PREVIEW_DIR / DEFAULT_MANIFEST.name
        sample_path = args.sample or PREVIEW_DIR / DEFAULT_SAMPLE.name
        output_path = args.output or PREVIEW_DIR / DEFAULT_OUTPUT.name

    if args.offline:
        source_audit = DEFAULT_AUDIT if audit_path != DEFAULT_AUDIT else audit_path
        rows = read_audit(source_audit)
    else:
        index_path = args.cache / "index.html"
        if args.refresh or not index_path.exists():
            index_data = download(INDEX_URL)
            if sha256_bytes(index_data) != PINNED_INDEX_SHA256:
                parse_index(index_data)  # raises the review-oriented digest error
            atomic_bytes(index_path, index_data)
        index_data = index_path.read_bytes()
        articles = parse_index(index_data)
        materialized = materialize_all(
            articles, args.cache, workers=args.workers, refresh=args.refresh, reocr=args.reocr
        )
        records = match_records(materialized, args.cdial, args.merges)
        rows = audit_rows(records, sha256_bytes(index_data))

    blocks = text_rows(rows)
    samples = sample_rows(rows, DEFAULT_SAMPLE)
    manifest = build_manifest(rows, blocks)
    write_csv(audit_path, AUDIT_FIELDS, rows)
    write_csv(output_path, TEXT_FIELDS, blocks)
    write_csv(sample_path, SAMPLE_FIELDS, samples)
    atomic_json(manifest_path, manifest)
    print(
        f"audited {len(rows)} KEWA articles; wrote {len(blocks)} text blocks on "
        f"{manifest['accepted_target_count']} CDIAL entries; statuses: "
        f"{manifest['audit_status_counts']}; output: {output_path}"
    )


if __name__ == "__main__":
    main()
