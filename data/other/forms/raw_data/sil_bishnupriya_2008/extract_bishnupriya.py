#!/usr/bin/env python3
"""Extract the six Bishnupriya wordlists in SIL ESR 2008-003.

The publisher PDF renders its legacy phonetic font correctly but does not expose ordinary
Unicode text.  A public Slideshare copy preserves the PDF text layer as fixed-layout transcript
text and page rasters.  This extractor verifies the 18-page transcript fingerprint, recovers the
14 PUA glyphs against the rendered source pages, restores superscript aspiration from its layout,
and expands no site codes.  Expansion belongs to the installer so the audit retains the printed
response records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
DEFAULT_HTML = Path("/tmp/bishnupriya-slideshare.html")
DEFAULT_OUTPUT = HERE / "wordlists.tsv"
MAP_FILE = HERE / "slideshare_pua_used.tsv"
IMAGE_MANIFEST = HERE / "source_page_images.tsv"
IMAGE_DIR = WORKSPACE / "tmp/pdfs/sil-surveys/bishnupriya-slides"

TRANSCRIPT_SHA256 = "1c42bad6a4ee278b4056397f3c5db960b091d5e1df749ff021137a0398aeac8b"
SPLITS = {
    35: 46, 36: 43, 37: 48, 38: 43, 39: 38, 40: 43, 41: 40, 42: 35,
    43: 42, 44: 49, 45: 51, 46: 45, 47: 42, 48: 40, 49: 34, 50: 45,
    51: 37, 52: 33,
}
EMPTY_ITEMS = {194, 218, 221, 222, 258, 259, 301, 303, 306}
# Fixed-layout character positions are only approximate because the legacy glyphs have
# proportional widths.  These are every response having more whitespace runs than superscript
# h markers; each selected run was checked on the pinned source-page raster.  The other response
# spaces are ordinary word boundaries.
ASPIRATION_RUN_OVERRIDES = {
    (37, "dʒuɡiɾi b at"): (1,), (37, "dukuɾi b at"): (1,),
    (117, "t ɛŋor pata"): (0,), (136, "bɔɽo b ai̯"): (1,),
    (137, "dʒɛt i bonok"): (0,), (138, "tʃ to b ai̯"): (0, 2),
    (139, "tʃ oto bon"): (0,), (143, "baɽi / ɡ ɔr"): (2,),
    (180, "lat i maɾa"): (0,), (184, "ɔpɛkk a kɔɾa"): (0,),
    (187, "ʃidd o kɔɾa"): (0,), (189, "pani k awa"): (1,),
    (196, "b ulɛ dʒawa"): (0,), (198, "ʃɔpno dæk a"): (1,),
    (198, "hɔpon dɛk ani"): (1,), (205, "d akka dɛwa"): (0,),
    (209, "p uti bunani"): (0,), (210, "p uti hina"): (0,),
    (224, "ɡ ɾina kɔɾa"): (0,), (267, "d iɾɛ d iɾɛ"): (0, 2),
    (269, "k ei̯ k ei̯"): (0, 2),
}
FIELDS = [
    "Item", "Gloss", "Similarity_Group", "Raw_Form", "Form", "Site_Codes",
    "Printed_Page", "Column", "Response", "Aspiration_Markers", "Review",
]


def load_map() -> tuple[dict[str, str], dict[str, int]]:
    mapping = {}
    expected = {}
    with MAP_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            char = chr(int(row["Codepoint"][2:], 16))
            mapping[char] = row["Glyph"].replace("◌", "")
            expected[char] = int(row["Occurrences"])
    return mapping, expected


def transcript_pages(html: Path) -> list[str]:
    text = html.read_text(encoding="utf-8")
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', text)
    if not match:
        raise AssertionError("Slideshare __NEXT_DATA__ transcript is missing")
    data = json.loads(match.group(1))
    pages = data["props"]["pageProps"]["slideshow"]["transcript"][34:52]
    payload = "\n\f\n".join(pages).encode()
    if hashlib.sha256(payload).hexdigest() != TRANSCRIPT_SHA256:
        raise AssertionError("wordlist transcript fingerprint drift")
    return pages


def verify_images() -> None:
    if not IMAGE_DIR.exists():
        return
    with IMAGE_MANIFEST.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            path = IMAGE_DIR / f"page-{row['Printed_Page']}.jpg"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["SHA256"] or path.stat().st_size != int(row["Bytes"]):
                raise AssertionError(f"source page-image drift: {path}")


def is_heading(text: str, page: int, column: int) -> re.Match[str] | None:
    match = re.match(r"^(\d{1,3})\s+(.+)$", text)
    if not match or "[" in text:
        return None
    number = int(match.group(1))
    if number > 4 or (page == 35 and column == 1 and number <= 7):
        return match
    if page == 35 and column == 2 and 8 <= number <= 15:
        return match
    return None


def choose_aspiration_runs(
    item: int, form: str, marker_x: list[int], form_start: int
) -> set[int]:
    runs = list(re.finditer(r" +", form))
    count = min(len(marker_x), len(runs))
    if not count:
        return set()
    override = ASPIRATION_RUN_OVERRIDES.get((item, form))
    if override is not None:
        if len(override) != count or any(index >= len(runs) for index in override):
            raise AssertionError(f"aspiration override drift at item {item}: {form!r}")
        return set(override)
    best = None
    for indexes in itertools.combinations(range(len(runs)), count):
        cost = sum(
            abs(marker_x[pos] - (form_start + runs[index].start()))
            for pos, index in enumerate(indexes)
        )
        candidate = (cost, indexes)
        if best is None or candidate < best:
            best = candidate
    return set(best[1])


def restore_aspiration(item: int, form: str, marker_x: list[int], form_start: int) -> str:
    runs = list(re.finditer(r" +", form))
    selected = choose_aspiration_runs(item, form, marker_x, form_start)
    pieces = []
    cursor = 0
    for index, run in enumerate(runs):
        pieces.append(form[cursor:run.start()])
        if index in selected:
            pieces.append("ʰ" + (" " if len(run.group()) > 1 else ""))
        else:
            pieces.append(" ")
        cursor = run.end()
    pieces.append(form[cursor:])
    pieces.append("ʰ" * (len(marker_x) - len(selected)))
    return "".join(pieces)


def parse(html: Path) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    pages = transcript_pages(html)
    verify_images()
    mapping, expected_counts = load_map()
    observed = Counter(char for page in pages for char in page if char in mapping)
    if observed != Counter(expected_counts):
        raise AssertionError(f"PUA glyph census drift: {observed}")
    unknown = sorted({char for page in pages for char in page if 0xE000 <= ord(char) <= 0xF8FF} - mapping.keys())
    if unknown:
        raise AssertionError(f"unmapped PUA glyphs: {[f'U+{ord(c):04X}' for c in unknown]}")

    blocks: dict[int, dict] = {}
    headings = []
    for page, transcript in zip(range(35, 53), pages):
        lines = transcript.splitlines()
        first_line = 6 if page == 35 else 4
        for column in (1, 2):
            item = None
            for line_number, line in enumerate(lines[first_line:], start=first_line):
                text = (line[:SPLITS[page]] if column == 1 else line[SPLITS[page]:]).rstrip()
                stripped = text.strip()
                if not stripped:
                    continue
                heading = is_heading(stripped, page, column)
                if heading:
                    item = int(heading.group(1))
                    headings.append(item)
                    blocks[item] = {
                        "gloss": heading.group(2).strip(), "page": page,
                        "column": column, "lines": [],
                    }
                    continue
                if stripped == "B.3. Wordlists":
                    continue
                if item is None:
                    raise AssertionError(f"orphan line p.{page} col.{column}: {text!r}")
                blocks[item]["lines"].append((line_number, text))

    if headings != list(range(1, 308)):
        raise AssertionError(f"item topology drift: {headings}")

    rows = []
    empty = []
    for item, block in blocks.items():
        pending_group = None
        pending_form = None
        pending_markers: list[int] = []
        response = 0
        for line_number, text in block["lines"]:
            stripped = text.strip()
            if re.fullmatch(r"h(?:\s+h)*", stripped):
                pending_markers.extend(i for i, char in enumerate(text) if char == "h")
                continue
            if stripped.startswith("--"):
                continue
            if re.fullmatch(r"\d", stripped):
                pending_group = int(stripped)
                continue
            if "[" not in stripped:
                pending_form = (line_number, stripped, text.index(stripped))
                continue

            bracket = re.search(r"\[\s*([0a-fo ]+)\s*\]\s*$", text)
            if not bracket:
                raise AssertionError(
                    f"unparsed response bracket p.{block['page']} item {item}: {text!r}"
                )
            before = text[:bracket.start()].rstrip()
            grouped = re.match(r"^\s*(\d+)\s*(.*)$", before)
            if grouped:
                group = int(grouped.group(1))
                raw_form = grouped.group(2).strip()
                form_start = before.find(raw_form) if raw_form else -1
            elif pending_group is not None:
                group = pending_group
                raw_form = before.strip()
                form_start = before.find(raw_form)
            else:
                raise AssertionError(f"response has no similarity group: {text!r}")
            if not raw_form and pending_form:
                _, raw_form, form_start = pending_form
            if not raw_form:
                raise AssertionError(f"response has no form: {text!r}")

            raw_codes = "".join(bracket.group(1).split())
            review = "source text-layer transcript; legacy glyphs checked against page image"
            if raw_codes == "o":
                raw_codes = "0"
                review += "; source prints lowercase o for the Bangla code 0"
            decoded_raw = "".join(mapping.get(char, char) for char in raw_form)
            aspirated = restore_aspiration(
                item, decoded_raw, sorted(pending_markers), form_start
            )
            decoded = unicodedata.normalize("NFC", aspirated)
            response += 1
            rows.append({
                "Item": item, "Gloss": block["gloss"], "Similarity_Group": group,
                "Raw_Form": raw_form, "Form": decoded, "Site_Codes": raw_codes,
                "Printed_Page": block["page"], "Column": block["column"],
                "Response": response, "Aspiration_Markers": len(pending_markers),
                "Review": review,
            })
            pending_group = None
            pending_form = None
            pending_markers = []
        if pending_group is not None or pending_form is not None or pending_markers:
            raise AssertionError(f"orphan continuation at item {item}")
        if response == 0:
            empty.append({
                "Item": item, "Gloss": block["gloss"], "Printed_Page": block["page"],
                "Column": block["column"], "Reason": "source prints heading but no responses",
            })

    if len(rows) != 746 or {row["Item"] for row in empty} != EMPTY_ITEMS:
        raise AssertionError(f"response topology drift: rows={len(rows)} empty={empty}")
    if sum(int(row["Aspiration_Markers"]) for row in rows) != 161:
        raise AssertionError("superscript-aspiration census drift")
    if any(set(str(row["Site_Codes"])) - set("0abcdef") for row in rows):
        raise AssertionError("unknown site code")
    return rows, empty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", nargs="?", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, empty = parse(args.html)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    expanded = sum(len(str(row["Site_Codes"])) for row in rows)
    print(f"printed_responses={len(rows)} expanded={expanded} empty_prompts={len(empty)}")


if __name__ == "__main__":
    main()
