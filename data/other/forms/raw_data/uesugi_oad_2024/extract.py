"""Layout extraction for the *Old Avestan Dictionary* text layer.

The PDF is born-digital (Adobe Acrobat, Times New Roman + Wingdings/Verdana symbol fonts). This
module turns each dictionary page into ordered *lines* of styled spans:

* columns are split at the page's horizontal midpoint (the two text columns start at x≈110/122
  and x≈304/317; nothing in the body crosses the gutter);
* a line is a cluster of base characters whose ``top`` lies within ``LINE_TOLERANCE`` points of
  the cluster's first character (body lines are ≈14 pt apart; raised accents, detached dots below
  and superscripts sit within ±3 pt);
* combining marks (Unicode ``Mn``) are positioned glyphs in this PDF, sometimes emitted out of
  stream order and sometimes rendered detached (``aš ̣a``, ``ā ̊``). Each is re-attached to the
  base character it overlaps horizontally on the nearest line, then the text is NFC-normalized;
* spaces are the PDF's own space glyphs plus a space wherever two base glyphs are separated by
  more than ``GAP`` points; a mark never receives a space before it;
* font classes: ``W3`` Wingdings 3 (the attestation bullet, extracted as ``9``), ``W`` Wingdings
  (the pointing hand, extracted as ``F``), ``V`` Verdana (``◊``), ``B`` bold, ``BI`` bold
  italic, ``I`` italic, ``R`` roman; characters smaller than 8 pt in the body are superscripts
  (homograph numbers, the corrected-reading ``+``) and are rewritten as superscript code points.

The running head (top < ``HEAD_LIMIT``) holds the printed page number and a guide word; it is
returned separately and never enters the body text.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata as ud
from pathlib import Path

EXTRACTOR_VERSION = "2026-10-25.5"
LINE_TOLERANCE = 5.0
GAP = 1.2
MARK_REACH = 8.0
HEAD_LIMIT = 90.0
SUPERSCRIPT_SIZE = 8.0
SUPERSCRIPTS = str.maketrans("0123456789+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺")


def font_class(fontname: str) -> str:
    name = fontname.split("+", 1)[-1]
    if "Wingdings3" in name:
        return "W3"
    if "Wingdings" in name:
        return "W"
    if "Verdana" in name:
        return "V"
    if "BoldItalic" in name or "Bold,Italic" in name:
        return "BI"
    if "Bold" in name:
        return "B"
    if "Italic" in name:
        return "I"
    return "R"


def _is_mark(text: str) -> bool:
    return bool(text) and all(ud.combining(ch) for ch in text)


def _attach_marks(chars: list[dict]) -> list[dict]:
    """Return base glyphs, each with a ``marks`` list of the combining marks it carries.

    A mark normally follows its base in the content stream and is drawn as a zero-width glyph
    over it, so the stream predecessor is preferred when it lies on the same line and within
    ``MARK_REACH`` points; some marks are drawn in a separate run (``ā̊`` before a gap), which
    the stream order still resolves. Otherwise the geometrically nearest base wins."""
    bases: list[dict] = []
    index: dict[int, dict] = {}
    last_base = None
    pending: list[tuple[dict, dict | None]] = []
    for c in chars:
        if _is_mark(c["text"]):
            pending.append((c, last_base))
            continue
        base = dict(c, marks=[])
        bases.append(base)
        last_base = base
    for mark, pred in pending:
        mx = (mark["x0"] + mark["x1"]) / 2
        if pred is not None and abs(pred["top"] - mark["top"]) <= 9 \
                and pred["x0"] - 2 <= mx <= pred["x1"] + MARK_REACH:
            pred["marks"].append(mark["text"])
            # a mark drawn in its own run past the base glyph widens the base: no space
            pred["x1"] = max(pred["x1"], mx + 3.5)
            continue
        best, best_score = None, None
        for base in bases:
            dtop = abs(base["top"] - mark["top"])
            if dtop > 9:
                continue
            inside = base["x0"] - 0.5 <= mx <= base["x1"] + 0.5
            distance = abs((base["x0"] + base["x1"]) / 2 - mx)
            score = (0 if inside else 1, distance + dtop * 0.25)
            if best_score is None or score < best_score:
                best, best_score = base, score
        if best is None:
            # an isolated mark: keep it as its own glyph so nothing is lost
            bases.append(dict(mark, marks=[], text="\u25cc" + mark["text"]))
        else:
            best["marks"].append(mark["text"])
    return bases


def _lines(bases: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for base in sorted(bases, key=lambda c: (c["top"], c["x0"])):
        if lines and abs(base["top"] - lines[-1][0]["top"]) <= LINE_TOLERANCE:
            lines[-1].append(base)
        else:
            lines.append([base])
    for line in lines:
        line.sort(key=lambda c: c["x0"])
    return lines


def _spans(line: list[dict]) -> list[list[str]]:
    spans: list[list[str]] = []
    prev = None
    for glyph in line:
        cls = font_class(glyph["fontname"])
        text = glyph["text"]
        if glyph["size"] < SUPERSCRIPT_SIZE and text.strip():
            text = text.translate(SUPERSCRIPTS)
        text += "".join(glyph["marks"])
        if prev is not None and glyph["x0"] - prev["x1"] > GAP and text != " " \
                and prev["text"] != " ":
            spans.append([spans[-1][0] if spans else cls, " "])
        if spans and spans[-1][0] == cls:
            spans[-1][1] += text
        else:
            spans.append([cls, text])
        prev = glyph
    out = []
    for cls, text in spans:
        text = ud.normalize("NFC", text)
        if out and out[-1][0] == cls:
            out[-1][1] += text
        else:
            out.append([cls, text])
    return out


def extract_page(page) -> dict:
    mid = page.width / 2
    head = [c for c in page.chars if c["top"] < HEAD_LIMIT]
    body = [c for c in page.chars if c["top"] >= HEAD_LIMIT]
    out = {"lines": []}
    head_bases = _attach_marks(head)
    number = "".join(c["text"] for c in sorted(head_bases, key=lambda c: c["x0"])
                     if c["text"].isdigit() and abs((c["x0"] + c["x1"]) / 2 - mid) < 20)
    guide = "".join(c["text"] + "".join(c["marks"]) for c in sorted(head_bases, key=lambda c: c["x0"])
                    if not (c["text"].isdigit() and abs((c["x0"] + c["x1"]) / 2 - mid) < 20))
    out["printed_number"] = int(number) if number else None
    # guide-word homograph digits are small (5.7 pt)
    out["guide"] = ud.normalize("NFC", guide.strip())
    for col in (0, 1):
        chars = [c for c in body if ((c["x0"] + c["x1"]) / 2 < mid) == (col == 0)]
        for line in _lines(_attach_marks(chars)):
            spans = _spans(line)
            text = "".join(t for _, t in spans)
            if not text.strip():
                continue
            out["lines"].append({
                "col": col, "top": round(line[0]["top"], 1), "x0": round(line[0]["x0"], 1),
                "spans": spans,
            })
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(pdf_path: Path, first: int, last: int, cache: Path | None = None,
            pdf_sha: str | None = None) -> list[dict]:
    """Extract PDF pages ``first..last`` (1-based, inclusive), reusing a matching cache."""
    if cache is not None and cache.exists():
        with cache.open(encoding="utf-8") as handle:
            meta = json.loads(handle.readline())
            if meta.get("version") == EXTRACTOR_VERSION and meta.get("sha256") == pdf_sha \
                    and meta.get("range") == [first, last]:
                return [json.loads(line) for line in handle]
    import pdfplumber  # local import: only needed when the cache is cold

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for number in range(first, last + 1):
            page = pdf.pages[number - 1]
            record = extract_page(page)
            record["pdf_page"] = number
            pages.append(record)
            page.flush_cache()
            if hasattr(page, "close"):
                page.close()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"version": EXTRACTOR_VERSION, "sha256": pdf_sha,
                                     "range": [first, last]}) + "\n")
            for record in pages:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(cache)
    return pages
