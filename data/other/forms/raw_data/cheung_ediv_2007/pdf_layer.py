"""Stage 1 of the Cheung (2007) importer: the PDF text layer as glyph-named lines.

The born-digital PDF embeds 23 Type 1 subset fonts. The transliteration fonts
(``TimesNewRoman+`` and its italic/bold cuts, ``TimesNewRomanStar``, ``TITUSCyberbit``,
``ArialUnicodeMS``) carry custom glyphs that have no usable ``ToUnicode`` entry, so
ordinary extractors print ``(cid:N)`` or nothing. Every font is a single PDF object
shared by all pages, so each ``(font, code)`` has one glyph name document-wide.

This module bypasses ``ToUnicode`` for every font that has an ``/Encoding /Differences``
array and records the *glyph name* of each character instead (``\\x01Font:name\\x02``
tokens). Decoding the names to Unicode is stage 2 (``glyphs.csv`` + ``decode.py``), so
the decoding table can be audited and revised without re-reading the PDF.

Lines are rebuilt from character baselines. Overprinted duplicate glyphs (the bold
headword font draws ``*`` twice at the same position) are dropped; 5.0/6.5 pt characters
are attached to the nearest text line as superscripts/subscripts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

PDF_SHA256 = "44c23cec076f0948e5d7747dd68ca58115f5acccb3cdb9916abc92ddeda6f952"
PDF_PAGES = 625
EXTRACT_VERSION = 5

TOKEN_OPEN, TOKEN_CLOSE = "\x01", "\x02"
_PATCHED = False


def short_font(basefont: str) -> str:
    return basefont.split("+", 1)[-1]


def _patch_pdfminer() -> None:
    """Make every Differences-encoded simple font report glyph-name tokens."""
    global _PATCHED
    if _PATCHED:
        return
    from pdfminer import pdffont
    from pdfminer.pdftypes import resolve1
    from pdfminer.psparser import literal_name

    orig_init = pdffont.PDFSimpleFont.__init__
    orig_to_unichr = pdffont.PDFSimpleFont.to_unichr

    def init(self, descriptor, widths, spec):
        orig_init(self, descriptor, widths, spec)
        names: dict[int, str] = {}
        encoding = resolve1(spec.get("Encoding")) if "Encoding" in spec else None
        if isinstance(encoding, dict):
            code = None
            for item in resolve1(encoding.get("Differences", [])) or []:
                item = resolve1(item)
                if isinstance(item, int):
                    code = item
                elif code is not None:
                    names[code] = literal_name(item)
                    code += 1
        self._cheung_names = names
        try:
            self._cheung_font = short_font(literal_name(spec["BaseFont"]))
        except Exception:  # pragma: no cover - every font in this PDF has a BaseFont
            self._cheung_font = "?"

    def to_unichr(self, cid):
        name = getattr(self, "_cheung_names", {}).get(cid)
        if name is None:
            return orig_to_unichr(self, cid)
        return f"{TOKEN_OPEN}{self._cheung_font}:{name}{TOKEN_CLOSE}"

    pdffont.PDFSimpleFont.__init__ = init
    pdffont.PDFSimpleFont.to_unichr = to_unichr
    _PATCHED = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_lines(page) -> list[dict]:
    chars = []
    seen: dict[str, list[tuple[float, float]]] = {}
    for ch in page.chars:
        positions = seen.setdefault(ch["text"], [])
        if any(abs(ch["x0"] - x) < 0.3 and abs(ch["top"] - t) < 0.3 for x, t in positions):
            continue  # overprinted duplicate glyph (the bold headword font draws `*` twice)
        positions.append((ch["x0"], ch["top"]))
        chars.append(ch)
    body = [c for c in chars if c["size"] >= 7.5]
    small = [c for c in chars if c["size"] < 7.5]
    body.sort(key=lambda c: (c["bottom"], c["x0"]))
    lines: list[dict] = []
    for ch in body:
        if lines and abs(ch["bottom"] - lines[-1]["base"]) <= 3.0:
            lines[-1]["chars"].append(ch)
        else:
            lines.append({"base": ch["bottom"], "chars": [ch]})
    for line in lines:
        sizes = sorted(c["size"] for c in line["chars"])
        line["size"] = sizes[len(sizes) // 2]
        line["base"] = sorted(c["bottom"] for c in line["chars"])[len(line["chars"]) // 2]
        line["top"] = min(c["top"] for c in line["chars"])
    # A 7.9 pt homonym index raised beside a 12 pt headword (``*baud²``) forms its own
    # pseudo-line; fold such tiny lines into the larger line they overlap as scripts.
    kept = []
    for line in lines:
        mid = (line["top"] + line["base"]) / 2
        if len(line["chars"]) <= 3 and any(
            other is not line and other["size"] >= line["size"] * 1.2
            and other["top"] - 4 <= mid <= other["base"] + 4
            for other in lines
        ):
            small.extend(line["chars"])
            continue
        kept.append(line)
    lines = kept
    for ch in small:
        mid = (ch["top"] + ch["bottom"]) / 2
        best = None
        for line in lines:
            if line["top"] - 6 <= mid <= line["base"] + 6:
                dist = abs(mid - (line["top"] + line["base"]) / 2)
                if best is None or dist < best[0]:
                    best = (dist, line)
        if best is None:
            lines.append({"base": ch["bottom"], "top": ch["top"], "size": ch["size"],
                          "chars": [ch]})
            continue
        line = best[1]
        ch = dict(ch)
        ch["script"] = "sup" if ch["bottom"] < line["base"] - 1.0 else "sub"
        line["chars"].append(ch)
    lines.sort(key=lambda l: l["base"])
    out = []
    for line in lines:
        cs = sorted(line["chars"], key=lambda c: c["x0"])
        spans: list[dict] = []
        prev_x1 = None
        for c in cs:
            font = short_font(c["fontname"])
            script = c.get("script", "")
            size = round(c["size"], 1)
            gap = prev_x1 is not None and c["x0"] - prev_x1 > 1.6 and not c["text"].isspace()
            if (spans and spans[-1]["f"] == font and spans[-1]["s"] == size
                    and spans[-1]["k"] == script and not gap):
                spans[-1]["t"] += c["text"]
            else:
                spans.append({"f": font, "s": size, "k": script, "x": round(c["x0"], 1),
                              "t": (" " if gap and spans and not spans[-1]["t"].endswith(" ")
                                    else "") + c["text"]})
            prev_x1 = c["x1"]
        out.append({"base": round(line["base"], 1), "x0": round(cs[0]["x0"], 1),
                    "x1": round(cs[-1]["x1"], 1), "size": round(line["size"], 1),
                    "spans": spans})
    return out


def extract(pdf_path: Path, cache: Path, *, verify: bool = True) -> list[list[dict]]:
    """Return ``pages[pdf_index] -> lines`` (cached as gzip JSON outside the repo)."""
    if cache.exists():
        data = json.loads(gzip.decompress(cache.read_bytes()))
        if data.get("version") == EXTRACT_VERSION and data.get("sha256") == PDF_SHA256:
            return data["pages"]
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Cheung (2007) PDF not found at {pdf_path}; it is copyrighted and not checked in"
        )
    if verify and sha256(pdf_path) != PDF_SHA256:
        raise ValueError(f"unexpected PDF SHA-256 for {pdf_path}")
    _patch_pdfminer()
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        assert len(pdf.pages) == PDF_PAGES, len(pdf.pages)
        for page in pdf.pages:
            pages.append(_page_lines(page))
            page.flush_cache()
            page.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(
        {"version": EXTRACT_VERSION, "sha256": PDF_SHA256, "pages": pages},
        ensure_ascii=False).encode()))
    tmp.replace(cache)
    return pages
