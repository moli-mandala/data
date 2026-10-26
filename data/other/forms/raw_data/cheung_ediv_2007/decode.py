"""Stage 2: decode glyph-name tokens to Unicode lines with inline style markers.

Style markers (Private Use Area, never present in the source text):
``I_ON/I_OFF`` italic (citation forms), ``B_ON/B_OFF`` bold (headwords),
``SUP_ON/SUP_OFF`` and ``SUB_ON/SUB_OFF`` raised/lowered 5–6.5 pt characters.
Combining marks are attached to the preceding base letter *before* any style change,
so a mark printed in another font never separates from its letter.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLYPHS = HERE / "glyphs.csv"

I_ON, I_OFF, B_ON, B_OFF = "", "", "", ""
SUP_ON, SUP_OFF, SUB_ON, SUB_OFF = "", "", "", ""
MARKERS = I_ON + I_OFF + B_ON + B_OFF + SUP_ON + SUP_OFF + SUB_ON + SUB_OFF
TOKEN = re.compile("\x01([^:\x02]+):([^\x02]+)\x02")
PLACEHOLDER_FONTS = {"PersianNaskhStar": "[Arabic script]"}


class UnknownGlyph(ValueError):
    pass


@lru_cache(maxsize=1)
def glyph_table() -> dict[tuple[str, str], str]:
    with GLYPHS.open(encoding="utf-8", newline="") as handle:
        return {(r["Font"], r["Glyph"]): r["Output"] for r in csv.DictReader(handle)}


@lru_cache(maxsize=None)
def _agl_ascii(name: str) -> str | None:
    from fontTools import agl
    char = agl.toUnicode(name)
    return char if len(char) == 1 and char.isascii() else None


def decode_glyph(font: str, name: str) -> str:
    table = glyph_table()
    if (font, name) in table:
        return table[(font, name)]
    char = _agl_ascii(name)
    if char is None:
        raise UnknownGlyph(f"{font}:{name}")
    return char


def _style(font: str) -> str:
    if font.startswith("GreekMousaStar"):
        return "g"  # resolved per word in line_items
    if "Italic" in font or font.endswith("Ita"):
        return "i"
    if "Bold" in font:
        return "b"
    return "r"


def line_items(line: dict) -> list[tuple[str, str, str]]:
    """Return ``(text, style, script)`` items, one per decoded glyph or placeholder."""
    items: list[tuple[str, str, str]] = []
    for span in line["spans"]:
        font, script = span["f"], span["k"]
        placeholder = next((v for k, v in PLACEHOLDER_FONTS.items() if font.startswith(k)), None)
        if placeholder:
            text = TOKEN.sub("", span["t"]).strip()
            if items and items[-1][0] == placeholder:
                continue
            items.append((placeholder, "r", ""))
            continue
        style = _style(font)
        pos = 0
        for m in TOKEN.finditer(span["t"]):
            for ch in span["t"][pos:m.start()]:
                items.append((ch, "r" if ch == " " else style, script))
            out = decode_glyph(m.group(1), m.group(2))
            if out:
                items.append((out, style, script))
            pos = m.end()
        for ch in span["t"][pos:]:
            items.append((ch, "r" if ch == " " else style, script))
    # attach postfix combining marks to the preceding base glyph
    merged: list[list[str]] = []
    for text, style, script in items:
        if merged and text and all(unicodedata.combining(c) for c in text):
            merged[-1][0] += text
            continue
        merged.append([text, style, script])
    # Greek-font letters spell Sogdian/Choresmian β δ γ θ inside italic citation forms, and
    # Bactrian and Greek citations are printed wholly in the (upright) Greek font. Both are
    # citation forms, so Greek-font glyphs take the citation (italic) style; punctuation
    # glyphs of the Greek font stay roman.
    start = 0
    while start < len(merged):
        end = start
        while end < len(merged) and merged[end][0] != " ":
            end += 1
        word = merged[start:end]
        if any(item[1] == "g" for item in word):
            for item in word:
                # Bactrian ϸ (Sims-Williams' sho) is set with the text font's thorn
                if item[0] == "þ":
                    item[0], item[1] = "ϸ", "g"
                if item[1] == "g":
                    item[1] = "i" if any(c.isalpha() or unicodedata.combining(c)
                                         for c in item[0]) else "r"
            # Greek-script (Bactrian) words take their hyphens, apostrophes and inner
            # brackets from the text font: αζ(α)βαρ-, -βαρο, ϸαο-
            opened = False
            for k, item in enumerate(word):
                if item[1] == "i":
                    if item[0] == "(":
                        opened = True
                    elif item[0] == ")":
                        opened = False
                    continue
                if item[0] in "-°/" or (item[0] == "’" and k + 1 < len(word)
                                          and word[k + 1][0][:1].isalpha()):
                    item[1] = "i"  # a word-final ’ closes a gloss (‘soul, πνεῦμα’)
                elif item[0] == "(" and k > 0:
                    item[1], opened = "i", True
                elif item[0] == ")" and opened:
                    item[1], opened = "i", False
        else:
            # any form: a bracket opened inside an italic run but closed in the roman font
            # (Bal. βī(θ)/…, set "βī(θ" + roman ")") belongs to the form
            opened = False
            for item in word:
                if item[0] == "(" and item[1] == "i":
                    opened = True
                elif item[0] == ")" and opened:
                    item[1], opened = "i", False
        start = end + 1
    # a roman hyphen/apostrophe/period between two italic glyphs of one word is part of the
    # form (Greek-font Bactrian αγι-σινδο uses the roman hyphen of the text font)
    for k in range(1, len(merged) - 1):
        if (merged[k][1] == "r" and merged[k][0] in "-’.|" and merged[k - 1][1] == "i"
                and merged[k + 1][1] == "i"):
            merged[k][1] = "i"
    return [tuple(x) for x in merged]


def render(line: dict, *, markup: bool = True) -> str:
    out = []
    cur_style, cur_script = "r", ""
    on = {"i": I_ON, "b": B_ON, "sup": SUP_ON, "sub": SUB_ON}
    off = {"i": I_OFF, "b": B_OFF, "sup": SUP_OFF, "sub": SUB_OFF}
    for text, style, script in line_items(line):
        if text == " ":
            style = cur_style  # spaces never open or close a style run
        if markup:
            if script != cur_script:
                if cur_script:
                    out.append(off[cur_script])
                if script:
                    out.append(on[script])
                cur_script = script
            if style != cur_style:
                if cur_style != "r":
                    out.append(off[cur_style])
                if style != "r":
                    out.append(on[style])
                cur_style = style
        out.append(text)
    if markup:
        if cur_style != "r":
            out.append(off[cur_style])
        if cur_script:
            out.append(off[cur_script])
    return unicodedata.normalize("NFC", "".join(out))


def plain(text: str) -> str:
    return "".join(ch for ch in text if ch not in MARKERS)
