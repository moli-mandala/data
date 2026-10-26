"""Segment the extracted OAD dictionary lines into raw source units.

Reading order is page → left column → right column → line top. Every body line belongs to
exactly one unit. A unit starts at a line whose first visible glyph is a structural marker:

* ``◊`` (Verdana) — a non-verb lemma;
* ``♣`` — a verb lemma (root headword ``√…``);
* the Wingdings 3 bullet (extracted as ``9``) — an attestation of the current lemma;
* the Wingdings pointing hand (extracted as ``F``) — a cross-reference from a form to a lemma
  (an ``F`` *inside* a line is an in-text "see" pointer and stays in its unit's prose);
* a short roman verb-stem line (``pres. ah-/h-``, ``s-aor. gāuš-``) — a stem sub-heading;
* a centred short bold line — an alphabet section heading (``a``, ``ā``, ``x́``, ``xᵛ`` …).

Any other line continues the current unit, which is how page and column breaks are bridged.
"""

from __future__ import annotations

import re

STEM_LINE = re.compile(
    r"^(?:(?:pres|perf|aor|fut|pass|pluperf|caus|des|int|redupl|them|s-aor|is-aor|Öaor|"
    r"root|impf|sbj)(?:\.|/|\s)*)+\s*"
    r"[^\s():‘’|]+-(?:\s*/\s*[^\s():‘’|]+-)*(?:\s+(?:or|<)\s+[^\s():‘’|]+)*\s*$"
)


def line_text(line: dict) -> str:
    return "".join(text for _, text in line["spans"])


def first_glyph(line: dict) -> tuple[str, str]:
    for cls, text in line["spans"]:
        if text.strip():
            return cls, text.strip()[0]
    return "", ""


def column_start(page_lines: list[dict], col: int) -> float:
    xs = sorted(l["x0"] for l in page_lines if l["col"] == col)
    return xs[len(xs) // 10] if xs else 0.0


def classify(line: dict, col_x: float) -> str:
    cls, glyph = first_glyph(line)
    text = line_text(line).strip()
    if cls == "V" and glyph == "◊":
        return "lemma"
    if cls == "R" and glyph == "♣":
        return "verb"
    if cls == "W3" and glyph == "9":
        return "attestation"
    if cls == "W" and glyph == "F":
        return "xref"
    visible = [(c, t) for c, t in line["spans"] if t.strip()]
    if visible and all(c == "B" for c, _ in visible) and len(text) <= 3 \
            and line["x0"] - col_x > 50:
        return "section"
    if cls == "R" and STEM_LINE.match(text):
        return "stem"
    return "cont"


def segment(pages: list[dict]) -> list[dict]:
    units: list[dict] = []
    for page in pages:
        lines = page["lines"]
        for col in (0, 1):
            col_x = column_start(lines, col)
            for line in sorted((l for l in lines if l["col"] == col), key=lambda l: l["top"]):
                kind = classify(line, col_x)
                loc = {"pdf_page": page["pdf_page"], "printed_page": page["pdf_page"] - 29,
                       "col": col + 1, "top": line["top"]}
                if kind == "cont" and units:
                    units[-1]["lines"].append(dict(line, **loc))
                    continue
                if kind == "cont":
                    kind = "orphan"
                units.append({"kind": kind, "lines": [dict(line, **loc)], **loc})
    return units
