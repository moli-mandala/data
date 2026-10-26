"""Layout reconstruction for the cached OCR of Kamioka & Yamada 1979 (see ocr_runner.py).

Turns one cached page (Tesseract word boxes on the cleaned right page, recorded underlines and
overstrike marks, Persian word boxes on the left page) into ordered text lines whose tokens know
whether they are underlined (= Lari) and carry their re-attached diacritics.
"""

from __future__ import annotations

import difflib
import re
import statistics
import unicodedata as ud

COMBINING = {"macron": "\u0304", "diaeresis": "\u0308", "caron": "\u030c",
             "acute": "\u0301", "grave": "\u0300"}
# letters each mark may sit on in this source (the typewriter overstrikes)
TAKES = {
    "macron": set("aeiou"),
    "diaeresis": set("aeiou"),
    "caron": set("szcgj"),
    "acute": set("aeiou"),
    "grave": set("aeiou"),
}
# order in which stacked marks are written on one letter (macron/diaeresis before the accent)
MARK_ORDER = {"\u0304": 0, "\u0308": 0, "\u030c": 0, "\u0301": 1, "\u0300": 1}
NARROW = set("'‘’`,.;:!|\"“”")
HALF_NARROW = set("iljtf1I()[]")


def half_width(ch: str, pitch: float) -> float:
    if ch in NARROW:
        return 0.15 * pitch
    if ch in HALF_NARROW:
        return 0.3 * pitch
    return 0.4 * pitch


def underline_for(word: dict, underlines: list[list[int]], base: float) -> float:
    """Fraction of the word's width that has an underline just below the line's baseline."""
    wx0, wx1 = word["x0"], word["x1"]
    covered = 0
    for y0, y1, x0, x1 in underlines:
        if not (base - 6 <= y0 <= base + 22):
            continue
        ov = min(wx1, x1) - max(wx0, x0)
        if ov > 0:
            covered += ov
    return covered / max(1, wx1 - wx0)


def attach_marks(word: dict, marks: list[dict], pitch: float) -> tuple[str, list[dict]]:
    """Insert combining marks into the word text; returns (text, per-mark resolution log)."""
    text = word["text"]
    n = len(text)
    log = []
    if not marks or n == 0:
        return text, log
    c0 = word["x0"] + half_width(text[0], pitch)
    if n > 1:
        cl = word["x1"] - half_width(text[-1], pitch)
        p = (cl - c0) / (n - 1)
        if not (0.8 * pitch <= p <= 1.2 * pitch):
            p = pitch
    else:
        p = pitch
    add: dict[int, list[str]] = {}
    for m in sorted(marks, key=lambda m: m["x0"]):
        L = m["letter"]
        cx = (L[2] + L[3]) / 2
        idx = round((cx - c0) / p)
        cls = classify_mark(m)
        entry = {"cls": cls, "x": (m["x0"] + m["x1"]) // 2, "y": m["y0"], "idx": idx, "status": ""}
        if cls == "dot":
            ch = text[idx] if 0 <= idx < n else ""
            if ch.lower() in "ij" or any(text[j].lower() in "ij" for j in (idx - 1, idx + 1) if 0 <= j < n):
                entry["status"] = "i-dot"
            elif ch.lower() in "u":
                entry["status"] = "single-dot-over-u"
            else:
                entry["status"] = "stray-dot"
            log.append(entry)
            continue
        if cls not in COMBINING:
            entry["status"] = "unclassified"
            log.append(entry)
            continue
        ok = TAKES[cls]
        chosen = None
        for j in (idx, idx - 1, idx + 1):
            if 0 <= j < n and text[j].lower() in ok:
                chosen = j
                break
        if chosen is None:
            entry["status"] = "unplaced"
            log.append(entry)
            continue
        entry["idx"] = chosen
        entry["status"] = "placed" if chosen == idx else "placed-shifted"
        add.setdefault(chosen, []).append(COMBINING[cls])
        log.append(entry)
    out = []
    for j, ch in enumerate(text):
        out.append(ch)
        for c in sorted(set(add.get(j, [])), key=MARK_ORDER.get):
            out[-1] += c
    return ud.normalize("NFC", "".join(out)), log


def page_lines(page: dict, pitch: float = 21.3) -> list[dict]:
    """Ordered lines of the right page with token metadata."""
    words = [w for w in page["words"]]
    # group by Tesseract line id, then order by y
    groups: dict[tuple, list[dict]] = {}
    for w in words:
        groups.setdefault((w["block"], w["par"], w["line"]), []).append(w)
    lines = []
    marks = page["marks"]
    gl = page.get("glyphs") or [[] for _ in words]
    for w, g in zip(words, gl):
        w["_glyphs"] = g
    for key, ws in groups.items():
        ws.sort(key=lambda w: w["x0"])
        base = statistics.median(w["y1"] for w in ws)
        # baseline: bottom of words without descenders is better; use the minimum of the
        # most common bottoms
        bottoms = sorted(w["y1"] for w in ws)
        base = bottoms[len(bottoms) // 2] if len(bottoms) < 3 else statistics.median_low(bottoms)
        toks = []
        for w in ws:
            wm = [m for m in marks
                  if w["x0"] - 4 <= (m["letter"][2] + m["letter"][3]) / 2 <= w["x1"] + 4
                  and w["y0"] - 30 <= m["letter"][0] <= w["y1"]]
            text, log = attach_marks(w, wm, pitch)
            ul = underline_for(w, page["underlines"], base)
            cells = glyph_cells(w.get("_glyphs", []))
            lead_top = cells[0]["top_only"] if cells else None
            if ul >= 0.5:
                text, arep = fix_apostrophes_by_glyphs(text, w.get("_glyphs", []), True)
            else:
                arep = []
            text, qrep = fix_q_g_by_glyphs(text, w.get("_glyphs", []), ul >= 0.5)
            arep = arep + qrep
            toks.append({"text": text, "raw": w["text"], "x0": w["x0"], "x1": w["x1"],
                         "y0": w["y0"], "y1": w["y1"], "conf": w["conf"],
                         "ul": ul, "marks": log, "lead_top": lead_top,
                         "apostrophe": arep})
        lines.append({"y": min(w["y0"] for w in ws), "ybase": base,
                      "x0": ws[0]["x0"], "toks": toks})
    lines.sort(key=lambda l: l["y"])
    # merge Tesseract lines that share a baseline (split words on one printed line)
    merged: list[dict] = []
    for l in lines:
        if merged and abs(l["ybase"] - merged[-1]["ybase"]) <= 12:
            merged[-1]["toks"].extend(l["toks"])
            merged[-1]["toks"].sort(key=lambda t: t["x0"])
            merged[-1]["x0"] = min(merged[-1]["x0"], l["x0"])
        else:
            merged.append(l)
    return merged


def persian_lines(page: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for w in page["persian"]:
        groups.setdefault((w["block"], w["par"], w["line"]), []).append(w)
    out = []
    for ws in groups.values():
        ws.sort(key=lambda w: -w["x1"])  # right-to-left
        out.append({"y": min(w["y0"] for w in ws), "ybase": statistics.median(w["y1"] for w in ws),
                    "text": " ".join(w["text"] for w in ws)})
    out.sort(key=lambda l: l["y"])
    return out


def line_text(line: dict, mark_lari: bool = False) -> str:
    parts = []
    for t in line["toks"]:
        s = t["text"]
        if mark_lari and t["ul"] >= 0.5:
            s = "_" + s + "_"
        parts.append(s)
    return " ".join(parts)


DIGITS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})


def entry_number(tok_text: str) -> int | None:
    m = re.fullmatch(r"[^\w]*([0-9OolI|SB]{4})[.,:;]?", tok_text)
    if not m:
        return None
    s = m.group(1).translate(DIGITS)
    if not s.isdigit():
        return None
    if not re.search(r"\d", m.group(1)):
        return None
    return int(s)


def classify_mark(mark: dict) -> str:
    """Final mark class.  The runner's provisional ``caron`` class (wide, 7–13 px tall) also
    catches the slanted typewriter stress accent; a caron is a symmetric V with ink in both top
    corners, an acute has ink only at its top right and its foot at the left."""
    import numpy as np
    cls = mark["cls"]
    if cls != "caron" or not mark.get("bm"):
        return cls
    b = np.array([[c == "1" for c in r] for r in mark["bm"]])
    h, w = b.shape
    k = max(2, h // 3)
    side = max(1, int(w * 0.35))
    top = b[:k]
    tl = bool(top[:, :side].any())
    tr = bool(top[:, w - side:].any())
    if tl and tr:
        return "caron"
    if tr and not tl:
        return "acute"
    if tl and not tr:
        return "grave"
    return "unknown"


APOS_LOOKALIKE = set("tijlIJ1|‘’`")


def _clusters(text: str) -> list[str]:
    out: list[str] = []
    for ch in ud.normalize("NFD", text):
        if ud.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def reconcile_apostrophes(primary: str, alt: str) -> tuple[str, list[str]]:
    """Repair the glottal-stop apostrophe of ``primary`` (cleaned-image reading with the
    re-attached diacritics) from ``alt`` (independent reading of the raw page image).

    The cleaned bilevel image makes Tesseract read the typewriter apostrophe as ``t`` or
    ``i`` (``gāti`` for ``gā'i``) or add a spurious ``t`` after it (``'takonem``), whereas the
    raw-image reading keeps it but loses the overstrike diacritics.  Only edits whose one side is
    an apostrophe are taken from ``alt``; every other character, and every diacritic, stays as
    in ``primary``.  Spaces are never inserted or removed."""
    cl = _clusters(primary)
    base = "".join(c[0] for c in cl).replace("‘", "'").replace("’", "'").replace("`", "'")
    a = ud.normalize("NFD", alt)
    a = "".join(c for c in a if not ud.combining(c)).replace("‘", "'").replace("’", "'")
    a = a.replace("`", "'")
    sm = difflib.SequenceMatcher(None, base.lower(), a.lower(), autojunk=False)
    repairs = []
    new = list(cl)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        ps, as_ = base[i1:i2], a[j1:j2]
        if op == "replace" and ps == "g" and as_ == "q":
            # typewriter q and g: where the two readings disagree, q is right (all 20
            # disagreements checked: qermez, qeymat, qesmat, qors, farq, dayqa, …)
            new[i1] = "q" + new[i1][1:]
            repairs.append(f"g->q at {i1}")
        elif op == "replace" and as_ == "'" and len(ps) == 1 and ps in APOS_LOOKALIKE:
            new[i1] = "'"
            repairs.append(f"{ps}->' at {i1}")
        elif op == "replace" and as_ == "'" and ps in ("'t", "t'", "'i"):
            new[i1:i2] = ["'"] + [""] * (i2 - i1 - 1)
            repairs.append(f"{ps}->' at {i1}")
        elif op == "delete" and ps == "t" and (
                (i1 > 0 and base[i1 - 1] == "'") or (i2 < len(base) and base[i2] == "'")):
            new[i1] = ""
            repairs.append(f"spurious t at {i1}")
        elif op == "insert" and as_ == "'" and 0 < i1 < len(base) and base[i1 - 1] != " " \
                and base[i1] == "t" and i1 + 1 < len(base) and base[i1 + 1] in "aeiouāü":
            # the raw reading has the apostrophe *and* the spurious t ('t): the primary t is
            # the misread apostrophe
            new[i1] = "'"
            repairs.append(f"t->' at {i1} (alt 't)")
        elif op == "insert" and as_ == "'" and 0 < i1 < len(base) and base[i1 - 1] != " " \
                and base[i1].isalpha() and base[i1 - 1].isalpha():
            new[i1] = "'" + new[i1]
            repairs.append(f"inserted ' at {i1}")
        elif op == "insert" and as_ == "'" and i1 < len(base) and (i1 == 0 or base[i1 - 1] == " ") \
                and base[i1] in "aeiouāü":
            new[i1] = "'" + new[i1]
            repairs.append(f"inserted initial ' at {i1}")
    out = ud.normalize("NFC", "".join(new))
    if out.count(" ") != primary.count(" "):
        return primary, []
    return out, repairs


def glyph_cells(glyphs: list[list[int]]) -> list[dict]:
    """Merge a word's components into character cells (x-overlapping components belong to one
    typewriter character) and mark the cells whose ink lies wholly above the x-height."""
    # specks and underline remnants are not characters
    glyphs = [g for g in glyphs if g[3] - g[1] >= 4 and (g[2] - g[0]) * (g[3] - g[1]) >= 16]
    if not glyphs:
        return []
    letters = [g for g in glyphs if g[3] - g[1] >= 17] or \
        [g for g in glyphs if g[3] - g[1] >= 12]  # apostrophes are 14-16 px, letters ~20
    if not letters:
        return []
    base = statistics.median(g[3] for g in letters)
    cells: list[dict] = []
    for g in sorted(glyphs):
        ov = (min(g[2], cells[-1]["x1"]) - max(g[0], cells[-1]["x0"])) if cells else 0
        if cells and ov > 0.5 * min(g[2] - g[0], cells[-1]["x1"] - cells[-1]["x0"]):
            c = cells[-1]
            c["x1"] = max(c["x1"], g[2])
            c["y0"] = min(c["y0"], g[1])
            c["y1"] = max(c["y1"], g[3])
            c["n"] += 1
        else:
            cells.append({"x0": g[0], "x1": g[2], "y0": g[1], "y1": g[3], "n": 1})
    for c in cells:
        c["top_only"] = c["y1"] <= base - 11 and c["y1"] - c["y0"] <= 18 and c["x1"] - c["x0"] <= 12
    return cells


def fix_apostrophes_by_glyphs(text: str, glyphs: list[list[int]],
                              underlined: bool = False) -> tuple[str, list[str]]:
    """Use the typewriter's one-character-per-cell geometry to repair the apostrophe: a cell
    whose ink lies wholly above the x-height is ``'``; OCR reads it as ``t`` or ``i`` or adds a
    spurious ``t`` after it.  Applied only when the cells and characters can be put in exact
    one-to-one correspondence (after at most one spurious ``t``); otherwise nothing changes."""
    cells = glyph_cells(glyphs)
    cl = _clusters(text)
    if not cells:
        return text, []
    if not any(c["top_only"] for c in cells):
        return text, []
    base = [c[0] for c in cl]
    reps = []
    if len(base) == len(cells) + 1:
        # a spurious t after an apostrophe cell
        for j in range(len(base) - 1):
            if base[j] in "'‘’`" and base[j + 1] == "t":
                trial = cl[:j + 1] + cl[j + 2:]
                if len(trial) == len(cells) and cells[j]["top_only"] and \
                        not cells[j + 1]["top_only"]:
                    cl = trial
                    base = [c[0] for c in cl]
                    reps.append(f"spurious t after ' at {j + 1}")
                    break
    if len(cells) == len(base) + 1:
        tops = [j for j, c in enumerate(cells) if c["top_only"]]
        if len(tops) == 1 and "'" not in base and all(ch.isalpha() for ch in base):
            j = tops[0]
            cl = cl[:j] + ["'"] + cl[j:]
            base = [c[0] for c in cl]
            reps.append(f"missing ' inserted at {j}")
    if len(base) != len(cells):
        return text, reps if reps else []
    for j, c in enumerate(cells):
        if c["top_only"] and base[j] in "‘’`\"“”" and j == 0:
            cl[j] = "'"
        elif c["top_only"] and base[j] in "‘’`":
            cl[j] = "'"
        elif c["top_only"] and base[j] in APOS_LOOKALIKE:
            reps.append(f"{base[j]}->' at {j} (glyph)")
            cl[j] = "'"
        elif not c["top_only"] and base[j] in "'‘’`" and c["y1"] - c["y0"] >= 14:
            pass  # a full-height glyph read as an apostrophe: leave for review
    return ud.normalize("NFC", "".join(cl)), reps


QG_THRESHOLD = 65  # descender centroid (% of glyph width): g 38–49, q 77–83 on this scan


def fix_q_g_by_glyphs(text: str, glyphs: list[list[int]],
                      underlined: bool = True) -> tuple[str, list[str]]:
    """Typewriter ``q`` and ``g`` differ in the descender: a straight stem at the right for q,
    a hook toward the left for g.  The runner records the horizontal centroid of each glyph's
    bottom quarter; a letter read as g with a right-hand descender is q and vice versa.
    Applied only when the word's tall glyphs and its letters correspond one to one."""
    if not any(ch in "gqj" for ch in text) or not glyphs or len(glyphs[0]) < 5:
        return text, []
    tall = []
    for g in sorted(gl for gl in glyphs if gl[3] - gl[1] >= 12):
        if tall and min(g[2], tall[-1][2]) - max(g[0], tall[-1][0]) > \
                0.5 * min(g[2] - g[0], tall[-1][2] - tall[-1][0]):
            continue  # a second piece of the same character
        tall.append(g)
    cl = _clusters(text)
    pos = [i for i, c in enumerate(cl) if c[0].isalpha() or c[0] in "'‘’"]
    if len(pos) != len(tall):
        return text, []
    reps = []
    base_y = statistics.median(g[3] for g in tall)
    for i, g in zip(pos, tall):
        ch = cl[i][0]
        if ch == "j" and g[3] <= base_y + 3 and underlined and len(tall) >= 3:
            # j has a descender; a glyph that stops on the baseline is l
            cl[i] = "l" + cl[i][1:]
            reps.append(f"j->l at {i} (no descender)")
        elif ch == "g" and g[4] >= QG_THRESHOLD:
            cl[i] = "q" + cl[i][1:]
            reps.append(f"g->q at {i} (descender {g[4]}%)")
        elif ch == "q" and 0 <= g[4] < 55:
            cl[i] = "g" + cl[i][1:]
            reps.append(f"q->g at {i} (descender {g[4]}%)")
    return ud.normalize("NFC", "".join(cl)), reps
