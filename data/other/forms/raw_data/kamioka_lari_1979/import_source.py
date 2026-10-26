#!/usr/bin/env python3
"""Importer for Kamioka & Yamada 1979, *Lārestāni Studies 1: Lari Basic Vocabulary* (ILCAA).

Reads the pinned OCR cache written by ``ocr_runner.py`` (the scan itself is copyrighted and is
not in the repository; the runner fails clearly when it is absent) and parses the 1000 numbered
questionnaire entries of the vocabulary part (printed pp. 1–177 = PDF pp. 10–186).

Dry run (default) writes nothing and prints counts; ``--install`` writes

* ``data/other/forms/20261025-kamioka-lari.csv`` (15-column headerless rich CSV),
* ``raw_data/kamioka_lari_1979/audit.csv`` (one row per raw OCR line unit),
* ``raw_data/kamioka_lari_1979/sample.csv`` (seeded 20-entry raw-vs-image audit),
* ``raw_data/kamioka_lari_1979/manifest.json``.

Hand decisions (OCR corrections confirmed against the page image) live in ``corrections.csv``
and are applied by Entry_Key / unit; nothing is emended silently.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
import random
import re
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout  # noqa: E402
import ocr_runner  # noqa: E402

ROOT = HERE.parents[4]
CSV_OUT = ROOT / "data/other/forms/20261025-kamioka-lari.csv"
AUDIT_OUT = HERE / "audit.csv"
SAMPLE_OUT = HERE / "sample.csv"
MANIFEST_OUT = HERE / "manifest.json"
CORRECTIONS = HERE / "corrections.csv"
SAMPLE_REVIEW = HERE / "sample_review.csv"

SOURCE = "kamioka-yamada1979lari"
LANGUAGE = "Lar"
DIALECT_TAG = "dialect:Lar:kamioka-lar:L%C4%81r"
REVIEW_TAG = "uncertain"
FIRST_PDF_PAGE, LAST_PDF_PAGE = 10, 186
PAGE_OFFSET = 9  # printed page = PDF page - 9 throughout the vocabulary
ENTRY_COUNT = 1000
PITCH = 22.5
SEED = 20261025  # round 1 (errors fixed); the published sample is round 2
SAMPLE_SEED = 20261102  # round 9, the published sample (all rounds in sample_review.csv)

LARI_LETTERS = set("abcdefghijklmnopqrstuvwxyzāüšžáéíóúǘ")
LARI_ALLOWED = LARI_LETTERS | set("' -")
QUOTE_OPEN = ('"', "“", "”", "''", "‘‘", "’’", "'\"", "\"'")
GLOTTAL_CHARS = "'‘’`´"
LABELS = {"vi.": ["verb", "intr"], "vt.": ["verb", "tr"], "adv.": ["adv"]}


def nfc(s: str) -> str:
    return ud.normalize("NFC", s)


def strip_marks(s: str) -> str:
    return "".join(c for c in ud.normalize("NFD", s) if not ud.combining(c))


# ----------------------------------------------------------------------------- loading

def load_pages() -> dict[int, dict]:
    pages = {}
    missing = []
    for p in range(FIRST_PDF_PAGE, LAST_PDF_PAGE + 1):
        d = ocr_runner.load(p)
        if d is None or d.get("marks_version") != ocr_runner.MARKS_VERSION or "alt" not in d:
            missing.append(p)
            continue
        pages[p] = d
    if missing:
        sys.exit(f"OCR cache missing/stale for PDF pages {missing[:10]}…; run ocr_runner.py "
                 f"(needs the scan at {ocr_runner.PDF_PATH})")
    return pages


# ----------------------------------------------------------------------------- lines

JUNK_TOKEN = re.compile(r"^(?:[^\w'‘’\"“”(\[]|_)+$|^[A-Z']$")
NUM_TOKEN = re.compile(r"^[^\w]*[0-9OoIlJ|)\]]?([0-9OoIl|SB]{3})[.,:;]?$")
FULL_NUM = re.compile(r"^[^\w]*([0-9OoIl|SB]{4})[.,:;]?$")


def number_candidates(text: str) -> list[int]:
    """Possible entry numbers printed in a token (OCR digit confusions repaired)."""
    out = []
    m = FULL_NUM.match(text)
    if m:
        s = m.group(1).translate(layout.DIGITS)
        if s.isdigit():
            out.append(int(s))
            out.append(int(s[1:]))  # misread leading 0 (9157 for 0157)
    m = NUM_TOKEN.match(text)
    if m:
        s = m.group(1).translate(layout.DIGITS)
        if s.isdigit() and re.search(r"\d", m.group(1)):
            out.append(int(s))  # leading 0 lost in the gutter
    return list(dict.fromkeys(out))


def tok_is_quote(t: dict) -> bool:
    s = t["text"]
    return s.startswith(QUOTE_OPEN) or (s[:1] in GLOTTAL_CHARS and s[1:2].isupper())


HEAD_JUNK = re.compile(r"^[{}\\=|—_°<>@#*+^%$&;:'‘’.\-]+$")
CLOSER = re.compile(r"[\"”'’!][.,;:)\]!]*$")


def gloss_open(line: "Line", unit: "Line") -> bool:
    """True when ``line`` (the last physical line of ``unit``) leaves a quoted gloss or a
    bracketed comment unfinished: a gloss/comment opener occurs in the unit and the line's last
    token does not end in a closing quote or bracket (OCR often reads ``"`` as ``!`` or ``'``)."""
    toks = unit.all_toks()
    opened = any((i > 0 and tok_is_quote(t) or t["text"][:1] in "[(") and t["ul"] < 0.5
                 for i, t in enumerate(toks))
    if not opened or not line.toks:
        return False
    tail = line.toks[-1]["text"]
    if CLOSER.search(tail) or tail.endswith(("]", ")", "].", ").")):
        return False
    # an opened bracket closed later on the line
    text = " ".join(t["text"] for t in toks)
    if text.count("[") > text.count("]") or text.count("(") > text.count(")"):
        return True
    return True


def quote_balance(text: str) -> int:
    t = text.replace("''", '"').replace("‘‘", '"').replace("’’", '"')
    return sum(t.count(c) for c in '"“”')


class Line:
    def __init__(self, page: int, idx: int, raw: dict):
        self.page = page
        self.idx = idx
        self.y = raw["y"]
        self.x0 = raw["x0"]
        self.toks = raw["toks"]
        self.cont: list[Line] = []

    @property
    def text(self) -> str:
        return " ".join(t["text"] for t in self.toks)

    @property
    def raw_text(self) -> str:
        return " ".join(t["raw"] for t in self.toks)

    def all_toks(self):
        out = list(self.toks)
        for c in self.cont:
            out.extend(c.toks)
        return out

    @property
    def conf(self) -> float:
        cs = [t["conf"] for t in self.toks if t["conf"] >= 0]
        return sum(cs) / len(cs) if cs else 0.0

    @property
    def uid(self) -> str:
        return f"p{self.page:03d}:l{self.idx:02d}"


def page_lines(pages: dict[int, dict]):
    """All right-page lines in reading order, with page numbers and noise separated."""
    lines: list[Line] = []
    page_numbers: dict[int, str] = {}
    noise: list[Line] = []
    for p in sorted(pages):
        raw = layout.page_lines(pages[p], PITCH)
        alts = alt_lines(pages[p])
        alts_stripped = [strip_marks(a).lower() for a in alts]
        for i, r in enumerate(raw):
            ln = Line(p, i, r)
            ln.repairs = [f"{t['raw']}: {r_}" for t in ln.toks for r_ in t.get("apostrophe", [])]
            for t in ln.toks:
                # a leading quotation mark on a Lari word whose first glyph is a full letter
                # (no apostrophe glyph: an OCR speck) and which stands before a consonant, where
                # the glottal stop never occurs (Introduction 2.2 (3), p. xi)
                if t["ul"] >= 0.5 and t["text"][:1] in "\"“”'‘’`" and \
                        t["text"][1:2].isalpha() and t.get("lead_top") is False and \
                        strip_marks(t["text"][1:2]).lower() not in "aeiou":
                    ln.repairs.append(f"glyph: spurious leading mark {t['text']}")
                    t["text"] = t["text"][1:]
            # apostrophes from the independent raw-image reading
            txt = " ".join(t["text"] for t in ln.toks)
            base = strip_marks(txt).lower()
            best, score = None, 0.0
            for a, s_ in zip(alts, alts_stripped):
                if abs(len(s_) - len(base)) > 0.4 * max(len(base), 1) + 4:
                    continue
                r_ = difflib.SequenceMatcher(None, base, s_, autojunk=False).ratio()
                if r_ > score:
                    best, score = a, r_
            if best is not None and score >= 0.7:
                fixed, reps = layout.reconcile_apostrophes(txt, best)
                if reps:
                    parts = fixed.split(" ")
                    if len(parts) == len(ln.toks):
                        changed = []
                        for t, s_ in zip(ln.toks, parts):
                            # only underlined (Lari) words: English has its own apostrophes
                            if s_ != t["text"] and t["ul"] >= 0.5:
                                changed.append(f"{t['text']}->{s_}")
                                t["text"] = s_
                        if changed:
                            ln.repairs += [f"alt: {', '.join(changed)}"]
                # an upper-case Lari word is an OCR confusion (TOZ for roz): take the raw-image
                # reading of that word when it is lower-case and of the same length
                pw = [strip_marks(t["text"]) for t in ln.toks]
                aw = strip_marks(best).split()
                sm = difflib.SequenceMatcher(None, [w.lower() for w in pw], [w.lower() for w in aw],
                                             autojunk=False)
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op != "replace" or i2 - i1 != j2 - j1:
                        continue
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        t, a = ln.toks[i], aw[j]
                        core = re.sub(r"[^A-Za-z]", "", t["text"])
                        if t["ul"] >= 0.5 and len(core) >= 2 and core.isupper() and \
                                a.islower() and len(a) == len(t["text"]):
                            ln.repairs.append(f"alt: upper-case {t['text']}->{a}")
                            t["text"] = a
            # drop leading junk tokens (gutter marks, bars)
            while ln.toks and JUNK_TOKEN.match(ln.toks[0]["text"]) and len(ln.toks) > 1:
                ln.toks = ln.toks[1:]
            ln.x0 = ln.toks[0]["x0"] if ln.toks else ln.x0
            txt = ln.text.strip(" .,;:|!'")
            if ln.y > 2700 and re.fullmatch(r"\d{1,3}", txt or "x"):
                page_numbers[p] = txt
                continue
            lines.append(ln)
    return lines, page_numbers, noise


# ----------------------------------------------------------------------------- segments

def split_top(text: str, sep: str) -> list[str]:
    """Split at top-level separators (outside (), [] and quotes)."""
    out, cur, depth, q = [], [], 0, False
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch in '"“”' and depth == 0:
            q = not q
        if ch == sep and depth == 0 and not q:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


XREF = re.compile(r"^\s*(?:cf\.|see\b|idem\b)", re.I)


def annotations(rest: str) -> list[tuple[str, str]]:
    """Classify the bracketed/parenthesised/quoted material after a form."""
    out = []
    # a closing bracket lost to OCR (read as ``|``) at the end of the unit
    rest = re.sub(r"\[([^\]]*?)\s*[|!]?\s*$", lambda m: "[" + m.group(1) + "]", rest) \
        if rest.count("[") > rest.count("]") else rest
    for m in re.finditer(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)|\[([^\]]*)\]|[\"“”]([^\"“”]*)[\"“”]?", rest):
        body = (m.group(1) if m.group(1) is not None else m.group(2) if m.group(2) is not None
                else m.group(3)).strip()
        if m.group(3) is not None:
            out.append(("gloss", body))
            continue
        if XREF.match(body) or re.fullmatch(r"[\d\s,.f]+", body):
            out.append(("xref", body))
            continue
        inner_q = re.match(r"^[\"“”'‘’]+(.*?)[\"“”'’!]+\s*(.*)$", body)
        if body.lower().startswith("lit."):
            out.append(("literal", body[4:].strip().strip("\"“”'")))
        elif inner_q:
            out.append(("gloss", inner_q.group(1).strip()))
            if inner_q.group(2):
                tail = inner_q.group(2).strip(" ,")
                parts = [x.strip() for x in tail.split(",")]
                rem = [x for x in parts if x and not XREF.match(x)]
                if rem:
                    out.append(("note", ", ".join(rem)))
                if any(XREF.match(x) for x in parts):
                    out.append(("xref", tail))
        elif body in ("vt.", "vi.", "adv."):
            out.append(("label", body))
        else:
            # qualifier, possibly with an embedded cross-reference
            parts = [x.strip() for x in body.split(",")]
            rem = [x for x in parts if x and not XREF.match(x)]
            if rem:
                out.append(("qualifier", ", ".join(rem)))
            if len(rem) < len(parts):
                out.append(("xref", body))
    return out


def clean_form(s: str) -> tuple[str, list[str]]:
    """Mechanical OCR repairs on a Lari form; returns (form, repairs)."""
    repairs = []
    orig = s
    s = nfc(s.strip())
    s = re.sub(r"[‘’`´]", "'", s)
    # Tesseract doubles a word-initial letter as lower+upper case (sSina, xXaš, Kkowsala) and
    # the apostrophe as two quotes (‘'atam)
    fixed = re.sub(r"(?<![^\s'-])([^\W\d_])(?=([^\W\d_]))",
                   lambda m: "" if m.group(1) != m.group(2) and
                   m.group(1).lower() == m.group(2).lower() else m.group(1), s)
    fixed = re.sub(r"(?<!\S)'{2,}", "'", fixed)
    if fixed != s:
        repairs.append("doubled initial")
        s = fixed
    s = s.replace("_", " ")
    # typewriter letters read as digits inside Lari words (2üd for züd)
    fixed = re.sub(r"(?<=[^\W\d])[2]|[2](?=[^\W\d])", "z", s)
    fixed = re.sub(r"(?<=[^\W\d])0|0(?=[^\W\d])", "o", fixed)
    fixed = re.sub(r"(?<=[^\W\d])1|1(?=[^\W\d])", "l", fixed)
    if fixed != s:
        repairs.append("digit->letter")
        s = fixed
    s = re.sub(r"^[\"“”]", "'", s)  # an opening double quote on an underlined word is ʾ
    s = re.sub(r"(?<=\s)[\"“”](?=\w)", "'", s)
    s = s.strip(" .,;:!?|_—")
    for o, c in ("()", "[]"):
        if s.count(o) != s.count(c):
            s = s.replace(o, "").replace(c, "")
    s = s.strip(" .,;:!?|_—")
    s = re.sub(r"\s+", " ", s)
    low = s.lower()
    if low != s:
        repairs.append("lowercased")
        s = low
    s = nfc(s)
    if s != orig.strip():
        repairs.append("punctuation")
    return s, repairs


def alternates(form: str) -> list[tuple[str, str]]:
    """Expand printed alternation: ``~`` (phonological variant) and ``/`` (interchangeable).

    Returns [(form, kind)] with the first item the main form (kind '')."""
    forms = [(form, "")]
    if "~" in form:
        parts = [p.strip() for p in form.split("~")]
        left = parts[0].split()
        out = [(" ".join(left), "")]
        for r in parts[1:]:
            rw = r.split()
            if not rw:
                continue
            k = len(rw)
            out.append((" ".join(left[:-k] + rw) if k < len(left) else " ".join(rw), "sound-variant"))
        forms = out
    final = []
    for f, kind in forms:
        words = f.split(" ")
        if any("/" in w for w in words):
            opts = [w.split("/") for w in words]
            combos = [[]]
            for o in opts:
                combos = [c + [x] for c in combos for x in (o if o != [""] else [""]) if x != ""] or combos
            for j, c in enumerate(combos[:4]):
                final.append((" ".join(c), kind if j == 0 else "alternate"))
        else:
            final.append((f, kind))
    seen, out = set(), []
    for f, k in final:
        f = f.strip()
        if f and f not in seen:
            seen.add(f)
            out.append((f, k))
    return out


def clean_gloss(g: str) -> tuple[str, list[str]]:
    tags = []
    g = g.strip()
    g = re.sub(r"^[\"“”'‘’!]+|[\"“”'‘’!]+$", "", g).strip()
    for lab, t in LABELS.items():
        pat = r"\(" + re.escape(lab) + r"\)"
        if re.search(pat, g):
            g = re.sub(r"\s*" + pat, "", g).strip()
            tags.extend(t)
    # cross-references are locators, not meaning
    g = re.sub(r"\s*\[(?:see\b[^\]]*|\d{4}[^\]]*)\]", "", g)
    g = re.sub(r"\s*\((?:cf\.|see\b)[^)]*\)", "", g)
    g = re.sub(r"\s*\(\d{4}f?\.?\)", "", g)
    g = re.sub(r"\s+", " ", g).strip()
    return g, tags


def is_sentence(lari: str, gloss: str) -> bool:
    if re.search(r"[.?!]\s*$", lari.strip()) or re.search(r"\w[.?!]\s+\S", lari):
        return True
    g = gloss.strip().strip("\"“”'‘’!")
    if g.lower().startswith("idem"):
        return True
    if g[:1].isupper() and len(lari.split()) >= 2:
        return True  # a phrase with a proper-noun or sentence translation (usage example)
    return bool(g[:1].isupper() and re.search(r"[.?!]$", g) and not g.endswith("etc."))


# ----------------------------------------------------------------------------- parsing

def concept_gloss(english: str) -> tuple[str, str, list[str]]:
    """Gloss, residual note and tags from the English questionnaire entry."""
    tags = []
    note = []
    words = english.split()
    keep = []
    for w in words:
        if w in LABELS:
            tags.extend(LABELS[w])
        else:
            keep.append(w)
    e = " ".join(keep).strip(" |—-,;:'‘’\"“”")
    for lab, t in LABELS.items():
        pat = r"\s*\(" + re.escape(lab) + r"\)"
        if re.search(pat, e):
            e = re.sub(pat, "", e)
            tags.extend(t)
    e = re.sub(r"\s*\((?:cf\.|see\b)[^)]*\)", "", e).strip()
    if e.endswith(".") and not re.search(r"\b(?:etc|[a-z])\.$", e):
        e = e[:-1].rstrip()
    bracket = re.findall(r"\[([^\]]*)\]", e)
    e = re.sub(r"\s*\[[^\]]*\]", "", e).strip()
    m = re.match(r"^([^/]+?)/(\S.*)$", e)
    if m and not re.search(r"/\s*$", e):
        # a French or other equivalent specifying the English entry (Abbreviations, p. xvii)
        head, equiv = m.group(1).strip(), m.group(2).strip()
        rest = ""
        mm = re.match(r"^(\S+(?:\s(?:de|du|peu|à)\b\s?\S*)*)(.*)$", equiv)
        if mm:
            equiv, rest = mm.group(1), mm.group(2)
        e = (head + rest).strip()
        note.append(f"questionnaire equivalent: {equiv}")
    for b in bracket:
        if XREF.match(b):
            continue
        e = f"{e} ({b})"
    return e, "; ".join(note), tags


def parse(pages: dict[int, dict]):
    lines, page_numbers, _ = page_lines(pages)

    # ---- page-level column positions
    lari_col = {}
    for p in pages:
        xs = []
        for ln in lines:
            if ln.page != p:
                continue
            if ln.toks and number_candidates(ln.toks[0]["text"]):
                ul = [t["x0"] for t in ln.toks[1:] if t["ul"] >= 0.5]
                if ul:
                    xs.append(min(ul))
        lari_col[p] = sorted(xs)[len(xs) // 2] if xs else 800

    # ---- head lines and entry numbers (sequence-checked)
    expected = 1
    heads: dict[int, Line] = {}
    head_info: dict[int, str] = {}
    kinds: dict[str, str] = {}
    order = {ln.uid: i for i, ln in enumerate(lines)}
    for ln in lines:
        if not ln.toks or ln.x0 > 600:
            continue
        cands = number_candidates(ln.toks[0]["text"])
        if not cands and len(ln.toks) > 1 and len(ln.toks[0]["text"]) <= 2:
            # a stray mark before the number ("_ 0067.", "' 0863.", "B 0916.")
            c2 = number_candidates(ln.toks[1]["text"])
            if c2:
                ln.toks = ln.toks[1:]
                cands = c2
        if not cands:
            continue
        hit = next((c for c in cands if expected <= c <= expected + 3), None)
        if hit is None:
            continue
        heads[hit] = ln
        kinds[ln.uid] = "head"
        ln.number = hit
        if f"{hit:04d}" not in ln.toks[0]["text"]:
            head_info[hit] = f"number read as {ln.toks[0]['text']!r}"
        expected = hit + 1
    # an illegible number between two read ones: the line in the number column between them
    for n in range(2, ENTRY_COUNT + 1):
        if n in heads or (n - 1) not in heads:
            continue
        prev = heads[n - 1]
        nxt = next((heads[m] for m in range(n + 1, n + 4) if m in heads), None)
        if nxt is None:
            continue
        between = [ln for ln in lines[order[prev.uid] + 1:order[nxt.uid]]
                   if ln.toks and ln.x0 <= prev.x0 + 60 and kinds.get(ln.uid) != "head"]
        if len(between) == 1 and (nxt.number == n + 1):
            ln = between[0]
            heads[n] = ln
            kinds[ln.uid] = "head"
            ln.number = n
            head_info[n] = (f"number illegible ({ln.toks[0]['text']!r}); "
                            "inferred from the sequence")
    missing_numbers = [n for n in range(1, ENTRY_COUNT + 1) if n not in heads]

    # ---- assign the other lines to entries, merge continuations, drop noise
    units = []  # (entry number, Line, kind)
    current = None
    prev_unit: Line | None = None
    noise = []
    for ln in lines:
        if kinds.get(ln.uid) == "head":
            current = ln.number
            units.append([current, ln, "head"])
            prev_unit = ln
            continue
        if current is None:
            noise.append((ln, "before first entry"))
            continue
        last = prev_unit.cont[-1] if prev_unit.cont else prev_unit
        prev_text = " ".join(t["text"] for t in last.toks)
        open_gloss = gloss_open(last, prev_unit)
        hyphen = prev_text.rstrip().endswith("-")
        # continuation lines are indented under the line they continue
        ref_x = prev_unit.x0 if prev_unit is not last else last.x0
        if kinds.get(prev_unit.uid) == "head":
            ref_x = prev_unit.x0 + 140
        indented = ln.x0 >= ref_x + 45
        near = ln.y - last.y < 110
        far = ln.x0 > 600
        lari_like = any(t["ul"] >= 0.5 for t in ln.toks)
        letters = sum(c.isalpha() for c in ln.text)
        if letters == 0 or re.fullmatch(r"[\W\d_]*\d[\W\d_]*", ln.text):
            noise.append((ln, "no letters (folio fragment or mark)"))
            continue
        if far and not (near and (open_gloss or hyphen) and ln.conf >= 60) and not (
                lari_like and ln.conf >= 70 and near and len(ln.toks) <= 4):
            noise.append((ln, "outside text columns (drawing or margin)"))
            continue
        if ln.conf < 35 and not (open_gloss and indented):
            noise.append((ln, "low-confidence OCR noise"))
            continue
        lari_wrap = (kinds.get(prev_unit.uid) != "head" and not split_sub(prev_unit)[1]
                     and not re.search(r"[.?!\"”:]$", prev_text.strip())
                     and sum(t["ul"] >= 0.5 for t in prev_unit.toks) * 2 > len(prev_unit.toks))
        first = ln.toks[0]
        gloss_line = first["ul"] < 0.5 and (
            (first["text"].startswith(DQ) and first["text"].lstrip("\"“”'‘’")[:1].isupper())
            or first["text"][:1] == "(")
        utext = " ".join(t["text"] for t in prev_unit.all_toks())
        open_note = utext.startswith("[") and utext.count("[") > utext.count("]")
        if near and (gloss_line or open_note or
                     (indented and (open_gloss or hyphen or far or lari_wrap))):
            prev_unit.cont.append(ln)
            continue
        units.append([current, ln, "sub"])
        prev_unit = ln

    return lines, heads, units, noise, missing_numbers, head_info, page_numbers, lari_col


def unit_text(ln: Line) -> str:
    txt = ln.text
    for c in ln.cont:
        if txt.rstrip().endswith("-") and not txt.rstrip().endswith(" -"):
            txt = txt.rstrip()[:-1] + c.text.lstrip("’'‘ -")
        else:
            txt = txt + " " + c.text
    return txt


def split_head(ln: Line, col: int) -> tuple[str, str, list[dict]]:
    toks = [t for t in ln.all_toks()[1:] if not HEAD_JUNK.match(t["text"])]
    any_ul = any(t["ul"] >= 0.5 for t in ln.toks[1:])
    start = None
    depth = 0
    for i, t in enumerate(toks):
        s = t["text"]
        gap = t["x0"] - toks[i - 1]["x1"] if i else t["x0"] - ln.toks[0]["x1"]
        big_gap = gap >= 60
        in_col = (t["x0"] >= col - 40 and (big_gap or not any_ul)) or (i > 0 and big_gap)
        if depth == 0 and s[:1] in "([":
            if in_col:
                start = i  # the Lari column holds only a comment, e.g. "(nonexistent)"
                break
        elif depth == 0 and any(t is u for u in ln.toks) and (t["ul"] >= 0.5 or in_col):
            start = i
            break
        depth += s.count("(") + s.count("[") - s.count(")") - s.count("]")
        depth = max(depth, 0)
    if start is None:
        # no Lari on the head line; any continuation tokens are English
        return " ".join(t["text"] for t in toks), "", [], toks
    eng = " ".join(t["text"] for t in toks[:start])
    lari_toks = toks[start:]
    # drop low-confidence non-underlined debris after the last underlined token (drawings)
    last_ul = max((j for j, t in enumerate(lari_toks) if t["ul"] >= 0.5), default=0)
    lari_toks = [t for j, t in enumerate(lari_toks)
                 if not (j > 0 and t["ul"] < 0.5 and (
                     re.search(r"[\\=°{}<>@#*+^%$&|]", t["text"])
                     or re.fullmatch(r"[A-Z]{2,}[^a-z]*", t["text"])
                     or (j > last_ul and t["conf"] < 70
                         and not re.search(r"[(\[\])\"“”]", t["text"]))))]
    words = []
    depth = 0
    for t in lari_toks:
        s = t["text"]
        if depth == 0 and s[:1] in GLOTTAL_CHARS + "\"“”" and s[1:2].isalpha() and \
                not s[1:2].isupper():
            s = "'" + s[1:]  # glottal stop read as a quotation mark
        depth += s.count("(") + s.count("[") - s.count(")") - s.count("]")
        words.append(s)
    lari = " ".join(words)
    return eng, lari, lari_toks, toks[:start]


DQ = ('"', "“", "”", "''", "‘‘", "’’", "'\"", "\"'", "‘'", "'‘")


def split_sub(ln: Line) -> tuple[str, str]:
    """Lari part and gloss part of a sub-line.  A gloss opens at a double quotation mark that
    is not the misread glottal stop of an underlined word, at a single quotation mark before a
    capital, or at a single quotation mark on a non-underlined word when the unit ends in a
    closing quotation mark (OCR often reads the opening ``"`` as ``‘``)."""
    toks = ln.all_toks()
    closes = any(CLOSER.search(t["text"]) for t in toks[1:])
    k = None
    for i, t in enumerate(toks):
        if i == 0:
            continue
        s = t["text"]
        prev = toks[i - 1]["text"]
        ul = t["ul"] >= 0.5
        if s[:1] in "([" and not ul:
            k = i
            break
        if s.startswith(DQ):
            later_dq = any(u["text"].startswith(DQ) for u in toks[i + 1:])
            if not ul or not later_dq or re.search(r"[.?!,]$", prev) or s[1:2].isupper() or \
                    (len(s) > 2 and s[2:3].isupper()):
                k = i
                break
            continue
        if s[:1] in "'‘’`" and not ul:
            # the glottal stop only precedes a vowel; a quote before a consonant, a capital or a
            # second quote opens the gloss, and so does a vowel-initial word ('a glass,
            # 'one egg) when no proper double-quoted gloss follows on the unit
            if s[1:2].isupper() or (closes and (s[1:2] not in "aeiouāüáéíóú" or
                                                not any(u["text"].startswith(DQ)
                                                        for u in toks[i + 1:]))):
                k = i
                break
    if k is None and closes and toks[0]["ul"] >= 0.5:
        # the opening quotation mark was lost: the gloss starts at the first word that is not
        # underlined after the underlined Lari
        for i, t in enumerate(toks):
            if i and t["ul"] < 0.5 and toks[i - 1]["ul"] >= 0.5 and \
                    not re.search(r"[.?!]$", toks[i - 1]["text"]):
                k = i
                break
    if k is None:
        return " ".join(t["text"] for t in toks), ""
    return " ".join(t["text"] for t in toks[:k]), " ".join(t["text"] for t in toks[k:])


def first_gloss(rest: str) -> tuple[str, str]:
    """The quoted gloss at the start of ``rest`` and whatever follows it."""
    r = rest.strip()
    r = r.replace("''", '"').replace("‘‘", '"').replace("’’", '"')
    r = re.sub(r"^['‘’`][\"'‘’!]?", '"', r) if r[:1] in "'‘’`" else r
    if r[:1] in "\"“”":
        body = r[1:]
        # the closing quotation mark is the first one outside parentheses/brackets; OCR also
        # reads it as ``!``.  Whatever follows is kept for annotation parsing (and the audit).
        depth = 0
        for i, ch in enumerate(body):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth = max(0, depth - 1)
            elif depth == 0 and (ch in "\"“”" or (ch == "!" and (i + 1 == len(body)
                                                                or body[i + 1] in " \"'"))):
                rest = body[i + 1:].lstrip("\"“”'!")
                return body[:i], rest
        m = re.match(r"^(.*?)(\s*[(\[].*)$", body)
        if m and not re.search(r"[)\]]", m.group(1)):
            return m.group(1), m.group(2)
        return body, ""
    if r and r[:1] not in "([" and re.search(r"[\"”'’!]\s*$", r):
        return re.sub(r"[\"”'’!]+\s*$", "", r), ""
    return "", r


def mark_log(toks: list[dict]) -> Counter:
    c = Counter()
    for t in toks:
        for e in t["marks"]:
            if e["status"] != "i-dot":
                c[e["status"]] += 1
    return c


def load_corrections() -> dict[str, dict[str, str]]:
    """Hand decision table: OCR readings corrected against the page image, keyed by audit
    Unit_ID and field (``English``/``Lari`` on head lines, ``Lari``/``Gloss`` on sub-lines)."""
    out: dict[str, dict[str, str]] = {}
    if not CORRECTIONS.exists():
        return out
    with CORRECTIONS.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["Unit_ID"], {})[r["Field"]] = nfc(r["Corrected"])
    return out


def alt_lines(page: dict) -> list[str]:
    return [l.strip() for l in page["alt"].splitlines() if l.strip()]


def alt_match(text: str, alts: list[str]) -> tuple[str, float]:
    base = strip_marks(text).lower()
    best, score = "", 0.0
    for a in alts:
        r = difflib.SequenceMatcher(None, base, strip_marks(a).lower()).ratio()
        if r > score:
            best, score = a, r
    return best, round(score, 3)


def alt_english(n: int, alts: list[str], lari: str) -> str | None:
    """The English questionnaire entry of entry ``n`` in the raw-image reading: the text after
    the entry number up to the first Lari word."""
    first = strip_marks(lari.split()[0]).lower().strip(",.;'‘’\"") if lari.split() else ""
    for a in alts:
        m = re.match(r"^\W*[0-9OoIl|J)\]]?(\d{3})[.,]\s+(.*)$", a)
        if not m or int(m.group(1)) != n % 1000:
            continue
        words = m.group(2).split()
        out = []
        for w in words:
            ws = strip_marks(w).lower().strip(",.;'‘’\"")
            if first and (ws == first or (len(first) > 2 and ws.startswith(first[:3]))):
                break
            out.append(w)
        else:
            if first:
                return None
        return " ".join(out) or None
    return None


def persian_for(pages, heads, n) -> str:
    ln = heads[n]
    nxt = heads.get(n + 1)
    out = []
    for p in range(ln.page, (nxt.page if nxt else ln.page) + 1):
        y0 = ln.y - 40 if p == ln.page else 0
        y1 = nxt.y - 40 if nxt and p == nxt.page else 10**6
        for pl in layout.persian_lines(pages[p]):
            if y0 <= pl["y"] < y1 and pl["y"] < 2750:
                out.append(pl["text"])
    return " | ".join(out)


def build():
    pages = load_pages()
    (lines, heads, units, noise, missing_numbers, head_info, page_numbers,
     lari_col) = parse(pages)
    corrections = load_corrections()
    used_corrections = set()
    review = load_sample_review()
    # units whose reading was checked against the page image carry no OCR review marker
    verified = set(corrections) | {u for u, r in review.items() if r.get("Material_Error") == "no"}
    alts = {p: alt_lines(d) for p, d in pages.items()}

    forms: list[dict] = []
    audit: list[dict] = []
    concept_of: dict[int, str] = {}

    def emit(n, ln, item_key, form_text, gloss, notes, tags, variant_of=""):
        key = f"{SOURCE}:{n:04d}:{item_key}"
        gloss = re.sub(r"\s*\[(?:see\b[^\]]*|\d{4}[^\]]*)\]|\s*\((?:cf\.|see\b)[^)]*\)"
                       r"|\s*\(\d{4}f?\.?\)", "", gloss).strip()
        # English words hyphenated at a line break (cor- responds)
        gloss = re.sub(r"(?<=[a-z])- (?=[a-z])", "", gloss)
        notes = re.sub(r"(?<=[a-z])- (?=[a-z])", "", notes)
        row = {
            "Language_ID": LANGUAGE, "Parameter_ID": "", "Form": nfc(form_text),
            "Gloss": gloss, "Native": "", "Phonemic": "", "Notes": notes,
            "Source": f"{SOURCE}[p. {ln.page - PAGE_OFFSET}, entry {n:04d}]",
            "Cognateset": "", "Etymology": "", "Entry_Key": key,
            "Variant_Of_Key": variant_of, "Borrowed_From_Key": "",
            "Derivation_Parent_Keys": "",
            "Tags": " ".join(dict.fromkeys(
                [*tags, *([] if ln.uid in verified else [REVIEW_TAG]), DIALECT_TAG])),
        }
        forms.append(row)
        return key

    def emit_item(n, ln, base_key, raw_form, gloss, notes, tags):
        """One printed form with its alternates; returns (keys, unparseable reasons)."""
        form, repairs = clean_form(raw_form)
        keys, problems = [], []
        alts_ = alternates(form)
        main_key = None
        for j, (f, kind) in enumerate(alts_, start=1):
            bad = sorted({c for c in f if c not in LARI_ALLOWED and not ud.combining(c)})
            if bad or not re.search(r"[a-zāüšž]", f):
                problems.append(f"characters outside the Lari transcription: {''.join(bad) or f!r}")
                continue
            if main_key is None:
                main_key = emit(n, ln, base_key, f, gloss, notes, tags)
                keys.append(main_key)
            else:
                k = emit(n, ln, f"{base_key}:v{j}", f, gloss, notes,
                         [*tags, kind or "alternate"], variant_of=main_key)
                keys.append(k)
        return keys, problems, repairs

    entry_rows = {}
    for n, ln, kind in units:
        toks = ln.all_toks()
        raw = unit_text(ln)
        marks = mark_log(toks)
        a_text, a_score = alt_match(raw, alts[ln.page])
        base = {
            "Unit_ID": ln.uid, "PDF_Page": ln.page, "Printed_Page": ln.page - PAGE_OFFSET,
            "Entry": f"{n:04d}", "Kind": kind, "Raw_OCR": raw,
            "Line_Box": f"{ln.x0},{ln.y},{max(t['x1'] for t in toks)},"
                        f"{max(t['y1'] for t in toks)}",
            "Raw_OCR_Undiacritised": " ".join(
                [ln.raw_text] + [c.raw_text for c in ln.cont]),
            "Continuation_Units": " ".join(c.uid for c in ln.cont),
            "Apostrophe_Repairs": "; ".join(ln.repairs + [r for c in ln.cont for r in c.repairs]),
            "Alt_OCR": a_text, "Alt_Similarity": a_score,
            "OCR_Confidence": round(ln.conf, 1),
            "Underlined_Tokens": sum(1 for t in toks if t["ul"] >= 0.5),
            "Marks_Placed": marks["placed"] + marks["placed-shifted"],
            "Marks_Unresolved": marks["unplaced"] + marks["unclassified"] + marks["stray-dot"]
            + marks["single-dot-over-u"],
            "Persian_OCR": "", "English": "", "Lari": "", "Gloss": "", "Annotations": "",
            "Status": "", "Reason": "", "Emitted_Keys": "", "Review": "ocr:unreviewed",
            "Language_ID": LANGUAGE, "Dialect_Tag": DIALECT_TAG,
        }
        if kind == "head":
            base["Persian_OCR"] = persian_for(pages, heads, n)
            eng, lari, _, eng_toks = split_head(ln, lari_col[ln.page])
            corr = corrections.get(ln.uid, {})
            alt_eng = alt_english(n, alts[ln.page], lari)
            base["Alt_English"] = alt_eng or ""
            sim = difflib.SequenceMatcher(None, strip_marks(eng).lower().strip(),
                                          strip_marks(alt_eng or "").lower().strip()).ratio()
            if alt_eng and sim < 0.8 and (any(t["conf"] < 60 for t in eng_toks)
                                          or not eng.strip()):
                base["Reason"] += f"English entry from the raw-image reading (cleaned: {eng!r}); "
                eng = alt_eng
            if corr:
                used_corrections.add(ln.uid)
                eng = corr.get("English", eng)
                lari = corr.get("Lari", lari)
                base["Reason"] += "OCR corrected against the page image; "
                base["Review"] = "ocr:image-corrected"
            gloss, note, ltags = concept_gloss(eng)
            concept_of[n] = gloss
            base["English"] = eng
            base["Lari"] = lari
            base["Gloss"] = gloss
            if head_info.get(n):
                base["Reason"] = head_info[n] + "; " + base["Reason"]
            emitted, problems = [], []
            if not lari.strip() or lari.strip()[:1] in "([":
                if re.match(r"\s*[\[(]\s*see\b", lari, re.I):
                    base["Annotations"] = lari
                    base["Status"] = "cross_reference_only"
                    base["Reason"] += "head lists only a cross-reference"
                elif lari.strip():
                    base["Annotations"] = lari
                    base["Status"] = "no_head_form"
                    base["Reason"] += "the Lari column holds only a comment"
                else:
                    base["Status"] = "no_head_form"
                    base["Reason"] += ("no Lari on the head line (blank in source, or forms "
                                       "only in sub-entries/sentences)")
                audit.append(base)
                continue
            senses = [s.strip() for s in gloss.split(";")]
            groups = split_top(lari, ";")
            paired = len(groups) > 1 and len(groups) == len(senses)
            item_no = 0
            ann_all = []
            for gi, grp in enumerate(groups):
                for item in split_top(grp, ","):
                    if not item.strip():
                        continue
                    m = re.match(r"^([^(\[\"“”]*)(.*)$", item.strip())
                    form_part, rest = m.group(1), m.group(2)
                    if not form_part.strip():
                        ann_all.append(rest)
                        continue
                    anns = annotations(rest)
                    ann_all.extend(f"{k}:{v}" for k, v in anns)
                    g = senses[gi] if paired else gloss
                    notes = [note] if note else []
                    tags = list(ltags)
                    own = [v for k, v in anns if k == "gloss"]
                    quals = [v for k, v in anns if k == "qualifier"]
                    if own:
                        g, gt = clean_gloss(own[0])
                        tags.extend(gt)
                        notes.append(f"under questionnaire entry '{gloss}'")
                    for q in quals:
                        if len(q.split()) <= 4:
                            g = f"{g} ({q})"
                        else:
                            notes.append(q)
                    for k, v in anns:
                        if k == "literal":
                            notes.append(f"lit. '{v}'")
                        elif k == "note":
                            notes.append(v)
                        elif k == "label":
                            tags.extend(LABELS[v])
                    item_no += 1
                    keys, probs, reps = emit_item(n, ln, str(item_no), form_part, g,
                                                  "; ".join(x for x in notes if x), tags)
                    emitted += keys
                    problems += probs
            base["Annotations"] = " | ".join(ann_all)
            base["Emitted_Keys"] = " ".join(emitted)
            if emitted:
                base["Status"] = "installed"
                base["Reason"] += "head forms" + (f"; dropped: {'; '.join(problems)}"
                                                  if problems else "")
            else:
                base["Status"] = "ocr_unparseable"
                base["Reason"] += "; ".join(problems) or "no parsable form"
            audit.append(base)
            entry_rows[n] = emitted
            continue

        # ---- sub-line
        if ln.toks[0]["text"].startswith("[") and ln.toks[0]["ul"] < 0.5:
            base["Status"] = "editorial_note"
            base["Reason"] = "bracketed authors' note (not a form)"
            audit.append(base)
            continue
        lari, rest = split_sub(ln)
        gloss_raw, after = first_gloss(rest)
        corr = corrections.get(ln.uid, {})
        if corr:
            used_corrections.add(ln.uid)
            lari = corr.get("Lari", lari)
            gloss_raw = corr.get("Gloss", gloss_raw)
            base["Review"] = "ocr:image-corrected"
        anns = annotations(after) if after else []
        base["Lari"] = lari
        base["Gloss"] = gloss_raw
        base["Annotations"] = " | ".join(f"{k}:{v}" for k, v in anns)
        if is_sentence(lari, gloss_raw):
            base["Status"] = "example_sentence"
            base["Reason"] = "Lari example sentence with translation (not a lexical form)"
            audit.append(base)
            continue
        if not re.search(r"[a-zA-Zāüšž]", lari):
            base["Status"] = "ocr_unparseable"
            base["Reason"] = "no Lari text before the gloss"
            audit.append(base)
            continue
        if not gloss_raw:
            own = [v for k, v in anns if k == "gloss"]
            if own:
                gloss_raw = own[0]
                anns = [(k, v) for k, v in anns if k != "gloss"]
        g, gtags = clean_gloss(gloss_raw)
        notes = []
        quals = [v for k, v in anns if k == "qualifier"]
        if not g and quals and concept_of.get(n):
            # a bracketed description instead of a quoted gloss: a kind of the concept
            g = f"{concept_of[n]} ({'; '.join(quals)})"
            quals = []
        if not g:
            notes.append(f"listed without gloss under questionnaire entry {n:04d} "
                         f"'{concept_of.get(n, '')}'")
        if quals and g:
            g = f"{g} ({'; '.join(quals)})"
        for k, v in anns:
            if k == "literal":
                notes.append(f"lit. '{v}'")
            elif k == "note":
                notes.append(v)
            elif k == "label":
                gtags.extend(LABELS[v])
        sub_no = sum(1 for r in audit if r["Entry"] == f"{n:04d}" and r["Kind"] == "sub"
                     and r["Status"] == "installed") + 1
        lari_toks = ln.all_toks()[:len(lari.split())]
        lconf = sum(t["conf"] for t in lari_toks) / max(1, len(lari_toks))
        if not g and not corr and (any(t["ul"] < 0.5 for t in lari_toks) or lconf < 75 or
                                   sum(c.isalpha() for c in lari) < 3):
            base["Status"] = "ocr_noise"
            base["Reason"] = "unglossed fragment that is not wholly underlined Lari or is " \
                             "low-confidence (drawing, margin or broken line)"
            audit.append(base)
            continue
        keys, probs = [], []
        items = [x for x in split_top(lari, ",") if x.strip()]
        for ii, item in enumerate(items, start=1):
            sk = f"sub:{sub_no}" if len(items) == 1 else f"sub:{sub_no}:{ii}"
            k_, p_, _ = emit_item(n, ln, sk, item, g, "; ".join(notes), gtags)
            keys += k_
            probs += p_
        base["Emitted_Keys"] = " ".join(keys)
        if keys:
            base["Status"] = "installed"
            base["Reason"] = "sub-form with its own printed gloss" if g else \
                "sub-form listed without gloss"
            if probs:
                base["Reason"] += "; dropped: " + "; ".join(probs)
        else:
            base["Status"] = "ocr_unparseable"
            base["Reason"] = "; ".join(probs)
        audit.append(base)

    for ln, why in noise:
        audit.append({
            "Unit_ID": ln.uid, "PDF_Page": ln.page, "Printed_Page": ln.page - PAGE_OFFSET,
            "Entry": "", "Kind": "noise", "Raw_OCR": ln.text,
            "Raw_OCR_Undiacritised": ln.raw_text, "Continuation_Units": "", "Alt_OCR": "",
            "Alt_Similarity": "", "OCR_Confidence": round(ln.conf, 1),
            "Underlined_Tokens": sum(1 for t in ln.toks if t["ul"] >= 0.5),
            "Marks_Placed": "", "Marks_Unresolved": "", "Persian_OCR": "", "English": "",
            "Lari": "", "Gloss": "", "Annotations": "", "Status": "ocr_noise", "Reason": why,
            "Emitted_Keys": "", "Review": "", "Language_ID": "", "Dialect_Tag": "",
        })
    for p, txt in sorted(page_numbers.items()):
        ok = txt == str(p - PAGE_OFFSET)
        audit.append({
            "Unit_ID": f"p{p:03d}:folio", "PDF_Page": p, "Printed_Page": p - PAGE_OFFSET,
            "Entry": "", "Kind": "folio", "Raw_OCR": txt, "Raw_OCR_Undiacritised": txt,
            "Continuation_Units": "", "Alt_OCR": "", "Alt_Similarity": "", "OCR_Confidence": "",
            "Underlined_Tokens": "", "Marks_Placed": "", "Marks_Unresolved": "",
            "Persian_OCR": "", "English": "", "Lari": "", "Gloss": "", "Annotations": "",
            "Status": "page_number", "Reason": "printed folio agrees" if ok else
            "folio OCR disagrees with PDF-9 offset", "Emitted_Keys": "", "Review": "",
            "Language_ID": "", "Dialect_Tag": "",
        })
    audit.sort(key=lambda r: (r["PDF_Page"], r["Unit_ID"]))
    # review state from the seeded image audit
    reviewed = load_sample_review()
    for r in audit:
        if r["Unit_ID"] in reviewed and r["Review"] == "ocr:unreviewed":
            r["Review"] = "ocr:image-checked"
    unused = sorted(set(corrections) - used_corrections)
    if unused:
        sys.exit(f"corrections.csv units not found: {unused}")
    return {
        "pages": pages, "forms": forms, "audit": audit, "heads": heads,
        "missing_numbers": missing_numbers, "page_numbers": page_numbers,
        "corrections": corrections,
    }


def load_sample_review() -> dict[str, dict]:
    if not SAMPLE_REVIEW.exists():
        return {}
    with SAMPLE_REVIEW.open(encoding="utf-8", newline="") as f:
        return {r["Unit_ID"]: r for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------- outputs

FORM_COLUMNS = ["Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
                "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
                "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]
AUDIT_COLUMNS = ["Unit_ID", "PDF_Page", "Printed_Page", "Line_Box", "Entry", "Kind", "Status",
                 "Reason",
                 "Raw_OCR", "Raw_OCR_Undiacritised", "Continuation_Units", "Alt_OCR",
                 "Apostrophe_Repairs", "Alt_English",
                 "Alt_Similarity", "OCR_Confidence", "Underlined_Tokens", "Marks_Placed",
                 "Marks_Unresolved", "English", "Lari", "Gloss", "Annotations", "Persian_OCR",
                 "Emitted_Keys", "Language_ID", "Dialect_Tag", "Review"]


def sample_units(audit: list[dict], k: int = 20, seed: int = SEED) -> list[dict]:
    """Seeded sample of installed head/sub units spread over the whole vocabulary."""
    pool = [r for r in audit if r["Status"] == "installed"]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, k), key=lambda r: r["Unit_ID"])


def csv_text(rows, columns, header=True) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if header:
        w.writerow(columns)
    for r in rows:
        w.writerow([r.get(c, "") for c in columns])
    return buf.getvalue()


def write_outputs(res: dict) -> dict:
    forms, audit = res["forms"], res["audit"]
    form_text = csv_text(forms, FORM_COLUMNS, header=False)
    CSV_OUT.write_text(form_text, encoding="utf-8")
    AUDIT_OUT.write_text(csv_text(audit, AUDIT_COLUMNS), encoding="utf-8")
    review = load_sample_review()
    sample = sample_units(audit, seed=SAMPLE_SEED)
    cols = ["Unit_ID", "PDF_Page", "Printed_Page", "Entry", "Kind", "Raw_OCR", "Lari", "Gloss",
            "Emitted_Keys", "Image_Reading", "Material_Error", "Error_Class", "Note"]
    rows = []
    for r in sample:
        rv = review.get(r["Unit_ID"], {})
        rows.append({**r, "Image_Reading": rv.get("Image_Reading", ""),
                     "Material_Error": rv.get("Material_Error", ""),
                     "Error_Class": rv.get("Error_Class", ""), "Note": rv.get("Note", "")})
    SAMPLE_OUT.write_text(csv_text(rows, cols), encoding="utf-8")
    manifest = make_manifest(res, form_text)
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return manifest


def make_manifest(res: dict, form_text: str) -> dict:
    audit = res["audit"]
    st = Counter(r["Status"] for r in audit)
    forms = res["forms"]
    return {
        "bibliography": "Kamioka, Kōji & Minoru Yamada. 1979. Lārestāni Studies 1: Lari Basic "
                        "Vocabulary. Tokyo: Institute for the Study of Languages and Cultures of "
                        "Asia and Africa (ILCAA), Tokyo University of Foreign Studies.",
        "scan": {"file": ocr_runner.PDF_PATH.name, "sha256": ocr_runner.PDF_SHA256,
                 "pdf_pages": ocr_runner.PDF_PAGES, "redistributed": False,
                 "format": "image-only CCITT 300 dpi, /Rotate 270, each PDF page a two-page "
                           "spread (left: Persian prompts; right: English/Lari)"},
        "rights": "Copyrighted; scan not checked in. Extracted lexical facts with locators only.",
        "page_ranges": {"front_matter_pdf": [1, 9], "vocabulary_pdf": [FIRST_PDF_PAGE,
                                                                       LAST_PDF_PAGE],
                        "vocabulary_printed": [1, 177], "printed_offset": PAGE_OFFSET,
                        "excluded_indexes_pdf": [187, 232],
                        "excluded": "Lari, English and Persian indexes (printed pp. 179–227+) "
                                    "and the Persian left-hand pages (OCR kept in the audit only)"},
        "ocr": {"runner": "data/other/forms/raw_data/kamioka_lari_1979/ocr_runner.py",
                "params": ocr_runner.PARAMS, "params_hash": ocr_runner.PARAMS_HASH,
                "marks_version": ocr_runner.MARKS_VERSION, "alt_pass": ocr_runner.ALT_CMD,
                "pitch_px": PITCH},
        "counts": {
            "entries_expected": ENTRY_COUNT,
            "entries_found": len(res["heads"]),
            "missing_entry_numbers": res["missing_numbers"],
            "audit_units": len(audit),
            "audit_status": dict(sorted(st.items())),
            "installed_rows": len(forms),
            "variant_rows": sum(1 for f in forms if f["Variant_Of_Key"]),
            "folios_read": len(res["page_numbers"]),
            "folios_disagreeing": sum(1 for p, t in res["page_numbers"].items()
                                      if t != str(p - PAGE_OFFSET)),
            "corrections_applied": sum(len(v) for v in res["corrections"].values()),
        },
        "outputs": {"forms": "data/other/forms/20261025-kamioka-lari.csv",
                    "audit": "data/other/forms/raw_data/kamioka_lari_1979/audit.csv",
                    "sample": "data/other/forms/raw_data/kamioka_lari_1979/sample.csv",
                    "forms_sha256": hashlib.sha256(form_text.encode()).hexdigest()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--install", action="store_true", help="write the CSV/audit/sample/manifest")
    args = ap.parse_args(argv)
    res = build()
    st = Counter(r["Status"] for r in res["audit"])
    print(f"entries {len(res['heads'])}/{ENTRY_COUNT}; missing {res['missing_numbers']}")
    print(f"audit units {len(res['audit'])}: {dict(sorted(st.items()))}")
    print(f"installed rows {len(res['forms'])}")
    if args.install:
        m = write_outputs(res)
        print("wrote", CSV_OUT.relative_to(ROOT), "sha256", m["outputs"]["forms_sha256"][:12])
    else:
        print("dry run: nothing written (use --install)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
