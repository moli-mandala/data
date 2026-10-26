#!/usr/bin/env python3
"""Reproducible OCR runner for Kamioka & Yamada 1979, *Lari Basic Vocabulary*.

The scan is image-only (CCITT, 300 dpi); every PDF page is a two-page spread whose left page
carries the Persian medium-language prompts and whose right page carries the numbered English
concept, the underlined typewritten Lari forms, sub-forms and example sentences.

Typewritten diacritics are overstruck marks (macron ``ā``, diaeresis ``ü``, caron ``š ž``, acute
stress) that Tesseract reads unreliably.  The runner therefore separates the two problems:

1. underlines (which identify Lari text) are detected as long thin horizontal runs, recorded,
   and erased;
2. overstrike marks are detected geometrically as small connected components sitting directly
   above a letter, classified by shape, recorded with their position, and erased (single dots,
   i.e. the dots of ``i``/``j``, are kept);
3. the cleaned right page is read by Tesseract (``eng``, ``--psm 6``, TSV word boxes); the
   importer re-attaches each recorded mark to the letter beneath it.

The left (Persian) page is read with ``fas`` for the audit only.

Everything is pinned in ``PARAMS``; a page is re-run only when its cache entry was produced with
different parameters.  Renders are deleted as soon as the page is read (disk is tight).

Usage::

    nice -n 10 env OMP_THREAD_LIMIT=1 .venv/bin/python -I \
        data/other/forms/raw_data/kamioka_lari_1979/ocr_runner.py [--pages 10-186]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as nd

PDF_PATH = Path(
    "/Users/aryamanarora/Documents/Linguistics/Indo-European/Iranic/"
    "[Lārestāni studies] Kōji Kamioka_ _ Minoru Yamada - Lari basic vocabulary, (1979, "
    "Institute for the study of Languages and Cultures of Asia and Africa) - libgen.li.pdf"
)
PDF_SHA256 = "9196621ac485cbbe60cd302712a0d5289e7b2c3ad118446d74361cde0710c569"
PDF_PAGES = 232
ROOT = Path(__file__).resolve().parents[5]  # .../jambu-all/data
CACHE_DIR = ROOT.parent / "tmp/iranic/lari/ocr"
TESSERACT = "/opt/homebrew/bin/tesseract"

PARAMS = {
    "version": 4,
    "dpi": 300,
    "split": "half",           # spread split at W // 2
    "threshold": 140,          # gray < threshold is ink
    "underline_min_run": 40,   # px
    "underline_max_thickness": 7,
    "mark_max_h": 16, "mark_max_w": 32, "mark_min_area": 8,
    "mark_gap": (1, 18),       # vertical gap between mark bottom and letter top
    "letter_min_h": 12,
    "margin": (110, 120),      # ignore components within this many px of the page edge (x, y)
    "text_box": (100, 1950),   # right page: blank gutter/edge shadows outside this x-range
    "right_cmd": ["-l", "eng", "--psm", "6", "-c", "preserve_interword_spaces=1"],
    "left_cmd": ["-l", "fas", "--psm", "6"],
}
PARAMS_HASH = hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:16]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(page: int) -> Image.Image:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    try:
        return pdf[page - 1].render(scale=PARAMS["dpi"] / 72, grayscale=True).to_pil()
    finally:
        pdf.close()


# ---------------------------------------------------------------------------- geometry

def find_underlines(ink: np.ndarray) -> list[list[int]]:
    """Long thin horizontal runs: [y0, y1, x0, x1] (y1, x1 inclusive/exclusive as numpy)."""
    runs = []
    min_run = PARAMS["underline_min_run"]
    rows = np.where(ink.sum(axis=1) >= min_run)[0]
    for y in rows:
        d = np.diff(np.concatenate([[0], ink[y].astype(np.int8), [0]]))
        st = np.where(d == 1)[0]
        en = np.where(d == -1)[0]
        for s, e in zip(st, en):
            if e - s >= min_run:
                runs.append((int(y), int(s), int(e)))
    segs: list[list[int]] = []
    for y, s, e in runs:
        for g in segs:
            if g[1] in (y - 1, y) and min(g[3], e) - max(g[2], s) > 0.6 * min(g[3] - g[2], e - s):
                g[1] = y
                g[2] = min(g[2], s)
                g[3] = max(g[3], e)
                break
        else:
            segs.append([y, y, s, e])
    return [g for g in segs if g[1] - g[0] + 1 <= PARAMS["underline_max_thickness"]]


def classify(bitmap: np.ndarray) -> str:
    """Shape class of one mark component (bool array).

    The typewriter marks in this source are the macron, the diaeresis (two dots), the caron and
    a narrow stress stroke; anything else is ``unknown`` and is reported by the importer."""
    h, w = bitmap.shape
    area = int(bitmap.sum())
    if h <= 10 and w <= 10 and area <= 70 and w <= 1.8 * h + 2:
        return "dot"
    if h <= 6 and w >= 10:
        return "macron"
    if w >= 13 and 7 <= h <= 13:
        return "caron"
    rows = [np.where(r)[0] for r in bitmap]
    rows = [r for r in rows if len(r)]
    if w <= 12 and h >= 6 and len(rows) >= 2:
        k = max(1, len(rows) // 3)
        tc = np.mean([r.mean() for r in rows[:k]])
        bc = np.mean([r.mean() for r in rows[-k:]])
        if tc - bc >= 1.5:
            return "acute"
        if bc - tc >= 1.5:
            return "grave"
        return "stroke"
    return "unknown"


def find_marks(ink: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Overstrike marks above letters; returns marks and a mask of pixels to erase."""
    H, W = ink.shape
    mx, my = PARAMS["margin"]
    lab, n = nd.label(ink, structure=np.ones((3, 3), bool))
    objs = nd.find_objects(lab)
    comps = []
    for k, s in enumerate(objs, start=1):
        y0, y1, x0, x1 = s[0].start, s[0].stop, s[1].start, s[1].stop
        if x0 < mx or x1 > W - mx or y0 < my or y1 > H - my:
            continue
        comps.append({"k": k, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                      "area": int((lab[s] == k).sum())})
    # spatial index by column bands
    letters = [c for c in comps if c["y1"] - c["y0"] >= PARAMS["letter_min_h"]]
    cand = [c for c in comps
            if c["y1"] - c["y0"] <= PARAMS["mark_max_h"] and c["x1"] - c["x0"] <= PARAMS["mark_max_w"]
            and c["area"] >= PARAMS["mark_min_area"]]
    gmin, gmax = PARAMS["mark_gap"]
    letters.sort(key=lambda c: c["x0"])
    xs = np.array([c["x0"] for c in letters]) if letters else np.zeros(0)
    marks = []
    for c in cand:
        cx = (c["x0"] + c["x1"]) / 2
        lo = np.searchsorted(xs, c["x0"] - 40)
        hi = np.searchsorted(xs, c["x1"] + 1)
        best = None
        for L in letters[lo:hi]:
            gap = L["y0"] - c["y1"]
            if not (gmin - 1 <= gap <= gmax):
                continue
            if not (L["x0"] - 3 <= cx <= L["x1"] + 3):
                continue
            if best is None or gap < best["y0"] - c["y1"]:
                best = L
        if best is None:
            continue
        bm = lab[c["y0"]:c["y1"], c["x0"]:c["x1"]] == c["k"]
        marks.append({"y0": c["y0"], "y1": c["y1"], "x0": c["x0"], "x1": c["x1"],
                      "cls": classify(bm), "bm": ["".join("1" if v else "0" for v in r) for r in bm], "letter": [best["y0"], best["y1"], best["x0"], best["x1"]],
                      "k": c["k"]})
    # pair dots over the same letter into a diaeresis
    by_letter: dict[tuple, list[dict]] = {}
    for m in marks:
        by_letter.setdefault(tuple(m["letter"]), []).append(m)
    out = []
    erase = np.zeros_like(ink)
    for letter, ms in by_letter.items():
        dots = [m for m in ms if m["cls"] == "dot"]
        others = [m for m in ms if m["cls"] != "dot"]
        lw = letter[3] - letter[2]
        if len(dots) >= 2:
            dots.sort(key=lambda m: m["x0"])
            d = {"y0": min(m["y0"] for m in dots), "y1": max(m["y1"] for m in dots),
                 "x0": dots[0]["x0"], "x1": dots[-1]["x1"], "cls": "diaeresis", "bm": [],
                 "letter": list(letter), "k": [m["k"] for m in dots]}
            others.append(d)
        elif len(dots) == 1:
            # a single dot centred over a letter is the dot of i/j unless the letter is wide
            # (then it is a damaged diaeresis); the importer decides with the OCR letter.
            dm = dots[0]
            dm = dict(dm, cls="dot")
            out.append(dm)
        for m in others:
            out.append(m)
            ks = m["k"] if isinstance(m["k"], list) else [m["k"]]
            for k in ks:
                sl = (slice(m["y0"], m["y1"]), slice(m["x0"], m["x1"]))
                erase[sl] |= lab[sl] == k
        _ = lw
    for m in out:
        m.pop("k", None)
    out.sort(key=lambda m: (m["y0"], m["x0"]))
    return out, erase


# ---------------------------------------------------------------------------- tesseract

def tesseract_tsv(img: Image.Image, args: list[str]) -> list[dict]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "page.png"
        img.save(p)
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        res = subprocess.run([TESSERACT, str(p), "stdout", *args, "tsv"], capture_output=True,
                             text=True, env=env, check=True)
    words = []
    lines = res.stdout.splitlines()
    head = lines[0].split("\t")
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) != len(head):
            continue
        r = dict(zip(head, f))
        if r["level"] != "5" or not r["text"].strip():
            continue
        words.append({"block": int(r["block_num"]), "par": int(r["par_num"]),
                      "line": int(r["line_num"]), "word": int(r["word_num"]),
                      "x0": int(r["left"]), "y0": int(r["top"]),
                      "x1": int(r["left"]) + int(r["width"]), "y1": int(r["top"]) + int(r["height"]),
                      "conf": float(r["conf"]), "text": r["text"]})
    return words


def process(page: int) -> dict:
    img = render(page)
    W, H = img.size
    half = W // 2
    left = img.crop((0, 0, half, H))
    right = img.crop((half, 0, W, H))
    del img
    a = np.array(right)
    ink = a < PARAMS["threshold"]
    tx0, tx1 = PARAMS["text_box"]
    ink[:, :tx0] = False
    ink[:, tx1:] = False
    ul = find_underlines(ink)
    clean = ink.copy()
    for y0, y1, x0, x1 in ul:
        clean[y0:y1 + 1, x0:x1] = False
    marks, erase = find_marks(clean)
    pre_marks = clean.copy()
    clean &= ~erase
    cleaned = Image.fromarray(np.where(clean, 0, 255).astype(np.uint8))
    words = tesseract_tsv(cleaned, PARAMS["right_cmd"])
    pwords = tesseract_tsv(left, PARAMS["left_cmd"])
    return {"page": page, "params": PARAMS_HASH, "marks_version": MARKS_VERSION, "size": [W, H],
            "underlines": ul, "marks": marks, "words": words, "persian": pwords,
            "glyphs": word_glyphs(pre_marks, words)}


MARKS_VERSION = 4  # mark geometry + bitmaps; recomputable without re-running Tesseract


def right_ink(page: int) -> np.ndarray:
    img = render(page)
    W, H = img.size
    a = np.array(img.crop((W // 2, 0, W, H)))
    ink = a < PARAMS["threshold"]
    tx0, tx1 = PARAMS["text_box"]
    ink[:, :tx0] = False
    ink[:, tx1:] = False
    return ink


def recompute_marks(page: int, d: dict) -> dict:
    ink = right_ink(page)
    for y0, y1, x0, x1 in d["underlines"]:
        ink[y0:y1 + 1, x0:x1] = False
    marks, _ = find_marks(ink)
    d["marks"] = marks
    d["glyphs"] = word_glyphs(ink, d["words"])
    d["marks_version"] = MARKS_VERSION
    return d


def word_glyphs(ink: np.ndarray, words: list[dict]) -> list[list[list[int]]]:
    """Connected components (x0, y0, x1, y1) inside each word box, underlines removed.  Used
    by the importer to tell the typewriter apostrophe (ink only above the x-height) from ``t``."""
    lab, n = nd.label(ink, structure=np.ones((3, 3), bool))
    objs = nd.find_objects(lab)
    boxes = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop] for s in objs]) \
        if objs else np.zeros((0, 4), int)
    out = []
    for w in words:
        if not len(boxes):
            out.append([])
            continue
        m = ((boxes[:, 0] < w["x1"] + 2) & (boxes[:, 2] > w["x0"] - 2)
             & (boxes[:, 1] < w["y1"] + 2) & (boxes[:, 3] > w["y0"] - 25))
        rows = []
        for k in np.where(m)[0]:
            x0, y0, x1, y1 = (int(v) for v in boxes[k])
            # horizontal centroid (percent of width) of the bottom quarter of the glyph's ink:
            # the descender of q is a stem on the right, that of g a hook to the left
            sub = lab[y1 - max(1, (y1 - y0) // 4):y1, x0:x1] == k + 1
            cols = np.where(sub)[1]
            rows.append([x0, y0, x1, y1,
                         int(round(100 * (cols.mean() + 0.5) / (x1 - x0))) if len(cols) else -1])
        out.append(sorted(rows))
    return out


ALT_CMD = ["-l", "eng", "--psm", "4"]  # independent second reading of the raw right page


def alt_text(page: int) -> str:
    """Plain Tesseract reading of the unprocessed right page, for comparison in the audit."""
    img = render(page)
    W, H = img.size
    right = img.crop((W // 2, 0, W, H))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "page.png"
        right.save(p)
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        res = subprocess.run([TESSERACT, str(p), "stdout", *ALT_CMD], capture_output=True,
                             text=True, env=env, check=True)
    return res.stdout


def cache_path(page: int) -> Path:
    return CACHE_DIR / f"p{page:03d}.json"


def load(page: int) -> dict | None:
    p = cache_path(page)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("params") == PARAMS_HASH else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="10-186")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not PDF_PATH.exists():
        sys.exit(f"scan not found: {PDF_PATH}")
    if sha256(PDF_PATH) != PDF_SHA256:
        sys.exit("scan sha256 mismatch")
    import pypdf
    n = len(pypdf.PdfReader(str(PDF_PATH)).pages)
    if n != PDF_PAGES:
        sys.exit(f"scan has {n} pages, expected {PDF_PAGES}")
    lo, hi = (int(x) for x in args.pages.split("-"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for page in range(lo, hi + 1):
        d = None if args.force else load(page)
        if d is not None and d.get("marks_version") == MARKS_VERSION and "alt" in d:
            continue
        if d is None:
            d = process(page)
        elif d.get("marks_version") != MARKS_VERSION:
            d = recompute_marks(page, d)
        if "alt" not in d:
            d["alt"] = alt_text(page)
        tmp = cache_path(page).with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False))
        tmp.replace(cache_path(page))
        print(page, len(d["words"]), len(d["marks"]), len(d["underlines"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
