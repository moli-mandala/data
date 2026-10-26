#!/usr/bin/env python3
"""Import Johnny Cheung, *Etymological Dictionary of the Iranian Verb* (Brill 2007).

Run from the git root (``data/``)::

    .venv/bin/python -I data/other/forms/raw_data/cheung_ediv_2007/import_source.py            # dry run
    .venv/bin/python -I data/other/forms/raw_data/cheung_ediv_2007/import_source.py --install  # write

Pipeline (see README.md): ``pdf_layer`` (glyph-named text layer, cached outside the repo)
→ ``decode`` (``glyphs.csv``) → ``structure`` (entries, sections, paradigms, reference
column) → ``items`` (Iranian citation items) → this module (rows, entry texts, audit).

Outputs: the 15-column forms CSV (Iranian attestations), the PIr etymon layer
(``data/other/params``), entry-text prose blocks keyed by the PIr etymon IDs, the per-unit
audit, a seeded 20-unit raw-vs-output sample, and a manifest with the PDF's SHA-256.
The copyrighted PDF is never copied; the importer fails clearly when it is absent and no
verified extraction cache exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode  # noqa: E402
import items as I  # noqa: E402
import pdf_layer  # noqa: E402
import structure  # noqa: E402
import tokens as T  # noqa: E402

SOURCE_ID = "cheung2007ediv"
STEM = "20261025-cheung-ediv"
SNAPSHOT_DATE = "2026-10-25"
ROOT = HERE.parents[4]  # git root (data/)
PDF = Path("/Users/aryamanarora/Documents/Linguistics/Indo-European/Iranic/Cheung Iranian.pdf")
CACHE = ROOT.parent / "tmp/iranic/cheung/extract.json.gz"
FORMS_OUT = ROOT / f"data/other/forms/{STEM}.csv"
PARAMS_OUT = ROOT / f"data/other/params/{STEM}.csv"
TEXTS_OUT = ROOT / f"data/other/entry_texts/{STEM}.csv"
AUDIT_OUT = HERE / "audit.csv"
SAMPLE_OUT = HERE / "sample.csv"
MANIFEST_OUT = HERE / "manifest.json"
SAMPLE_SEED = "cheung2007ediv-audit-3"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Unit_ID", "Unit_Type", "Root_ID", "Printed_Page", "PDF_Page", "Section", "Group",
    "Preverb", "Printed_Label", "Resolved_Label", "Language_ID", "Dialect_Tag", "Raw",
    "Form", "Phonemic", "Gloss", "Gloss_Scope", "Tags", "Notes", "Status", "Reason",
    "Uncertainty", "Entry_Key", "Etymology", "Passage_Locators",
]
TEXT_FIELDS = ["Form_ID", "Position", "Kind", "Format", "Content", "Source"]
SAMPLE_FIELDS = AUDIT_FIELDS + ["Verdict"]

PROSE_KIND = {
    "SANSKRIT": "comparison", "IE COGNATES": "comparison", "MISC": "comparison",
    "PIE": "etymology", "REFERENCES": "bibliographic-reference",
}
ASCII_ID = {
    "č": "ch", "ǰ": "j", "š": "sh", "ž": "zh", "θ": "th", "δ": "dh", "γ": "gh", "β": "bh",
    "ŋ": "ng", "ā": "aa", "ī": "ii", "ū": "uu", "ē": "ee", "ō": "oo", "ś": "sy", "ṛ": "r",
    "u̯": "w", "i̯": "y", "r̥": "rr", "x": "x",
}

# Manual verdicts for the seeded raw-vs-output sample (Unit_ID -> verdict).
# Seed 1 (20 rows) found the hyphen line-break class; seed 2 (20 rows) found none, but a
# coordinator review then found gloss over-capture, split Bactrian ϸ/hyphens, analysis in
# Notes and parenthetical-locator classes (all fixed and regression-tested). Seed 3 is a
# 50-row sample stratified over OPers, Bactr, Chvar, Sogd, Khot and modern languages (5+
# each), compared with the printed page: 0/50 material errors.
SAMPLE_VERDICTS: dict[str, str] = {
    "cheung-ediv-kan1:s2:g1:i1": "ok",
    "cheung-ediv-daH2:s2:g1:i1": "ok",
    "cheung-ediv-dar1:s2:g2:i3": "ok",
    "cheung-ediv-Har1:s2:g4:i4": "ok",
    "cheung-ediv-bar:s2:g1:i1": "ok",
    "cheung-ediv-jan:s8:g2:i2": "ok",
    "cheung-ediv-drauj:s7:g1:i2": "ok",
    "cheung-ediv-bar:s8:g1:i2": "ok",
    "cheung-ediv-raHz:s3:g1:i1": "ok",
    "cheung-ediv-gaaz:s4:g2:i2": "ok",
    "cheung-ediv-hrau:s6:g2:i3": "ok",
    "cheung-ediv-daH2:s6:g1:i2": "ok",
    "cheung-ediv-tamH:s4:g1:i2": "ok",
    "cheung-ediv-baxsh:s6:g1:i1": "ok",
    "cheung-ediv-dauH1:s4:g1:i1": "ok",
    "cheung-ediv-HaishH:s6:g2:i3": "ok",
    "cheung-ediv-Hraz:s6:g3:i6": "ok",
    "cheung-ediv-tash:s5:g1:i1": "ok",
    "cheung-ediv-baud2:s5:g2:i2": "ok",
    "cheung-ediv-xaaz:s2:g2:i4": "ok",
    "cheung-ediv-dauH2:s3:g2:i2": "ok",
    "cheung-ediv-sauch1:s5:g3:i5": "ok",
    "cheung-ediv-HwadH:s2:g1:i1": "ok",
    "cheung-ediv-hwar5:s2:g1:i1": "ok",
    "cheung-ediv-kan1:s5:g4:i7": "ok",
    "cheung-ediv-tap:s8:g1:i3": "ok",
    "cheung-ediv-braiH:s6:g1:i19": "ok",
    "cheung-ediv-skar:s7:g1:i2": "ok",
    "cheung-ediv-Hrab-f3:s2:g3:i10": "ok",
    "cheung-ediv-garH3:s4:g2:i3": "ok",
    "cheung-ediv-Hraz:s3:g7:i17": "ok",
    "cheung-ediv-Haz:s2:g1:i1": "ok",
    "cheung-ediv-waaz1:s1:g1:i3": "ok",
    "cheung-ediv-k_a_ur:s2:g1:i6": "ok",
    "cheung-ediv-xshnaush:s4:g1:i2": "ok: source prints no slash between the principal parts (checked in the PDF)",
    "cheung-ediv-jaiH1:s9:g1:i5": "ok",
    "cheung-ediv-zgar:s4:g1:i6": "ok",
    "cheung-ediv-xaH:s7:g1:i1": "ok",
    "cheung-ediv-_d_maan:s8:g1:i18": "ok",
    "cheung-ediv-dra_n_j:s3:g3:i3": "ok",
    "cheung-ediv-rauch:s7:g4:i9": "ok",
    "cheung-ediv-karsh-xrah:s7:g1:i4": "ok",
    "cheung-ediv-dar2:s5:g1:i1": "ok",
    "cheung-ediv-tarH1:s8:g5:i12": "ok",
    "cheung-ediv-sparH:s4:g2:i4": "ok",
    "cheung-ediv-gaz:s3:g1:i4": "ok",
    "cheung-ediv-dar2:s5:g1:i21": "ok",
    "cheung-ediv-raup1:s5:g1:i1": "ok",
    "cheung-ediv-kan1:s9:g3:i7": "ok",
    "cheung-ediv-gaH2:s4:g1:i3": "ok",
}


# --------------------------------------------------------------------------- headwords

HEAD_RE = re.compile(r"^\*(?P<root>\S+)(?:\s+(?P<q1>\?))?(?:\s+\((?P<alt>[^()]*(?:\([^()]*\)[^()]*)*)\))?\s*(?P<q>\?)?\s*$")
# Cheung prints two unnumbered *baǰ headwords (pp. 1 and 3); unnumbered same-root headwords
# are told apart by their printed page.
PAGE_DISAMBIGUATED = {"baǰ"}


def parse_headword(entry: structure.Entry) -> dict:
    text = structure.join_lines(entry.head_lines)
    bold_end = text.find(decode.B_OFF)
    bold = text[1:bold_end] if text.startswith(decode.B_ON) and bold_end > 0 else ""
    rest = text[bold_end + 1:] if bold_end > 0 else text
    # homonym index: a raised digit inside the bold run (*baud², *či̯aH³/čiH)
    hom = ""
    m = re.search(re.escape(decode.SUP_ON) + r"\s*(\d)\s*" + re.escape(decode.SUP_OFF), bold)
    if m:
        hom = m.group(1)
    bold_plain = re.sub(re.escape(decode.SUP_ON) + r"[^" + decode.SUP_OFF + r"]*" + re.escape(decode.SUP_OFF), "", bold)
    bold_plain = decode.plain(bold_plain).strip()
    rest_plain = T.strip_markers(rest).strip()
    question = False
    if rest_plain.startswith("?"):
        question = True
        rest_plain = rest_plain[1:].strip()
    hm = HEAD_RE.match(bold_plain)
    if not hm:
        raise ValueError(f"p. {entry.page}: unparsed headword {bold_plain!r}")
    root = hm.group("root")
    alt = (hm.group("alt") or "").strip()
    question = question or bool(hm.group("q")) or bool(hm.group("q1"))
    if alt == "?":
        alt, question = "", True
    alt_doubt = alt.endswith("?")
    gloss = ""
    if "‘" in rest_plain:
        gloss = rest_plain[rest_plain.index("‘") + 1:rest_plain.rindex("’")].strip() \
            if "’" in rest_plain else rest_plain[rest_plain.index("‘") + 1:].strip()
    return {
        "root": root, "homonym": hom, "alt": alt, "uncertain": question,
        "alt_uncertain": alt_doubt, "gloss": gloss, "page": entry.page,
        "raw": decode.plain(text).strip(),
    }


def root_id(head: dict) -> str:
    s = unicodedata.normalize("NFC", head["root"])
    for k in sorted(ASCII_ID, key=len, reverse=True):
        s = s.replace(k, ASCII_ID[k])
    s = s.replace("/", "-").replace("(", "_").replace(")", "_").replace("*", "")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", s):
        raise ValueError(f"unmapped character in root id {head['root']!r} -> {s!r}")
    suffix = f"-p{head['page']}" if head["root"] in PAGE_DISAMBIGUATED and not head["homonym"] else ""
    return f"cheung-ediv-{s}{head['homonym']}{suffix}"


def root_form(head: dict) -> str:
    return "*" + head["root"]


# --------------------------------------------------------------------------- helpers

def locator(page: int, last: int | None = None) -> str:
    if last and last != page:
        return f"{SOURCE_ID}[pp. {page}–{last}]"
    return f"{SOURCE_ID}[p. {page}]"


def markdown(text: str) -> str:
    """Section text → markdown: italic runs as _…_, literal * and _ escaped."""
    out, italic = [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in (decode.SUP_ON, decode.SUB_ON):
            close = decode.SUP_OFF if ch == decode.SUP_ON else decode.SUB_OFF
            j = text.find(close, i)
            j = len(text) if j < 0 else j
            inner = text[i + 1:j]
            # italic markers can straddle a script run; keep them balanced
            for c in inner:
                if c == decode.I_ON and not italic:
                    out.append("_"); italic = True
                elif c == decode.I_OFF and italic:
                    out.append("_"); italic = False
            out.append(T.to_script(decode.plain(inner), ch == decode.SUP_ON))
            i = j + 1
            continue
        if ch == decode.I_ON:
            if not italic:
                out.append("_"); italic = True
        elif ch == decode.I_OFF:
            if italic:
                spaces = 0
                while out and out[-1] == " ":
                    out.pop(); spaces += 1
                out.append("_" + " " * spaces); italic = False
        elif ch in decode.MARKERS:
            pass
        elif ch in "*_\\":
            out.append("\\" + ch)
        else:
            out.append(ch)
        i += 1
    if italic:
        out.append("_")
    md = "".join(out)
    md = re.sub(r"\s+_(?=[\s,.;:)])", "_", md)  # trailing space inside an italic run
    md = re.sub(r"_\s+_", " ", md)
    return re.sub(r"[ \t]+", " ", md).strip()


DISPLAY = {"NWIR": "NW Iranian", "NEIR": "NE Iranian", "PIE": "PIE", "MISC": "Misc.",
           "IE COGNATES": "IE cognates", "—": "Unlabelled"}


def display(label: str) -> str:
    return DISPLAY.get(label, label.title())


def key_part(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip()).replace(":", "∶")


# --------------------------------------------------------------------------- build

def build():
    pages = pdf_layer.extract(PDF, CACHE)
    lects = I.load_lects()
    entries = structure.entries_from_pages(pages)
    params, forms, texts, audit = [], [], [], []
    ids: Counter = Counter()
    keys: Counter = Counter()
    for entry in entries:
        head = parse_headword(entry)
        pid = root_id(head)
        ids[pid] += 1
        last = entry.last_page
        params.append([pid, "PIr", root_form(head), head["gloss"], locator(entry.page)])
        uncertainty = []
        if head["uncertain"]:
            uncertainty.append("etymology: root printed with ?")
        audit.append(_audit(f"{pid}", "root", pid, entry.page, entry.lines[0].pdf, "", "", "",
                            "", "", "PIr", "", head["raw"], root_form(head), "", head["gloss"],
                            "", "", f"alternative reconstruction: {head['alt']}" if head["alt"] else "",
                            "installed_etymon", "PIr etymon (params)", "; ".join(uncertainty), ""))
        position = 0
        # headword-level prose: alternative reconstruction / uncertainty
        if head["alt"] or head["uncertain"]:
            bits = []
            if head["uncertain"]:
                bits.append("Cheung marks the root reconstruction as uncertain (?).")
            if head["alt"]:
                bits.append(f"Alternative reconstruction printed beside the headword: {markdown(head['alt'])}.")
            position += 1
            texts.append([pid, position, "etymology", "markdown", " ".join(bits), locator(entry.page)])
        refs_by_section = []
        for s_index, section in enumerate(entry.sections):
            sec_label = section.label
            sec_text = section.text()
            sec_page = section.page
            sec_last = section.lines[-1].page if section.lines else sec_page
            unit = f"{pid}:s{s_index + 1}"
            if section.refs:
                refs_by_section.append((sec_label or "—", "; ".join(r for _, r in section.refs)))
            for p_index, block in enumerate(section.paradigms, 1):
                raw = structure.join_lines(block)
                audit.append(_audit(f"{unit}:paradigm{p_index}", "paradigm", pid, block[0].page,
                                    block[0].pdf, sec_label, "", "", "", "", "", "",
                                    decode.plain(raw), "", "", "", "", "", "", "audit_only_paradigm",
                                    "inflected forms with text passages: installed rows are lemmata", "", ""))
            if sec_label in I.FORM_SECTIONS:
                items, body, commentary = I.parse_section(sec_text, sec_label, lects)
                nonir = [it for it in items if it.status == "non_iranian"]
                for it in items:
                    _emit_item(it, pid, head, entry, section, unit, lects, forms, audit, keys)
                if commentary.strip():
                    position += 1
                    texts.append([pid, position, "etymology", "markdown",
                                  f"**{display(sec_label or 'Note')}:** {markdown(commentary)}",
                                  locator(sec_page, sec_last)])
                    audit.append(_audit(f"{unit}:comment", "prose", pid, sec_page, section.lines[0].pdf,
                                        sec_label, "", "", "", "", "", "", decode.plain(commentary),
                                        "", "", "", "", "", "", "entry_text", "◊ commentary → etymology block",
                                        "", ""))
                if sec_label == "MISC" and nonir:
                    position += 1
                    texts.append([pid, position, "comparison", "markdown",
                                  f"**Misc.:** {markdown(body)}", locator(sec_page, sec_last)])
                continue
            if sec_label in PROSE_KIND and sec_text.strip():
                position += 1
                name = {"PIE": "PIE", "IE COGNATES": "IE cognates", "REFERENCES": "References",
                        "SANSKRIT": "Sanskrit"}[sec_label]
                texts.append([pid, position, PROSE_KIND[sec_label], "markdown",
                              f"**{name}:** {markdown(sec_text)}", locator(sec_page, sec_last)])
                audit.append(_audit(f"{unit}:prose", "prose", pid, sec_page,
                                    section.lines[0].pdf if section.lines else entry.lines[0].pdf,
                                    sec_label, "", "", "", "", "", "", decode.plain(sec_text),
                                    "", "", "", "", "", "", "entry_text",
                                    f"{sec_label} → {PROSE_KIND[sec_label]} block", "", ""))
            elif sec_text.strip():
                raise ValueError(f"unhandled section {sec_label!r} at p. {sec_page}")
        if refs_by_section:
            position += 1
            texts.append([pid, position, "bibliographic-reference", "markdown",
                          "**Sources by section:** " + "; ".join(
                              f"{display(lab)}: {markdown(ref)}" for lab, ref in refs_by_section),
                          locator(entry.page, last)])
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise ValueError(f"duplicate root IDs: {dup}")
    return entries, params, forms, texts, audit


# Avestan / Old Persian / Bactrian / Manichaean text loci: Yt 10.134, DB 4.84, DNb.7, J11, M178 IIV.124
LOCUS = re.compile(r"(?:\?\s*)?(?:in\s+)?(?:[A-Z][A-Za-z]{0,9}\.?\s?)?(?:[0-9IVX]|[a-z]\.\d)[\w.’/,]*"
                   r"(?:\s?[\w.’]{1,4})?(?:\s?f{1,2}\.)?|(?:in\s+)?[A-Z][A-Za-z]{0,9}\.?|etc\.?|ibid\.?")
ETYMOLOGY_START = ("<", ">", "cf.", "rather", "orig.", "from", "perh", "prob", "possibl", "v.", "s.v.",
                   "=", "with", "not ", "differently", "borrowed", "dissim", "sec.", "old ", "lit.",
                   "*", "pace", "thus", "i.e.", "q.v.", "see", "contra", "pace", "derived", "denomin.",
                   "back-formation", "metathesis", "secondary", "analogical", "influenced", "contamin",
                   "LW", "loan", "folk")
CITATION = re.compile(r"\b[A-Z][\w’.-]+(?: [–-] [A-Z][\w’.-]+)?,? (?:\d{4}|[IVX]+)\b|l\.c\.|apud|\b[A-Z][\w.’ ]*:\s?§?\d")


def classify_note(note: str) -> tuple[str, str]:
    """Route an item parenthetical: tag | query-tag | etymology | locator | notes | drop."""
    note = note.strip()
    if note in ("?", "") or not any(c.isalnum() for c in note):
        return "drop", ""
    if re.fullmatch(r"<[^<>\s][^<>]*>", note):
        return "notes", note  # Old Persian graphemic transliteration <b-r-š-n-a>
    inner = note[1:-1].strip() if note.startswith("(") and note.endswith(")") else note
    parts = [p.strip() for p in re.split(r",\s+", inner) if p.strip()]
    pieces = [p for p in re.split(r"[,/\s]+|(?<=\.)-", inner) if p]
    if pieces and all(I.GRAMMAR.get(p) for p in pieces):
        return "tag", " ".join(t for p in pieces for t in I.GRAMMAR[p])
    q = inner.rstrip(" ?").strip()
    qpieces = [p for p in re.split(r"[,/\s]+|(?<=\.)-", q) if p]
    if inner.endswith("?") and qpieces and all(I.GRAMMAR.get(p) for p in qpieces):
        return "query-tag", " ".join(t for p in qpieces for t in I.GRAMMAR[p])
    if re.fullmatch(r"pret\.? stem|past stem", inner):
        return "tag", "pret stem"
    if note.startswith("(") and parts and all(LOCUS.fullmatch(p) for p in parts) \
            and any(re.search(r"\d", p) for p in parts):
        return "locator", inner
    if inner in ("also", "etc.", "etc", "e.g.", "and", "or", "i.e.") or re.fullmatch(r"\d+\.?", inner):
        return "drop", ""
    if note.startswith("(") and re.fullmatch(r"[A-Z0-9][\w.,/’ ]*", inner) and re.search(r"\d", inner) \
            and not re.search(r"\b[a-z]{5,}\b", inner):
        return "locator", inner  # manuscript loci: (P 3513.62v1 KT 1.248, Vajr. 12b4 KT 3.22)
    if inner.startswith(ETYMOLOGY_START) or " < " in inner or "*" in inner or CITATION.search(inner) \
            or re.search(r"\b(?:cf|rather|orig)\.?\s", inner):
        return "etymology", inner
    return "notes", note


def _emit_item(it: I.Item, pid, head, entry, section, unit, lects, forms, audit, keys):
    lect = lects.get(it.label, {})
    language = lect.get("Language_ID", "")
    dialect = lect.get("Dialect_Tag", "")
    uncertainty = []
    if it.doubt:
        uncertainty.append("etymology: source query (?) on the connection")
    if it.reading:
        uncertainty.append(f"transcription: {it.reading} reading printed")
    if it.form.startswith("*") or "/*" in it.form:
        uncertainty.append("transcription: starred (inferred) stem printed among the attestations")
    status = it.status
    reason = it.reason
    form = it.form
    gloss = it.gloss
    if gloss in ("id.", "id"):
        gloss = ""
        reason = (reason + "; " if reason else "") + "gloss 'id.' left blank (refers to prose)"
    entry_key = ""
    if status == "form":
        base = f"{SOURCE_ID}:{pid}:{key_part(it.label.lstrip('§'))}:{key_part(form)}"
        keys[base] += 1
        entry_key = base if keys[base] == 1 else f"{base}:{keys[base]}"
        tags = list(dict.fromkeys([*([dialect] if dialect else []), *it.tags,
                                   *(["uncertain"] if uncertainty else [])]))
        notes, etymology, loci = [], [], []
        if it.preverb:
            etymology.append(f"preverb {it.preverb.lstrip('+ ').strip()}" if it.preverb.startswith("+")
                             else f"preverb {it.preverb}")
        for note in it.notes:
            kind, value = classify_note(note)
            if kind == "tag":
                tags += [t for t in value.split() if t not in tags]
            elif kind == "query-tag":
                tags += [t for t in value.split() if t not in tags]
                if "uncertain" not in tags:
                    tags.append("uncertain")
                uncertainty.append(f"grammar: queried label ({note})")
            elif kind == "etymology":
                etymology.append(value)
            elif kind == "locator":
                loci.append(value)
            elif kind == "notes":
                notes.append(value)
        forms.append({
            "Language_ID": language, "Parameter_ID": pid, "Form": form, "Gloss": gloss,
            "Native": "", "Phonemic": it.phonemic, "Notes": "; ".join(notes),
            "Source": locator(section.page), "Cognateset": "", "Etymology": "; ".join(etymology),
            "Entry_Key": entry_key, "Variant_Of_Key": "", "Borrowed_From_Key": "",
            "Derivation_Parent_Keys": "", "Tags": " ".join(tags),
        })
        status = "installed_form"
    else:
        etymology, loci, notes = [], [], [n for n in it.notes if n != "?"]
    audit.append(_audit(
        f"{unit}:g{it.group + 1}:i{it.index + 1}", "form", pid, section.page,
        section.lines[0].pdf if section.lines else entry.lines[0].pdf, section.label,
        str(it.group + 1), it.preverb, it.printed_label, it.label, language, dialect,
        decode.plain(it.raw), form, it.phonemic, gloss,
        "inherited from following item" if it.gloss_inherited else ("printed" if gloss else ""),
        " ".join(tags) if it.status == "form" else " ".join(it.tags), "; ".join(notes), status,
        reason, "; ".join(uncertainty), entry_key,
        "; ".join(etymology), "; ".join(loci)))


def _audit(unit, kind, pid, page, pdf, section, group, preverb, printed_label, label, language,
           dialect, raw, form, phonemic, gloss, scope, tags, notes, status, reason, uncertainty,
           entry_key, etymology="", loci=""):
    return dict(zip(AUDIT_FIELDS, [unit, kind, pid, page, pdf, section, group, preverb,
                                   printed_label, label, language, dialect, raw, form, phonemic,
                                   gloss, scope, tags, notes, status, reason, uncertainty,
                                   entry_key, etymology, loci]))


STRATA = ("OPers", "Bactr", "Chvar", "Sogd", "Khot", "modern")
MIDDLE_OLD = {"Av", "OPers", "MPrs", "Parth", "Khot", "Tumsh", "Sogd", "Chvar", "Bactr"}


def seeded_sample(audit: list[dict], n: int = 50, per_stratum: int = 5) -> list[dict]:
    """Seeded raw-vs-output sample of installed forms: ``per_stratum`` rows from each of
    OPers, Bactr, Chvar, Sogd, Khot and the modern languages, the rest from all rows."""
    pool = [row for row in audit if row["Status"] == "installed_form"]
    ranked = sorted(pool, key=lambda r: hashlib.sha256(
        f"{SAMPLE_SEED}|{r['Unit_ID']}".encode()).hexdigest())
    chosen: list[dict] = []
    for stratum in STRATA:
        members = [r for r in ranked if (r["Language_ID"] == stratum) or
                   (stratum == "modern" and r["Language_ID"] not in MIDDLE_OLD)]
        chosen += [r for r in members if r not in chosen][:per_stratum]
    chosen += [r for r in ranked if r not in chosen][:n - len(chosen)]
    out = []
    for row in chosen:
        row = dict(row)
        row["Verdict"] = SAMPLE_VERDICTS.get(row["Unit_ID"], "")
        out.append(row)
    return out


def write_csv(path: Path, fields, rows, header=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if header:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(handle, lineterminator="\n")
            for row in rows:
                writer.writerow([row[f] for f in fields] if isinstance(row, dict) else row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--install", action="store_true", help="write the installed outputs")
    args = parser.parse_args()
    entries, params, forms, texts, audit = build()
    status = Counter(row["Status"] for row in audit)
    langs = Counter(row["Language_ID"] for row in forms)
    print(f"{len(entries)} roots -> {len(params)} params, {len(forms)} form rows, "
          f"{len(texts)} entry texts, {len(audit)} audit units")
    print("audit statuses:", dict(status))
    print("languages:", dict(langs.most_common()))
    if not args.install:
        return
    write_csv(FORMS_OUT, FORM_FIELDS, forms, header=False)
    write_csv(PARAMS_OUT, None, params, header=False)
    write_csv(TEXTS_OUT, TEXT_FIELDS, [dict(zip(TEXT_FIELDS, t)) for t in texts])
    write_csv(AUDIT_OUT, AUDIT_FIELDS, audit)
    sample = seeded_sample(audit)
    write_csv(SAMPLE_OUT, SAMPLE_FIELDS, sample)
    MANIFEST_OUT.write_text(json.dumps({
        "source_id": SOURCE_ID,
        "bibliography": "Cheung, Johnny. 2007. Etymological Dictionary of the Iranian Verb. "
                        "(Leiden Indo-European Etymological Dictionary Series 2.) Leiden: Brill.",
        "pdf_path": str(PDF), "pdf_sha256": pdf_layer.PDF_SHA256, "pdf_pages": pdf_layer.PDF_PAGES,
        "pdf_redistributed": False,
        "dictionary_pdf_pages": [structure.FIRST_PDF, structure.LAST_PDF],
        "printed_pages": [structure.FIRST_PDF - structure.PAGE_OFFSET,
                          structure.LAST_PDF - structure.PAGE_OFFSET],
        "snapshot_date": SNAPSHOT_DATE,
        "extraction": {
            "method": "born-digital text layer; glyph names from the fonts' /Differences "
                      "arrays decoded through glyphs.csv (no OCR)",
            "extract_version": pdf_layer.EXTRACT_VERSION,
            "cache": "tmp/iranic/cheung/extract.json.gz (outside the repo)",
            "glyph_table_rows": sum(1 for _ in open(HERE / "glyphs.csv", encoding="utf-8")) - 1,
        },
        "outputs": {
            "forms": str(FORMS_OUT.relative_to(ROOT)), "form_rows": len(forms),
            "params": str(PARAMS_OUT.relative_to(ROOT)), "param_rows": len(params),
            "entry_texts": str(TEXTS_OUT.relative_to(ROOT)), "entry_text_rows": len(texts),
            "audit": str(AUDIT_OUT.relative_to(ROOT)), "audit_rows": len(audit),
            "sample": str(SAMPLE_OUT.relative_to(ROOT)), "sample_rows": len(sample),
        },
        "audit_status_counts": dict(sorted(status.items())),
        "form_rows_by_language": dict(sorted(langs.items())),
        "sample_seed": SAMPLE_SEED,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("installed")


if __name__ == "__main__":
    main()
