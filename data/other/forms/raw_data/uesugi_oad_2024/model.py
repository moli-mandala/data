"""Turn segmented OAD units into installed rows, entry texts and audit rows."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata as ud
from collections import Counter, defaultdict
from pathlib import Path

from header import parse_attestation, parse_lemma, parse_xref
from parse import PROPOSED_TAGS, SUPERSCRIPT_DIGITS, raw_text

SOURCE_ID = "uesugi-catt2024oad"
LANGUAGE_ID = "Av"
OAV_TAG = "dialect:Av:OAv:Old%20Avestan"
# integration flips this once ``preverb`` is registered in tags.py (see proposals.md)
EMIT_PROPOSED_TAGS = True

FORM_FIELDS = ["Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
               "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
               "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]
TEXT_FIELDS = ["Form_ID", "Entry_Key", "Position", "Kind", "Format", "Content", "Source"]
AUDIT_FIELDS = ["Unit_ID", "Unit_Kind", "PDF_Page", "Printed_Page", "Column", "Status",
                "Reason", "Entry_Key", "Lemma_Key", "Parsed_Form", "Parsed_Gloss",
                "Parsed_Tags", "Grammar_Code", "Passages", "Target", "Parent_Keys",
                "Review", "Raw_Text", "Record_SHA256"]


def nfc(text: str) -> str:
    return ud.normalize("NFC", text)


def unit_id(unit: dict) -> str:
    return f"p{unit['printed_page']:03d}c{unit['col']}y{int(unit['top']):03d}"


def norm_headword(text: str) -> str:
    text = nfc(re.sub(r"√\s+", "√", text.strip()))
    return re.sub(r"\s+", " ", text)


def root_lookup_key(text: str) -> str:
    return "".join(ch for ch in nfc(text) if ch not in SUPERSCRIPT_DIGITS)


def locator(page: int) -> str:
    return f"{SOURCE_ID}[p. {page}]"


def load_editorial(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["Unit_ID"]: row for row in csv.DictReader(handle)}


def visible_tags(tags: list[str]) -> list[str]:
    return [t for t in tags if EMIT_PROPOSED_TAGS or t not in PROPOSED_TAGS]


def build_rows(units: list[dict], editorial: dict[str, dict]):
    lemmas, others = [], []
    current = None
    for unit in units:
        uid = unit_id(unit)
        if unit["kind"] in ("lemma", "verb"):
            rec = parse_lemma(unit)
            rec.update(unit=unit, uid=uid, page=unit["printed_page"], kind=unit["kind"],
                       attestations=0)
            ed = editorial.get(uid, {})
            rec["editorial"] = ed
            if ed.get("Headwords"):
                rec["headwords"] = [h.strip() for h in ed["Headwords"].split("|")]
            if ed.get("Gloss"):
                # "-" records a deliberate blank: the printed quotes are not lemma senses
                rec["glosses"] = [] if ed["Gloss"] == "-" else [ed["Gloss"]]
            if ed.get("Tags"):
                rec["tags"] = ed["Tags"].split()
            if ed.get("Variants_Or") == "-":
                rec["variants_or"] = []
            rec["headwords"] = [norm_headword(h) for h in rec["headwords"]]
            rec["variants_or"] = [norm_headword(h) for h in rec["variants_or"]]
            if unit["kind"] == "verb" and "verb" not in rec["tags"]:
                rec["tags"] = ["verb"] + rec["tags"]
            lemmas.append(rec)
            current = rec
        else:
            others.append((unit, uid, current))
            if unit["kind"] == "attestation" and current is not None:
                current["attestations"] += 1

    # printed identity -> Entry_Key; repeated printed headwords take a page (and position)
    installed = [r for r in lemmas if r["editorial"].get("Action", "") not in
                 ("subsection", "exclude")]
    count = Counter(r["headwords"][0] for r in installed)
    seen_page = Counter((r["headwords"][0], r["page"]) for r in installed)
    ordinal: Counter = Counter()
    for r in installed:
        hw = r["headwords"][0]
        key = f"{SOURCE_ID}:{hw}"
        if count[hw] > 1:
            # unnumbered printed homographs: page, then print order on that page
            key += f":p{r['page']}"
            ordinal[(hw, r["page"])] += 1
            if seen_page[(hw, r["page"])] > 1:
                key += f":{ordinal[(hw, r['page'])]}"
        r["key"] = key
    # a repeated sub-heading points its attestations at the lemma it repeats
    last_key: dict[str, str] = {}
    for r in lemmas:
        if "key" in r:
            last_key[r["headwords"][0]] = r["key"]
        elif r["editorial"].get("Action") == "subsection":
            r["key"] = last_key.get(r["headwords"][0], "")
            r["subsection_of"] = r["key"]
    roots = defaultdict(list)
    for r in installed:
        for hw in r["headwords"]:
            if hw.startswith("√"):
                roots[nfc(hw)].append(r["key"])
                roots[root_lookup_key(hw)].append(r["key"])
    for r in installed:
        r["parents"], r["root_review"] = resolve_roots(r, roots)

    forms, texts, audit = [], [], []
    for r in lemmas:
        emit_lemma(r, forms, texts, audit)
    for unit, uid, lemma in others:
        audit.append(audit_other(unit, uid, lemma))
    return lemmas, forms, texts, audit


def resolve_roots(rec: dict, roots: dict) -> tuple[list[str], str]:
    if rec["kind"] != "lemma" or not rec["root_groups"]:
        return [], ""
    groups = list(dict.fromkeys(rec["root_groups"]))
    if len(groups) > 1:
        return [], f"ambiguous: several printed roots {groups}"
    printed = nfc(groups[0])
    pieces = [p if p.startswith("√") else "√" + p for p in printed.split("/")]
    if any(not root_lookup_key(p).strip("√") for p in pieces):
        return [], f"ambiguous: printed root {printed} names several homographs"

    def keys_for(piece: str) -> list[str]:
        has_digit = any(ch in SUPERSCRIPT_DIGITS for ch in piece)
        return list(dict.fromkeys(roots.get(piece, []) if has_digit
                                  else roots.get(root_lookup_key(piece), [])))

    keys = keys_for(pieces[0])
    # a printed alternant (√²iš/aēš) must not contradict the first piece's unique match
    if len(keys) == 1 and all(not keys_for(p) or keys[0] in keys_for(p) for p in pieces[1:]) \
            and keys[0] != rec.get("key"):
        return keys, f"linked: printed root {printed}"
    if not keys:
        return [], f"unresolved: printed root {printed} is not an OAD headword"
    return [], f"ambiguous: printed root {printed} matches {keys}"


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def emit_lemma(r: dict, forms: list, texts: list, audit: list) -> None:
    ed = r["editorial"]
    action = ed.get("Action", "")
    page = r["page"]
    gloss = "; ".join(g for g in r["glosses"] if g)
    tags = visible_tags(r["tags"])
    dialect = [] if r["bracketed"] else [OAV_TAG]
    raw = raw_text(r["unit"])
    base = {"Unit_ID": r["uid"], "Unit_Kind": r["kind"], "PDF_Page": r["unit"]["pdf_page"],
            "Printed_Page": page, "Column": r["unit"]["col"], "Lemma_Key": "",
            "Grammar_Code": "", "Passages": "", "Target": "", "Raw_Text": raw,
            "Parsed_Gloss": gloss, "Parsed_Tags": " ".join(r["tags"])}
    review = []
    if r["bracketed"]:
        review.append("non-Old-Avestan lemma printed in square brackets: base Av, no OAv tag")
    if "preverb" in r["tags"] and not EMIT_PROPOSED_TAGS:
        review.append("proposed tag preverb withheld until registered")
    review.extend(r.get("flags", []))
    if r.get("root_review"):
        review.append("root " + r["root_review"])
    if ed.get("Note"):
        review.append("editorial: " + ed["Note"])
    if action in ("subsection", "exclude"):
        audit.append(dict(base, Status=f"audit_{action}", Reason=ed.get("Note", action),
                          Entry_Key="", Lemma_Key=r.get("key", ""),
                          Parsed_Form=" | ".join(r["headwords"]),
                          Parent_Keys="", Review="; ".join(review),
                          Record_SHA256=_sha(r["uid"], raw)))
        return
    key = r["key"]
    head = r["headwords"][0]
    row_tags = " ".join(tags + dialect)
    forms.append(dict(zip(FORM_FIELDS, [
        LANGUAGE_ID, "", head, gloss, "", "", "", locator(page), "", "", key, "", "",
        "|".join(r["parents"]), row_tags])))
    status = "installed_lemma_editorial" if ed else "installed_lemma"
    if not gloss:
        review.append("no lexical gloss printed (intentional blank)")
    audit.append(dict(base, Status=status, Reason="printed headword", Entry_Key=key,
                      Parsed_Form=head, Parent_Keys="|".join(r["parents"]),
                      Review="; ".join(review), Record_SHA256=_sha(r["uid"], raw)))
    blocks = []
    if r["residue"]:
        blocks.append(("comparison", " ".join(r["residue"])))
    notes = r["braces"] + r["readings"]
    if notes:
        blocks.append(("source-note", " ".join(notes)))
    for pos, (kind, content) in enumerate(blocks, 1):
        texts.append(dict(zip(TEXT_FIELDS, ["", key, pos, kind, "text", content,
                                             locator(page)])))
    gender = {"m", "f", "n", "mn", "mf", "fn"}
    variants = [("alt", v, None) for v in r["headwords"][1:]] + \
        [("or", v, None) for v in r["variants_or"]] + \
        [("alt", norm_headword(v), own) for v, own in r.get("gendered", [])]
    seen = {head}
    for how, form, own in variants:
        vtags = tags if own is None else \
            [t for t in tags if t not in gender] + [t for t in visible_tags(own) if t not in tags]
        vid = f"{r['uid']}:{how}:{form}"
        vbase = dict(base, Unit_ID=vid, Unit_Kind=f"variant-{how}", Lemma_Key=key,
                     Raw_Text="", Parsed_Form=form)
        if "°" in form or "˚" in form or form in seen:
            reason = ("abbreviated alternant (°) is not a full form" if ("°" in form or "˚" in form)
                      else "repeats the headword")
            audit.append(dict(vbase, Status="audit_variant_not_installed", Reason=reason,
                              Entry_Key="", Parent_Keys="", Review="",
                              Record_SHA256=_sha(vid)))
            continue
        seen.add(form)
        vkey = f"{key}:{how}:{form}"
        forms.append(dict(zip(FORM_FIELDS, [
            LANGUAGE_ID, "", form, gloss, "", "", "", locator(page), "", "", vkey, key, "",
            "", " ".join(vtags + ["alternate"] + dialect)])))
        reason = ("printed alternant headword" if how == "alt"
                  else "printed '(or …)' alternative spelling")
        audit.append(dict(vbase, Status="installed_variant", Reason=reason, Entry_Key=vkey,
                          Parent_Keys="", Review="", Record_SHA256=_sha(vid)))


def audit_other(unit: dict, uid: str, lemma: dict | None) -> dict:
    raw = raw_text(unit)
    kind = unit["kind"]
    lemma_key = lemma.get("key", "") if lemma else ""
    row = {"Unit_ID": uid, "Unit_Kind": kind, "PDF_Page": unit["pdf_page"],
           "Printed_Page": unit["printed_page"], "Column": unit["col"], "Entry_Key": "",
           "Lemma_Key": lemma_key, "Parsed_Form": "", "Parsed_Gloss": "", "Parsed_Tags": "",
           "Grammar_Code": "", "Passages": "", "Target": "", "Parent_Keys": "", "Review": "",
           "Raw_Text": raw, "Record_SHA256": _sha(uid, raw)}
    if kind == "attestation":
        a = parse_attestation(unit)
        row.update(Status="audit_attestation", Parsed_Form=a["form"], Grammar_Code=a["code"],
                   Passages=" ".join(a["passages"]),
                   Reason="inflected attestation with passage citations; lemmata only are "
                          "installed",
                   Review="bracketed non-Old-Avestan passage" if a["bracketed"] else "")
        if not lemma_key:
            row["Review"] = (row["Review"] + "; " if row["Review"] else "") + \
                "attestation of a lemma that is not installed"
    elif kind == "xref":
        x = parse_xref(unit)
        row.update(Status="audit_cross_reference", Parsed_Form=" | ".join(x["forms"]),
                   Target=" | ".join(x["target_bold"]),
                   Reason="F pointer from a form to its lemma; not a lexical record",
                   Lemma_Key="")
    elif kind == "stem":
        row.update(Status="audit_stem_heading", Parsed_Form=raw.strip(),
                   Reason="verb-stem sub-heading inside a root entry")
    elif kind == "section":
        row.update(Status="audit_section_heading", Parsed_Form=raw.strip(),
                   Reason="alphabet section heading", Lemma_Key="")
    else:
        row.update(Status="audit_unparsed", Reason="text outside any lemma")
    return row
