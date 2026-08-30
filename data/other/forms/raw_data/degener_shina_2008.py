#!/usr/bin/env python3
"""Install the complete glossary of Degener's 2008 Gilgit Shina text collection.

Almuth Degener, *Shina-Texte aus Gilgit (Nord-Pakistan): Sprichwörter und
Materialien zum Volksglauben, gesammelt von Mohammad Amin Zia* (Beiträge zur
Indologie 41, Wiesbaden: Harrassowitz), Glossar pp. 243--315.

The copyrighted Stanford ILL scans (two complementary deliveries) are not
redistributed. The checked-in raw layer is
``20260827-degener-shina-transcription.txt``: a complete, manually collated
verbatim transcription of every glossary paragraph, produced against 300 dpi
renders with 2x-zoom band verification; a Tesseract pass was used only for
digit cross-checks. ``H|`` lines are headword paragraphs, ``C|`` indented
sub-paragraphs (attestation numbers and inflected forms — retained in the
audit as source prose, not installed as rows, following the Buddruss
Waigali/Wama precedent), ``X|``/``XC|`` page-break continuations.

``20260827-degener-shina-editorial.csv`` carries the editorial layer: the
English gloss for every entry and per-entry resolutions for cross-references
and gloss-less entries.

Only printed headwords and explicit headline alternates become rows. Direct,
unhedged ``[T. N]`` assignments are linked to CDIAL; hedged claims (``zu``,
``vgl.``, ``?``), multi-number brackets, and all Burushaski/Indus-Kohistani
comparanda remain prose. A printed decimal sub-number (e.g. ``T. 11503.3``)
links to its integer CDIAL parent with the exact printed id preserved in the
prose. Pure ``s.`` cross-reference entries with a unique resolvable target are
installed as variants of the target; unresolvable ones stay audit-only.

Run from ``data/``; ``--pdf1``/``--pdf2`` optionally verify the scans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SOURCE_ID = "degener-shina2008"
SNAPSHOT_DATE = "2026-08-27"
COLLATION_DATE = "2026-08-27"
PDF1_SHA256 = "9589572ce61d21062454cfa54537c881080e9288bc55292f176fc12ca6264fd5"
PDF2_SHA256 = "11bb98531bf601c59f5252d361774320bccd0043c627d8c96094df5320517ec4"
PDF1_PAGES, PDF2_PAGES = 46, 38

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
TRANSCRIPTION = RAW_DIR / "20260827-degener-shina-transcription.txt"
EDITORIAL = RAW_DIR / "20260827-degener-shina-editorial.csv"
FORM_OUTPUT = ROOT / "data/other/forms/20260827-degener-shina.csv"
AUDIT_OUTPUT = RAW_DIR / "20260827-degener-shina-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260827-degener-shina-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260827-degener-shina-manifest.json"
CDIAL_CSV = ROOT / "data/cdial/cdial.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Unit_ID", "Printed_Page", "Scan",
    "Raw_Headline", "Raw_Subparagraphs", "Raw_Gloss_German", "English_Gloss",
    "Headwords", "Labels", "Turner_Claim", "Final_Status", "Final_Form",
    "Final_Parameter_ID", "Emitted_Keys", "Resolution", "Review",
    "Material_Error", "Source", "Record_SHA256",
]

LABELS = {
    "m.": ["m"], "fem.": ["f"], "fem": ["f"], "adj.": ["adj"], "adv.": ["adv"],
    "itr.": ["intr"], "trans.": ["tr"], "postpos.": ["postp"],
    "Partikel": ["part"], "Partikel:": ["part"],
    "Interrogativ-Partikel": ["part", "interr"],
    "Interrogativ-Partikel:": ["part", "interr"],
    "Präposition": ["prep"], "Konjunktion": ["conj"], "Konj.": ["conj"],
    "Kaus.": ["caus"], "kaus.": ["caus"], "pl": ["pl"], "(pl)": ["pl"],
    "EN": ["proper-noun"], "num.": ["num"], "pron.": ["pron"],
    "interj.": ["interj"], "subst.": ["noun"], "m./": ["m"], "sg": ["sg"],
    "ḿ.": ["m"],  # misprint for ``m.`` on p. 298, kept verbatim upstream
    "dem.": ["demonstrative"], "ordin.": ["ord"], "indef.": ["indef"],
    "EN.": ["proper-noun"], "Partikel.": ["part"], "ON": ["proper-noun"],
    "ON.": ["proper-noun"],
}
GERMAN_HINTS = re.compile(
    r"^(?:[A-ZÄÖÜ][a-zäöüß]|ein|eine[mnrs]?|sich|etwas|jmd|nur|auch|sehr|so|als|"
    r"der|die|das|dem|den|des|und|oder|zu[rm]?|mit|ohne|bei|nach|von|vor|über|"
    r"unter|gegen|durch|für|aus|an|auf|in|im|am|nicht|wie|wenn|hier|dort|alle[sr]?)"
)
SHINA_SHAPE = re.compile(
    r"[áéíóúãẽĩõũïc̣ċčǰšžŋγ~]|aa|ee|ii|oo|uu|[̣́̃]"
)
LIGHT_VERBS = {"th-", "b-", "d-", "ho-", "e-", "the-", "de-", "nikha-"}
CROSS_REF = re.compile(r"^(?P<hw>\S+(?:, \S+)*):? s\. (?P<target>[^.]+)\.?$")
EQUALS_REF = re.compile(r"^(?P<hw>\S+) = (?P<target>\S+?)\.?$")
TURNER_DIRECT = re.compile(r"^T\. (\d+(?:\.\d+)?)$")
UNCERTAIN_MARK = re.compile(r"⟦([^⟧|?]*)(?:\|[^⟧]*)?\??⟧")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def parse_transcription() -> list[dict]:
    """Stitch the page files into glossary entries in print order."""
    entries: list[dict] = []
    page = None
    idx = 0
    for line in TRANSCRIPTION.read_text(encoding="utf-8").splitlines():
        if line.startswith("# page "):
            page = int(line.split()[2])
            idx = 0
            continue
        if not line.strip() or line.startswith("#"):
            continue
        tag, _, payload = line.partition("|")
        payload = _nfc(payload.strip())
        if tag == "H":
            idx += 1
            entries.append({"page": page, "idx": idx, "head": payload, "subs": []})
        elif tag == "C":
            entries[-1]["subs"].append(payload)
        elif tag == "X":
            entries[-1]["head"] += " " + payload
        elif tag == "XC":
            entries[-1]["subs"].append(payload)
        else:
            raise ValueError(f"p{page}: unexpected line {line[:60]!r}")
    return entries


def split_headline(head: str):
    """Split a headword paragraph into (headwords, labels, gloss_de, brackets,
    trailing_cross_ref, flags)."""
    flags = []
    brackets = re.findall(r"\[[^\]]*\]", head)
    if head.count("[") != head.count("]"):
        flags.append("unbalanced-brackets")
    rest = re.sub(r"\s*\[[^\]]*\]\.?", "", head).strip()
    m = CROSS_REF.match(rest) or EQUALS_REF.match(rest)
    if m and not brackets:
        return ([h.rstrip(",:") for h in m.group("hw").split(", ")], [], "", [],
                m.group("target"), ["cross-ref"])
    tokens = rest.split()
    headwords, labels, i = [], [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.rstrip(",;/") in LABELS or tok in LABELS:
            break
        bare = tok.rstrip(",;:").rstrip("./")
        if i == 0:
            headwords.append(bare)
            i += 1
            continue
        prev_had_comma = tokens[i - 1].endswith(",")
        if not prev_had_comma and tok in LIGHT_VERBS:
            # a light verb joins the preceding noun into one construction
            headwords[-1] += " " + bare
            i += 1
            continue
        if prev_had_comma and (tok in LIGHT_VERBS or bare.endswith("-")) \
                and not GERMAN_HINTS.match(bare):
            headwords.append(bare)
            i += 1
            continue
        if not prev_had_comma and bare.endswith("-") and i <= 2 \
                and not GERMAN_HINTS.match(bare):
            headwords.append(bare)
            i += 1
            continue
        if prev_had_comma and SHINA_SHAPE.search(bare) and not GERMAN_HINTS.match(bare):
            headwords.append(bare)
            i += 1
            continue
        if tok.startswith("(+") and i <= 3:
            j = i
            while j < len(tokens) and not tokens[j].endswith(")"):
                j += 1
            labels.append(" ".join(tokens[i:j + 1]))
            i = j + 1
            continue
        break
    while i < len(tokens) and (tokens[i].rstrip(",;/") in LABELS
                               or tokens[i] in LABELS
                               or tokens[i].startswith("(+")):
        if tokens[i].startswith("(+"):
            j = i
            while j < len(tokens) and not tokens[j].endswith(")"):
                j += 1
            labels.append(" ".join(tokens[i:j + 1]))
            i = j + 1
        else:
            tok = tokens[i]
            labels.append(tok if tok in LABELS else tok.rstrip(",;/"))
            i += 1
    gloss = " ".join(tokens[i:]).strip().lstrip(":").strip()
    trailing = ""
    sm = re.search(r"\bs\. (?P<t>[^.]+)\.?$", gloss)
    if sm and not gloss.startswith("s."):
        trailing = sm.group("t")
        gloss = gloss[: sm.start()].rstrip()
    if not headwords:
        flags.append("no-headword")
    if not gloss and not trailing and "cross-ref" not in flags:
        flags.append("no-gloss")
    return headwords, labels, gloss.rstrip("."), brackets, trailing, flags


def classify_brackets(brackets: list[str], cdial_ids: set[str]):
    """Return (parameter_id, printed_id, loan, uncertain_reading).

    Links a Turner claim only when the whole entry prints exactly one
    unhedged, stand-alone ``T. N`` part (position in the bracket does not
    matter: ``[Bur. oq, T. 2538]`` is still a direct assignment). A bare
    number directly after a Turner part is a second etymon (``T. 145, 887``)
    and blocks the link; a bare number after a work citation is a page
    (``vgl. Berger 1983, 32``) and does not. Hedged parts (``zu``, ``vgl.``,
    ``?``) never link. For a printed decimal sub-number the integer CDIAL
    parent is linked."""
    loan = any(b[1:-1].strip().startswith("←") for b in brackets)
    uncertain = any("⟦" in b for b in brackets)
    turner_ids, blocked = [], False
    for b in brackets:
        inner = b[1:-1].strip().rstrip(".")
        # an internal grammar reference does not hedge the claim
        inner = re.sub(r"\.\s*Gramm\. [0-9.]+$", "", inner)
        parts = [p.strip() for p in re.split(r"[;,]", inner) if p.strip()]
        previous_was_turner = False
        for part in parts:
            tm = TURNER_DIRECT.match(part)
            if tm:
                turner_ids.append(tm.group(1))
                previous_was_turner = True
                continue
            if previous_was_turner and re.fullmatch(r"\d+(?:\.\d+)?", part):
                blocked = True  # a second bare etymon number: ambiguous
            if re.search(r"(?:^|\s)(?:zu|vgl\.)\s+T\. \d", part) or \
                    re.search(r"T\. \d[\d.]*\s*\?", part):
                blocked = True
            previous_was_turner = False
    parameter, printed = "", ""
    if len(turner_ids) == 1 and not blocked:
        printed = turner_ids[0]
        candidate = printed.split(".", 1)[0]
        if candidate in cdial_ids:
            parameter = candidate
    elif len(turner_ids) > 1:
        printed = ""
    return parameter, printed, loan, uncertain


def canonical_tags(labels: list[str], loan: bool, uncertain: bool) -> list[str]:
    tags: list[str] = []
    for label in labels:
        for tag in LABELS.get(label, []):
            if tag not in tags:
                tags.append(tag)
    if loan and "loanword" not in tags:
        tags.append("loanword")
    if uncertain and "uncertain" not in tags:
        tags.append("uncertain")
    return tags


def clean_prose(text: str) -> str:
    """Render transcription uncertainty markup readably for installed prose."""
    return UNCERTAIN_MARK.sub(lambda m: f"{m.group(1)}(?)", text)


def load_editorial() -> dict[str, dict[str, str]]:
    with EDITORIAL.open(encoding="utf-8", newline="") as handle:
        return {row["Key"]: row for row in csv.DictReader(handle)}


def load_cdial_ids() -> set[str]:
    ids = set()
    with CDIAL_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) > 1 and row[1]:
                ids.add(row[1])
    return ids


def short_key(entry: dict) -> str:
    return f"p{entry['page']}:e{entry['idx']:02d}"


def key_of(entry: dict) -> str:
    return f"{SOURCE_ID}:{short_key(entry)}"


def locator(entry: dict) -> str:
    return f"{SOURCE_ID}[p. {entry['page']}, glossary entry {entry['idx']}]"


def write_csv(path: Path, fields: list[str], rows: list[dict], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    entries = parse_transcription()
    editorial = load_editorial()
    cdial_ids = load_cdial_ids()

    parsed = []
    headword_to_key: dict[str, list[str]] = {}
    for entry in entries:
        hws, labels, gloss_de, brackets, trailing, flags = split_headline(entry["head"])
        ed = editorial.get(short_key(entry), {})
        if ed.get("Headwords_Override"):
            hws = [h.strip() for h in ed["Headwords_Override"].split("|")]
        if ed.get("Labels_Override"):
            labels = ed["Labels_Override"].split()
        if ed.get("Gloss_De_Override"):
            gloss_de = ed["Gloss_De_Override"]
        if ed.get("Gloss_De_Override") or ed.get("Headwords_Override"):
            flags = [f for f in flags if f != "no-gloss"]
        parameter, printed, loan, uncertain = classify_brackets(brackets, cdial_ids)
        record = dict(entry, hws=hws, labels=labels, gloss_de=gloss_de,
                      brackets=brackets, trailing=trailing, flags=flags,
                      parameter=parameter, printed=printed, loan=loan,
                      uncertain=uncertain, key=key_of(entry))
        parsed.append(record)
        if "cross-ref" not in flags:
            for hw in hws:
                headword_to_key.setdefault(hw, []).append(record["key"])

    forms, audit = [], []
    status_counts: Counter[str] = Counter()
    for record in parsed:
        ed = editorial.get(short_key(record), {})
        english = ed.get("English_Gloss", "")
        resolution = ed.get("Resolution", "")
        emitted: list[str] = []
        cross_ref = "cross-ref" in record["flags"]
        target_key = ""
        if cross_ref or record["trailing"]:
            targets = [t.strip().rstrip(".") for t in
                       re.split(r",| oder ", record["trailing"])] if record["trailing"] else []
            targets = [t for t in targets if t]
            if cross_ref and len(targets) == 1:
                # a cross-reference points at a headword, possibly a light-verb
                # construction; resolve on the full printed target
                candidates = headword_to_key.get(targets[0], [])
                if not candidates:
                    candidates = headword_to_key.get(targets[0].split()[0], [])
                if not candidates:
                    # conservative diacritic-insensitive fallback (unique only)
                    import unicodedata as _ud

                    def _fold(text):
                        return "".join(
                            ch for ch in _ud.normalize("NFD", text)
                            if not _ud.combining(ch) or ch == "\u0323")
                    want = _fold(targets[0])
                    candidates = [k for hw, ks in headword_to_key.items()
                                  if _fold(hw) == want for k in ks]
                if len(candidates) == 1:
                    target_key = candidates[0]
                    if not english:
                        english = editorial.get(target_key.removeprefix(
                            SOURCE_ID + ":"), {}).get("English_Gloss", "")

        page_scan = "TN446831" if record["page"] <= 279 else "TN447377"
        etym_brackets = [b for b in record["brackets"]
                         if not b[1:-1].strip().startswith("Gramm.")]
        note_brackets = [b for b in record["brackets"]
                         if b[1:-1].strip().startswith("Gramm.")]
        etymology_bits = []
        if record["parameter"]:
            claim = (f"Degener directly assigns this headword to Turner/CDIAL "
                     f"{record['printed']}.")
            if record["printed"] != record["parameter"]:
                claim += (f" The printed sub-number {record['printed']} is linked "
                          f"to its CDIAL parent entry {record['parameter']}.")
            etymology_bits.append(claim)
        if etym_brackets:
            etymology_bits.append(clean_prose(" ".join(etym_brackets)))
        if record["trailing"] and not cross_ref:
            etymology_bits.append(f"Cross-reference: s. {record['trailing']}.")
        etymology = " ".join(etymology_bits)
        notes_bits = [record["gloss_de"]]
        notes_bits += [l for l in record["labels"] if l.startswith("(+")]
        notes_bits += note_brackets
        notes = " ".join(bit for bit in notes_bits if bit)

        tags = canonical_tags(record["labels"], record["loan"], record["uncertain"])
        tag_field = " ".join(tags + ["dialect:Sh:gil:Gilgit"])

        if cross_ref:
            if target_key:
                status = "installed_cross_reference_variant"
                emitted.append(record["key"])
                forms.append(dict(zip(FORM_FIELDS, [
                    "Sh", "", record["hws"][0], english, "", "",
                    f"printed as a cross-reference: s. {record['trailing']}.",
                    locator(record), "", "", record["key"], target_key, "", "",
                    tag_field,
                ])))
                for ordinal, alt in enumerate(record["hws"][1:], start=2):
                    alt_key = f"{record['key']}:v{ordinal}"
                    emitted.append(alt_key)
                    forms.append(dict(zip(FORM_FIELDS, [
                        "Sh", "", alt, english, "", "",
                        f"printed as a cross-reference: s. {record['trailing']}.",
                        locator(record), "", "", alt_key, target_key, "", "",
                        tag_field,
                    ])))
            else:
                status = "audit_only_cross_reference"
        else:
            status = "installed_form"
            emitted.append(record["key"])
            forms.append(dict(zip(FORM_FIELDS, [
                "Sh", record["parameter"], record["hws"][0], english, "", "",
                notes, locator(record), "", etymology, record["key"], "", "", "",
                tag_field,
            ])))
            for ordinal, alt in enumerate(record["hws"][1:], start=2):
                alt_key = f"{record['key']}:v{ordinal}"
                emitted.append(alt_key)
                forms.append(dict(zip(FORM_FIELDS, [
                    "Sh", record["parameter"], alt, english, "", "", notes,
                    locator(record), "", etymology, alt_key, record["key"], "", "",
                    " ".join(tags + ["alternate", "dialect:Sh:gil:Gilgit"]),
                ])))
        status_counts[status] += 1

        payload = "|".join([record["head"], " ⁋ ".join(record["subs"])]).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Collation_Date": COLLATION_DATE,
            "Unit_ID": f"p{record['page']}:e{record['idx']:02d}",
            "Printed_Page": str(record["page"]), "Scan": page_scan,
            "Raw_Headline": record["head"],
            "Raw_Subparagraphs": " ⁋ ".join(record["subs"]),
            "Raw_Gloss_German": record["gloss_de"], "English_Gloss": english,
            "Headwords": " | ".join(record["hws"]),
            "Labels": " | ".join(record["labels"]),
            "Turner_Claim": record["printed"],
            "Final_Status": status,
            "Final_Form": record["hws"][0] if record["hws"] else "",
            "Final_Parameter_ID": record["parameter"],
            "Emitted_Keys": " ".join(emitted),
            "Resolution": resolution or (
                "manually collated printed glossary headword paragraph"),
            "Review": ("transcription uncertainty marker in comparanda"
                       if record["uncertain"] else
                       "full manual census against 300 dpi renders with zoomed "
                       "band verification; Tesseract digit cross-check"),
            "Material_Error": "no",
            "Source": locator(record),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    return entries, parsed, forms, audit, status_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf1", type=Path)
    parser.add_argument("--pdf2", type=Path)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    for arg, want in ((args.pdf1, PDF1_SHA256), (args.pdf2, PDF2_SHA256)):
        if arg and sha256(arg) != want:
            raise ValueError(f"unexpected scan SHA-256 for {arg}")

    entries, parsed, forms, audit, status_counts = build()
    editorial = load_editorial()
    keys = [row["Entry_Key"] for row in forms]
    assert len(keys) == len(set(keys))
    assert {r["page"] for r in parsed} == set(range(243, 316))
    explained = {key_of(r) for r in parsed
                 if "gloss-less" in editorial.get(
                     short_key(r), {}).get("Resolution", "")
                 or "equivalence" in editorial.get(
                     short_key(r), {}).get("Resolution", "")}
    missing_gloss = [row["Entry_Key"] for row in forms
                     if not row["Gloss"] and "proper-noun" not in row["Tags"]
                     and row["Entry_Key"].rsplit(":v", 1)[0] not in explained]

    if not args.install:
        print(f"parsed {len(entries)} entries -> {len(forms)} rows "
              f"({dict(status_counts)}); {len(missing_gloss)} rows without an "
              f"English gloss: {missing_gloss[:10]}")
        return

    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(json.dumps({
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": ("Degener, Almuth. 2008. Shina-Texte aus Gilgit "
                         "(Nord-Pakistan): Sprichwörter und Materialien zum "
                         "Volksglauben, gesammelt von Mohammad Amin Zia. "
                         "(Beiträge zur Indologie 41.) Wiesbaden: Harrassowitz."),
        "acquisition": ("Stanford Interlibrary Loan scan requests 446831 "
                        "(printed pp. 243–279) and 447377 (printed pp. 280–315), "
                        "two complementary web deliveries of the complete Glossar"),
        "pdf_sha256": {"TN446831": PDF1_SHA256, "TN447377": PDF2_SHA256},
        "pdf_pages": {"TN446831": PDF1_PAGES, "TN447377": PDF2_PAGES},
        "glossary_printed_pages": [243, 315],
        "pdf_redistributed": False,
        "rights": ("Copyrighted ILL scans supplied for private study, "
                   "scholarship, or research; the scans are not checked in."),
        "extraction": {
            "method": ("complete paragraph-by-paragraph manual collation of the "
                       "printed glossary against 300 dpi renders with 2x zoomed "
                       "band verification"),
            "ocr_reproducibility": [
                "tesseract -l deu+eng --psm 4 (digit cross-check only)",
            ],
            "checked_in_layer": str(TRANSCRIPTION.relative_to(ROOT)),
            "editorial_layer": str(EDITORIAL.relative_to(ROOT)),
            "glossary_entry_count": len(entries),
            "transcription_uncertain_headword_records": sum(
                1 for r in parsed if r["uncertain"]),
            "transcription_uncertain_readings": sum(
                line.count("⟦")
                for line in TRANSCRIPTION.read_text(encoding="utf-8").splitlines()
                if line.startswith(("H|", "C|", "X|", "XC|"))
            ),
            "normalizations": [
                "j + combining breve folded to ǰ (U+01F0): the italic font "
                "renders the haček on dotless j as a rounded arc",
            ],
        },
        "scope": {
            "included": ("every printed headword paragraph of the Glossar, "
                         "pp. 243–315, including light-verb constructions, "
                         "proper names, and cross-reference headwords"),
            "excluded": ("the proverb and folk-belief texts themselves; "
                         "attestation numbers and inflected forms in indented "
                         "sub-paragraphs (kept as audit prose); Burushaski, "
                         "Indus Kohistani, and other comparanda inside "
                         "etymology brackets; the Literaturverzeichnis"),
            "cdial_policy": ("direct unambiguous [T. N] assignments are linked; "
                             "hedged (zu/vgl./?), multi-number, and secondary "
                             "claims remain prose; printed decimal sub-numbers "
                             "link to their integer CDIAL parent"),
            "language_model": ("all forms belong to canonical Shina (Sh) and "
                               "carry the registered Gilgit dialect tag "
                               "(dialect:Sh:gil:Gilgit); Mohammad Amin Zia's "
                               "collection and cited informants remain "
                               "provenance"),
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)), "form_count": len(forms),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)), "audit_count": len(audit),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)), "sample_count": len(sample),
        },
        "status_counts": dict(status_counts),
        "unresolved": sorted(
            row["Unit_ID"] for row in audit
            if row["Final_Status"] == "audit_only_cross_reference"),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"installed {len(forms)} rows from {len(entries)} glossary entries "
          f"({dict(status_counts)})")


if __name__ == "__main__":
    main()
