#!/usr/bin/env python3
"""Install the complete final glossary in Aaley's 2021 *Kusunda Gipan*.

The PDF is not redistributed.  The frozen table preserves both embedded Preeti
glyph codes and their checked Unicode Devanagari decoding.  ``Form`` is a
deterministic graphemic romanization, not a claim of narrow phonetic analysis;
the printed Kusunda spelling remains in ``Native``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PACKAGE = Path(__file__).resolve().parent
SNAPSHOT = PACKAGE / "snapshot/glossary.tsv"
MANIFEST = PACKAGE / "manifest.json"
INSTALLED = ROOT / "data/other/forms/20260901-aaley-kusunda-gipan.csv"
AUDIT = PACKAGE / "20260901-aaley-kusunda-gipan-audit.csv"
SOURCE_KEY = "aaley2021kusundagipan"
PDF_SHA256 = "0b78336d6d7173c2ac18ab153fcda221fa0351cb708b42ad66c784c947b9bcd2"
SAMPLE_SEED = 20260901

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "PDF_Page", "Printed_Page", "Column", "Row", "Raw_Form", "Raw_Gloss",
    "Form_Devanagari", "Gloss_Nepali", "Variant_Index", "Form", "Gloss",
    "Status", "Reason", "Entry_Key", "Variant_Of_Key", "Citation",
    "Review_State",
]

# Source spellings such as भषा, क्पाल, ऋाज, and ऋामा are retained
# verbatim in the audit.  This table translates their evident intended Nepali
# senses; it does not silently repair the source text.
ENGLISH = {
    "कुकुर": "dog", "बाख्रा": "goat", "मासु": "meat", "आँप": "mango",
    "लसुन": "garlic", "ढोका": "door", "पिसाब": "urine", "तल": "below",
    "रुख": "tree", "सूर्य": "sun", "रात": "night", "जिब्रो": "tongue",
    "भोक": "hunger", "आँखा": "eye", "आँखाको नानी": "pupil of the eye", "नाक": "nose",
    "मकै": "maize", "टाउको": "head", "क्पाल": "forehead", "बारी": "cultivated field",
    "त्यता": "there", "बाटो": "path", "घुँडा": "knee", "केटी": "girl",
    "घर": "house", "छाती": "chest", "दिउँसो": "daytime", "सुल्पा": "smoking pipe",
    "ठूलो": "big", "छैन": "not exist", "मीठो": "sweet", "बेसार": "turmeric",
    "पहेँलो": "yellow", "फेरि": "again", "हवा": "air", "अरू": "other",
    "देउता": "deity", "गङगटो": "crab", "बिच्छी": "scorpion", "भात": "cooked rice",
    "तीतो": "bitter", "सिरानी": "pillow", "हलुको": "light (not heavy)", "एक": "one",
    "साज": "instrument", "सेतो": "white", "जुम्रा": "louse", "सुसेली": "whistle",
    "उल्लु": "owl", "नराम्रो": "bad", "चिसो पानी": "cold water", "रातो": "red",
    "डर": "fear", "काँडा": "thorn", "शरीर": "body", "नम": "name",
    "ऊ, त्यो": "he, she; that", "भषा": "language", "फूल": "flower", "साथी": "friend",
    "आफैँ": "self", "मोटो": "fat", "पेट": "belly", "झोल": "liquid; broth",
    "भोलि": "tomorrow", "गोरु": "ox", "अण्डा": "egg", "तातो": "hot",
    "कक्कड": "cucumber", "घाउ": "wound", "म्": "I", "बुढी महिला": "old woman",
    "आगो": "fire", "वन मान्छे": "wild man", "खरानी": "ashes", "ऋाज": "today",
    "मौरी": "bee", "पानी": "water", "चल्ला": "chick", "सर्प": "snake",
    "हामी": "we", "ठाउँ": "place", "दाउरा": "firewood", "अहिले": "now",
    "तीन": "three", "लोग्ने": "husband", "छोरा": "son", "दुई": "two",
    "माटो": "soil", "बुढो": "old man", "को": "of", "कसको": "whose",
    "माछा": "fish", "चन्द्रमा": "moon", "छोरी": "daughter", "तिमी": "you (singular)",
    "तिमीहरू": "you (plural)", "मावली बजै": "maternal grandmother", "माथि": "above", "गाई": "cow",
    "बज्य ै": "grandmother", "ममा": "maternal uncle", "माकुरो": "spider", "पाँच": "five",
    "चार": "four", "छेपारो": "lizard", "हिजो": "yesterday", "कपडा": "clothes",
    "होचो, छोटो": "short", "घोडा": "horse", "नरम": "soft", "बाहिर": "outside",
    "क्मिला": "ant", "सुर्तीको पात": "tobacco leaf", "भाइ": "younger brother", "धेरै": "many",
    "ढिलो": "late", "भैँसी": "buffalo", "ऋामा": "mother", "सपना": "dream",
    "गीत": "song", "दाजु": "elder brother", "रोटी": "flatbread", "टोपी": "cap",
    "राजा": "king", "बाघ": "tiger", "तरुल": "yam", "दिसा": "feces",
    "चिसो मौसम": "cold weather", "पुच्छर": "tail", "खुट्टा": "foot; leg", "बुबा": "father",
    "पाकेको": "ripe", "कान": "ear", "लामो": "long", "रगत": "blood",
    "सिस्नो": "nettle", "गह्रुङगो": "heavy", "राम्रो": "good", "तारा": "star",
    "काइँयो": "comb", "जाँड": "rice beer", "धागो": "thread", "भालु": "bear",
    "कालो": "black", "सबै": "all", "पात": "leaf", "नोट, पैसा": "money",
    "अनुहार": "face", "सुँगुर": "pig", "नुन": "salt", "नुनिलो": "salty", "टाढा": "far",
}

# The source's six-vowel chart uses अ for central /ə/ and आ for /a/;
# these are vowel qualities, not an Indic short/long opposition.
INDEPENDENT = {"अ": "ə", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "r", "ए": "e", "ओ": "o"}
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ", "च": "c", "छ": "ch",
    "ज": "j", "झ": "jh", "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph",
    "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "w",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
}
MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au"}


def romanize(value: str) -> str:
    out: list[str] = []
    chars = list(unicodedata.normalize("NFC", value))
    i = 0
    while i < len(chars):
        char = chars[i]
        if char in CONSONANTS:
            out.append(CONSONANTS[char])
            if i + 1 < len(chars) and chars[i + 1] == "्":
                i += 1
            elif i + 1 < len(chars) and chars[i + 1] in MATRAS:
                i += 1
                out.append(MATRAS[chars[i]])
            # Like Nepali orthography, the source suppresses the inherent
            # central vowel on a word-final consonant without printing virama.
            elif i + 1 == len(chars) or chars[i + 1] in {" ", ",", "/"}:
                pass
            else:
                out.append("ə")
        elif char in INDEPENDENT:
            out.append(INDEPENDENT[char])
        elif char == "ँ":
            out.append("\u0303")
        elif char in {" ", ",", "/"}:
            out.append(char)
        else:
            raise ValueError(f"Unmapped Devanagari character {char!r} in {value!r}")
        i += 1
    return unicodedata.normalize("NFC", "".join(out))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_snapshot() -> list[dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256(SNAPSHOT) != manifest["snapshot"]["sha256"]:
        raise ValueError("Gipan glossary snapshot checksum mismatch")
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 160:
        raise ValueError(f"Expected 160 glossary rows, found {len(rows)}")
    missing = sorted({row["Gloss_Nepali"] for row in rows} - ENGLISH.keys())
    if missing:
        raise ValueError(f"Missing English translations: {missing}")
    return rows


def citation(row: dict[str, str]) -> str:
    return f"{SOURCE_KEY}[p. {row['Printed_Page']}, col. {row['Column']}, row {row['Row']}]"


def transform(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    installed_keys: list[str] = []
    for row in rows:
        forms = [part.strip() for part in row["Form_Devanagari"].split("/")]
        base_key = (
            f"{SOURCE_KEY}:p{int(row['Printed_Page']):02d}:c{row['Column']}:"
            f"r{int(row['Row']):02d}:v01"
        )
        for variant_index, native in enumerate(forms, 1):
            form = romanize(native)
            gloss = ENGLISH[row["Gloss_Nepali"]]
            identity = (form, gloss)
            duplicate = identity in seen
            seen.add(identity)
            key = base_key[:-2] + f"{variant_index:02d}"
            variant_of = "" if variant_index == 1 else base_key
            tags = "sound-variant alternate" if variant_index > 1 else ""
            status = "excluded" if duplicate else "ingested"
            reason = "exact repeated form-and-gloss row" if duplicate else "printed glossary attestation"
            notes = (
                f"Printed Kusunda spelling: {native}; Nepali gloss: {row['Gloss_Nepali']}; "
                "Form is a graphemic romanization of the source Devanagari, not narrow IPA."
            )
            if not duplicate:
                installed.append([
                    "Kusunda", "", form, gloss, native, form, notes, citation(row),
                    "", "", key, variant_of, "", "", tags,
                ])
                installed_keys.append(key)
            audit.append({
                "PDF_Page": row["PDF_Page"], "Printed_Page": row["Printed_Page"],
                "Column": row["Column"], "Row": row["Row"],
                "Raw_Form": row["Raw_Form"], "Raw_Gloss": row["Raw_Gloss"],
                "Form_Devanagari": native, "Gloss_Nepali": row["Gloss_Nepali"],
                "Variant_Index": str(variant_index), "Form": form, "Gloss": gloss,
                "Status": status, "Reason": reason,
                "Entry_Key": "" if duplicate else key,
                "Variant_Of_Key": "" if duplicate else variant_of,
                "Citation": citation(row), "Review_State": "not-sampled",
            })
    sampled = set(random.Random(SAMPLE_SEED).sample(installed_keys, 20))
    for row in audit:
        if row["Entry_Key"] in sampled:
            row["Review_State"] = "verified-against-render"
    if len(installed) != 161 or len(audit) != 162:
        raise ValueError(f"Expected 161 installed / 162 audit, found {len(installed)} / {len(audit)}")
    return installed, audit


def verify_pdf(path: Path) -> None:
    if sha256(path) != PDF_SHA256:
        raise ValueError(f"Unexpected Gipan PDF checksum: {sha256(path)}")
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required only for --pdf verification") from exc
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 56:
            raise ValueError(f"Expected 56 PDF pages, found {len(pdf.pages)}")
        counts = [sum(len(table) - 1 for table in pdf.pages[i].extract_tables()) for i in (53, 54, 55)]
        if counts != [70, 74, 16]:
            raise ValueError(f"Unexpected glossary table census: {counts}")


def write_csv(path: Path, rows, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        else:
            csv.writer(stream, lineterminator="\n").writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.pdf:
        verify_pdf(args.pdf)
    installed, audit = transform(read_snapshot())
    output = args.output or (INSTALLED if args.install else Path("/tmp/gipan-kusunda-proposed.csv"))
    audit_output = args.audit_output or (AUDIT if args.install else Path("/tmp/gipan-kusunda-audit.csv"))
    write_csv(output, installed)
    write_csv(audit_output, audit, AUDIT_FIELDS)
    print(json.dumps({"source_rows": 160, "installed": len(installed), "audit_rows": len(audit), "output": str(output), "audit": str(audit_output)}, indent=2))


if __name__ == "__main__":
    main()
