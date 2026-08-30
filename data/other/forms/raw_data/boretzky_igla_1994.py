#!/usr/bin/env python3
"""Install the etymological appendix of Boretzky & Igla's 1994 Romani dictionary.

Norbert Boretzky and Birgit Igla, *Wörterbuch Romani-Deutsch-Englisch für den
südosteuropäischen Raum: Mit einer Grammatik der Dialektvarianten* (Wiesbaden:
Harrassowitz, 1994). Only the appendix ``Etymologien`` on printed pages 311--338
is ingested: the four alphabetical word lists ``Indische Etyma``,
``Iranische Etyma``, ``Armenische Etyma`` and ``Griechische Etyma``. The
A--Z dictionary proper (pp. 1--310), the two reverse indexes (pp. 339--362) and
the ``Variantengrammatik`` (pp. 363--417) are outside this ingest.

The scan is a copyrighted Harrassowitz volume and is not redistributed. The
checked-in transcription layer is
``20260825-boretzky-igla-etymologies-extract.psv``: every printed entry on those
28 pages was read off 400 dpi renders of the page images. Two Tesseract passes
(``--psm 3`` and ``--psm 6``, German model) were used only for navigation and
for discrepancy discovery; the page images, not the OCR, are authoritative,
because Tesseract does not reliably distinguish the source's ``č``/``ć`` and
``ž``/``ź`` or read its ``ř`` and ``ə`` at all.

What the source claims, and what is installed:

* Each list entry prints a Romani headword (sometimes several), an italic
  grammatical label, a German gloss, and a bracketed etymological note. The
  bracket is free-text scholarly prose and is preserved verbatim in
  ``Etymology``; it frequently rejects the etymology it cites.
* Entries in the Indic list that assert descent from an Old Indo-Aryan etymon
  are matched to CDIAL by conservatively normalised headword. A match is
  accepted only when it is unique and the bracket is not a rejection; every
  accepted, ambiguous and unmatched candidate is written to the audit.
* The Iranian, Armenian and Greek lists assert borrowing from donors that Jambu
  does not carry as nodes, so no graph edge can be built for them. Those claims
  stay in ``Etymology`` and the rows are installed unlinked.
* ``s. X`` cross-reference lines are pointers to a full entry elsewhere in the
  same appendix. They are not installed; the audit records each one.

Run from ``data/``::

    uv run python data/other/forms/raw_data/boretzky_igla_1994.py
    uv run python data/other/forms/raw_data/boretzky_igla_1994.py --install
    uv run python data/other/forms/raw_data/boretzky_igla_1994.py --pdf scan.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_ID = "boretzky1994romani"
SNAPSHOT_DATE = "2026-08-25"
COLLATION_DATE = "2026-08-25"
PDF_SHA256 = "2fc3678f13736e2c9f2f857a123803efde6955e56f2ad53ce4d3cb184771d6ee"
PDF_PAGES = 425
# 0-based PDF page i carries printed page i - 1; the appendix runs pp. 311-338.
PRINTED_PAGE_OFFSET = -1
FIRST_PRINTED_PAGE = 311
LAST_PRINTED_PAGE = 338
BASE_LANGUAGE_ID = "eur"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
EXTRACT = RAW_DIR / "20260825-boretzky-igla-etymologies-extract.psv"
FORM_OUTPUT = ROOT / "data/other/forms/20260825-boretzky-igla-etymologies.csv"
AUDIT_OUTPUT = RAW_DIR / "20260825-boretzky-igla-etymologies-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260825-boretzky-igla-etymologies-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260825-boretzky-igla-etymologies-manifest.json"
CDIAL_PARAMS = ROOT / "data/cdial/params.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Printed_Page", "PDF_Page", "Column", "Item",
    "List", "Letter", "Raw_Headword", "Raw_Grammar", "Raw_Gloss_German", "Raw_Etymology",
    "Unit_Form", "Unit_Relation", "Status", "Reason", "Entry_Key", "Language_ID",
    "Tags", "Gloss_English", "Source_Marks", "Citations", "Etymon_Cited",
    "Etymon_Status", "Etymon_Candidates", "Parameter_ID", "Review", "Record_SHA256",
]

LIST_NAMES = {
    "ind": "Indische Etyma",
    "iran": "Iranische Etyma",
    "arm": "Armenische Etyma",
    "gr": "Griechische Etyma",
}

# Printed abbreviations for works cited as the attestation of a headword that is not
# in the authors' own dictionary ("Bei Wörtern, die nicht in unserem Wörterbuch
# vorkommen, ist eine Quelle angegeben", p. 311). These are bibliographic provenance,
# not dialects, and become secondary CLDF citations.
SOURCE_MARKS = {
    "Sa": "sampson1926welsh",
    "So": "sowa1898zigeuner",
    "Thes": "thesleff1901finnland",
    "Thesleff": "thesleff1901finnland",
    "Rozw": "rozwadowski1936zakopane",
    "Col": "colocci1889zingari",
    "Finck": "finck1903zigeuner",
    "Ješ": "jesina1886romanicib",
    "Ješina": "jesina1886romanicib",
    "Lípa": "lipa1963cikanstina",
    "Bar": "barannikov1934gypsy",
    "Paspati": "paspati1870tchinghianes",
    "Bischoff": "bischoff1827woerterbuch",
    "Heinschink": "heinschink1989izmir",
}

# Printed dialect-group labels. Sinte Romani and the Balkan and Vlax groups are already
# canonical Jambu base languages; Caló reuses the registered Spanish-Romani dialect,
# whose Glottocode is calo1236.
DIALECT_MARKS = {
    "Sinti": ("RomSint", ""),
    "Arli": ("RomBalk", "dialect:RomBalk:arli:Arli"),
    "Bug.": ("RomBalk", "dialect:RomBalk:bugurdzi:Bugurd%C5%BEi"),
    "Urs": ("RomVlax", "dialect:RomVlax:ursari:Ursari"),
    "Caló": ("eur", "dialect:eur:sp:Spanish"),
}

# Italic grammatical labels -> canonical Jambu tags. ``part`` is Boretzky's participle
# label, not a particle; ``p-i`` is his mediopassive/intransitive ("Passiv-Intransitiv")
# derivation.
GRAMMAR_TAGS = {
    "m": "m noun", "f": "f noun", "n": "n noun",
    "m (f)": "mf noun", "f (m)": "mf noun", "m/f": "mf noun", "f/m": "mf noun",
    "m/(f)": "mf noun", "f, m": "mf noun", "f/ m": "mf noun", "pl/f": "f pl noun",
    "pl": "pl noun", "sg": "sg noun",
    "adj": "adj", "adj/adv": "adj adv", "adv": "adv", "adv komp": "adv",
    "tr": "verb tr", "itr": "verb intr", "tr/itr": "verb tr intr",
    "impers": "verb impersonal", "p-i": "verb intr pass",
    "num": "num", "pron": "pron", "pron poss": "pron poss", "pron refl": "pron refl",
    "pers pron": "pron personal", "interrog": "pron interr", "interj": "interj",
    "konj": "conj", "präp": "prep", "part": "participle", "ppp": "ppp",
    "neg": "neg", "vok": "voc noun", "imp": "impv verb", "refl": "verb refl",
    "aux, impers": "auxiliary verb impersonal",
    "verb": "verb",
    "präp + gen": "prep",
    "itr/tr": "verb intr tr",
    "part zu del": "participle",
    "prät zu džal": "verb pret",
    "p-i (part ušlo)": "verb intr pass",
    "Negation": "neg",
    "Negation des Imperativs": "neg part",
    "3.sg der Kopula, neg": "copula 3sg neg",
    "Verbalnomen": "suffix noun",
    "Anrede für männl./weibl. Person": "interj",
}

# Entries whose printed headword field bundles several lexemes, or whose gloss field
# carries a second printed headword. Each unit is (form, grammar, gloss_de, gloss_en,
# relation, marks); ``relation`` is "head", "variant" (printed alternate of the same
# lexeme) or "sibling" (a distinct lexeme sharing the entry's etymology).
CURATED = {
    ("313", "12"): [
        ("bilal", "itr", "schmelzen, auftauen", "to melt; to thaw", "head", ()),
        ("bilavel", "tr", "schmelzen, auftauen", "to melt; to thaw", "sibling", ()),
    ],
    ("313", "30"): [
        ("buke", "pl", "kalo buko Leber; parne buke Lunge",
         "liver (kalo buko); lungs (parne buke)", "head", ()),
        ("buko", "m", "kalo buko Leber; parne buke Lunge",
         "liver (kalo buko); lungs (parne buke)", "variant", ()),
    ],
    ("314", "07"): [
        ("čarjavel", "tr", "weiden", "to graze", "head", ()),
        ("čarjol", "itr", "weiden", "to graze", "sibling", ()),
    ],
    ("314", "22"): [
        ("čum", "f", "Kuß", "kiss", "head", ()),
        ("čumi-del", "verb", "küssen", "to kiss", "sibling", ()),
    ],
    # Three entries print no German definition: the italic label is the semantic content.
    ("315", "20"): [
        ("dino", "part zu del", "Partizip zu del", "given (participle of del)", "head", ()),
    ],
    ("316", "25"): [
        ("gelo", "prät zu džal", "Präteritum zu džal", "went (preterite of džal)",
         "head", ()),
    ],
    ("323", "08"): [
        ("-pe", "Verbalnomen", "Verbalnomen", "verbal-noun suffix", "head", ()),
    ],
    ("316", "06"): [
        ("dženo", "m", "(männliche) Person", "man; person", "head", ()),
        ("dženi", "f", "(weibliche) Person", "woman; person", "sibling", ()),
    ],
    ("316", "11"): [
        ("džung", "f", "Übel", "evil", "head", ()),
        ("džungalo", "adj", "schlecht", "bad", "sibling", ()),
    ],
    ("316", "27"): [
        ("gero", "adj", "arm", "poor", "head", ()),
        ("goro", "m", "Bauer", "peasant", "sibling", ("Rozw",)),
        ("goro", "m", "Verstorbener", "dead person", "sibling", ("Sinti",)),
    ],
    ("316", "29"): [
        ("geštani", "f", "Schwester des Mannes", "husband's sister", "head", ("Heinschink",)),
        ("džextani", "f", "Schwester des Mannes", "husband's sister", "variant", ("Thesleff",)),
    ],
    ("320", "04"): [
        ("khan(d)", "f", "Gestank", "stench", "head", ()),
        ("khandel", "verb", "stinken", "to stink", "sibling", ()),
    ],
    ("320", "27"): [
        ("ladžal", "itr", "sich schämen", "to be ashamed", "head", ()),
        ("ladžavel", "tr", "beschämen", "to shame", "sibling", ()),
    ],
    ("321", "25"): [
        ("manriklo", "m", "Kuchen", "cake", "head", ()),
        ("marikli", "f", "Kuchen", "cake", "variant", ()),
    ],
    ("321", "31"): [
        ("maškar", "präp", "zwischen; Mitte", "between; middle", "head", ()),
    ],
    ("322", "08"): [
        ("m(u)rtik", "", "ergreifen, packen (in del m(u)rtik)",
         "to seize; to grab (in del m(u)rtik)", "head", ()),
    ],
    ("322", "37"): [
        ("palal", "adv", "dahinter, hinten", "behind", "head", ()),
        ("pal(a)", "präp", "dahinter, hinten", "behind", "sibling", ()),
    ],
    ("324", "07"): [
        ("phabol", "itr", "brennen", "to burn", "head", ()),
        ("phabarel", "tr", "brennen", "to burn", "sibling", ()),
    ],
    ("323", "23"): [
        ("piro", "adj", "offen", "open", "head", ("Sa",)),
        ("pro", "adj", "offen", "open", "variant", ("Sinti",)),
    ],
    ("325", "27"): [
        ("sako", "pron", "jeder", "each", "head", ()),
        ("sakon", "pron", "jeder", "each", "variant", ()),
        ("hako", "pron", "jeder", "each", "variant", ("Sinti",)),
    ],
    ("325", "28"): [
        ("salo", "m", "Schwager", "brother-in-law", "head", ()),
        ("sali", "f", "Schwägerin", "sister-in-law", "sibling", ()),
    ],
    ("325", "41"): [
        ("sevi", "f", "Korb", "basket", "head", ()),
        ("sevli", "f", "Korb", "basket", "variant", ()),
        ("suvli", "f", "Korb", "basket", "variant", ("Sa",)),
    ],
    ("326", "02"): [
        ("sinto", "m", "Roma-Gruppe in Mitteleuropa", "Roma group of central Europe",
         "head", ()),
        ("sinte", "pl", "Roma-Gruppe in Mitteleuropa", "Roma group of central Europe",
         "variant", ()),
    ],
    ("326", "26"): [
        ("šišlo", "adj", "kräftig, robust", "strong; robust", "head", ("Sa",)),
        ("sislo", "adj", "kräftig, robust", "strong; robust", "variant", ()),
    ],
    ("327", "14"): [
        ("ter", "adv", "allmählich (in ter po ter)", "gradually (in ter po ter)",
         "head", ()),
    ],
    ("329", "32"): [
        ("xamavel", "", "gähnen", "to yawn", "head", ()),
        ("xamuvel", "", "gähnen", "to yawn", "variant", ("Sa",)),
        ("xamevel", "", "gähnen", "to yawn", "variant", ("So",)),
    ],
    ("330", "24"): [
        ("orde", "adv", "hier", "here", "head", ()),
        ("arde", "adv", "hier", "here", "variant", ("Rozw",)),
    ],
    ("331", "13"): [
        ("zen", "f", "Sattel", "saddle", "head", ()),
        ("zeja", "pl", "Rücken", "back", "sibling", ()),
    ],
    ("332", "02"): [
        ("džoro", "m", "Maultier", "mule", "head", ()),
        ("džori", "f", "Maultier", "mule", "variant", ()),
    ],
    ("332", "10"): [
        ("xor", "f", "Tiefe; tief", "depth; deep", "head", ()),
    ],
    ("332", "13"): [
        ("k(h)ilav", "f", "Pflaume", "plum", "head", ()),
        ("chjav-in", "f", "Pflaume", "plum", "variant", ()),
    ],
    ("333", "13"): [
        ("cevni", "f", "Schale, Hülse", "shell; husk", "head", ()),
        ("tsewni", "f", "Schale, Hülse", "shell; husk", "variant", ("Finck",)),
        ("čevni", "f", "Schale, Hülse", "shell; husk", "variant", ("Thes",)),
        ("čefni", "f", "Schale, Hülse", "shell; husk", "variant", ("Thes",)),
    ],
    ("333", "22"): [
        ("čiro(s)", "m", "Zeit", "time", "head", ()),
        ("ciros", "m", "Zeit", "time", "variant", ()),
        ("čirla", "adv", "längst, seit langem, vor langem", "long since; long ago",
         "sibling", ()),
    ],
    ("333", "27"): [
        ("dromin", "f", "Goldmünze", "gold coin", "head", ("Sinti",)),
        ("tromin", "f", "Goldmünze", "gold coin", "variant", ("Sa",)),
    ],
    ("334", "04"): [
        ("faj", "impers", "es scheint (mir)", "it seems (to me)", "head", ()),
        ("fanavel", "verb", "gefallen", "to please", "sibling", ("Thes",)),
    ],
    ("334", "07"): [
        ("filašni", "f", "Schloß, Gut", "castle; estate", "head", ("Thes",)),
        ("filišin", "f", "Schloß, Gut", "castle; estate", "variant", ("Sa",)),
        ("filecin", "f", "Schloß, Gut", "castle; estate", "variant", ("Sinti",)),
    ],
    ("334", "27"): [
        ("xoljavol", "itr", "böse werden", "to become angry", "head", ()),
        ("xoljarel", "tr", "ärgern", "to annoy", "sibling", ()),
    ],
    ("335", "07"): [
        ("kerjasi", "f", "Kirsche", "cherry", "head", ("So",)),
        ("kraš", "f", "Kirsche", "cherry", "variant", ("Sa",)),
        ("kraši", "f", "Kirsche", "cherry", "variant", ("Sa",)),
    ],
    ("335", "11"): [
        ("kloškerida", "f", "Rülpsen, Schluckauf", "belching; hiccup", "head", ("Finck",)),
        ("kločika", "f", "Rülpsen, Schluckauf", "belching; hiccup", "variant", ("Col",)),
        ("kockarida", "f", "Rülpsen, Schluckauf", "belching; hiccup", "variant", ()),
        ("lockarida", "f", "Rülpsen, Schluckauf", "belching; hiccup", "variant", ()),
    ],
    ("335", "25"): [
        ("korako", "m", "Krähe", "crow", "head", ("Sinti",)),
        ("korangos", "m", "Krähe", "crow", "variant", ("Urs",)),
    ],
    ("335", "34"): [
        ("kukli", "f", "Puppe", "doll", "head", ("So",)),
        ("kukla", "f", "Puppe", "doll", "variant", ("Thes",)),
    ],
    ("335", "43"): [
        ("lehusno", "adj", "Wöchnerin (in lehusno řomni)",
         "woman in childbed (in lehusno řomni)", "head", ()),
    ],
    ("336", "15"): [
        ("more", "Anrede für männl./weibl. Person", "he!", "hey! (to a man)", "head", ()),
        ("mori", "Anrede für männl./weibl. Person", "he!", "hey! (to a woman)",
         "variant", ()),
    ],
    ("336", "20"): [
        ("mura", "f", "Beere", "berry", "head", ("Thes",)),
        ("muros", "m", "Beere", "berry", "variant", ("Thes",)),
    ],
    ("336", "21"): [
        ("muskári", "m", "(männliches) Büffelkalb", "male buffalo calf", "head", ("Bug.",)),
        ("muskaris", "m", "(männliches) Büffelkalb", "male buffalo calf", "variant", ("Thes",)),
    ],
    ("336", "28"): [
        ("ora", "f", "Stunde", "hour", "head", ("Sa",)),
        ("kora", "f", "Stunde", "hour", "variant", ("So",)),
    ],
    ("336", "43"): [
        ("paraštuj", "f", "Freitag", "Friday", "head", ()),
        ("parasko", "f", "Freitag", "Friday", "variant", ("Sa",)),
    ],
    ("337", "14"): [
        ("podźa²", "pl", "Schuhe", "shoes", "head", ()),
        ("podźisarel", "verb", "Schuhe anziehen", "to put on shoes", "sibling", ()),
    ],
    ("337", "34"): [
        ("sali", "f", "Speichel", "saliva", "head", ()),
        ("salja", "pl", "Speichel", "saliva", "variant", ()),
    ],
    ("338", "09"): [
        ("šambona", "f", "Waldhorn", "hunting horn", "head", ("Sa",)),
        ("samuna", "f", "Pfeife", "pipe", "sibling", ("Thes",)),
    ],
    ("338", "16"): [
        ("tinanel", "tr", "schütteln; zittern", "to shake; to tremble", "head", ()),
        ("cinjazel", "tr", "schütteln; zittern", "to shake; to tremble", "variant", ()),
    ],
    ("338", "17"): [
        ("tirax", "m/f", "Schuh, Stiefel", "shoe; boot", "head", ()),
        ("čiox", "f", "Schuh, Stiefel", "shoe; boot", "variant", ("Sa",)),
    ],
}

# Printed headwords that carry a clitic or phrase in parentheses rather than a source
# label. The clitic is not part of the lexeme, so it is dropped from the display form
# and the printed shape stays in the audit's Raw_Headword.
INLINE_FORM_NOTES = {
    "faj (ma)": "faj",
    "xljel (pe(s))": "xljel",
    "hum (te)": "hum",
}

TRAILING_MARK = re.compile(r"\s*\(([^()]*)\)\s*$")
HOMONYM = re.compile(r"[¹²³]")
LEADING_GLOSS_MARK = re.compile(r"^\(([^()]*)\)\s*")

# Bracket wording that withdraws the etymology it cites. A record whose bracket carries
# any of these before the cited Old Indo-Aryan form is never linked.
REJECTIONS = (
    "kaum", "nicht", "unwahrscheinl", "scheidet aus", "falsch", "schwierig",
    "paßt", "ohne Etym", "fern", "wegen fehlender",
)
# Grammatical and philological labels that can follow "ai." instead of a headword.
ETYMON_LABELS = {
    "lok", "abl", "gen", "aor", "acc", "instr", "nom", "voc", "dat", "interrog",
    "adj", "pron", "ppp", "part", "kaus", "neutr", "pl", "sg", "pass", "itr", "tr",
    "dial", "pk", "pa", "ni", "mi", "hi", "rv", "ved", "panini", "mahābhārata",
}
ACCENTS = {"́", "̀"}


def normalise_etymon(text: str) -> str:
    """Fold a cited OIA form onto the CDIAL headword spelling for matching only.

    Vedic pitch accents, reconstruction asterisks, morpheme hyphens, homonym numbers
    and Turner's ``ē``/``ō`` are not contrastive between the two notations. Vowel
    length, retroflexion and sibilant quality are, and are preserved.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if c not in ACCENTS)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("*", "").replace("-", "").replace("ē", "e").replace("ō", "o")
    text = text.replace("ṛ", "r̥").replace("ḷ", "l̥")
    text = re.sub(r"[¹²³0-9()\[\]\"]", "", text)
    # A few CDIAL headwords keep the printed entry's trailing comma or period.
    text = text.strip().strip(",;:.").strip()
    return unicodedata.normalize("NFD", text.lower())


def load_cdial_index(path: Path = CDIAL_PARAMS) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return index
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 2 or not row[0].strip():
                continue
            index[normalise_etymon(row[1])].append(row[0])
    return index


def clause_start(etymology: str, position: int) -> int:
    return etymology.rfind(";", 0, position) + 1


def clause_of(etymology: str, position: int) -> str:
    """Return the semicolon-delimited clause of the bracket containing ``position``."""
    start = clause_start(etymology, position)
    end = etymology.find(";", position)
    return etymology[start: end if end != -1 else len(etymology)]


def cited_etyma(etymology: str) -> list[tuple[str, bool, int]]:
    """Return each Old Indo-Aryan form the bracket cites, with its clause's verdict.

    Boretzky & Igla routinely name an etymology in order to reject it, and just as
    routinely reject one clause before proposing another in the next. The rejection is
    therefore scoped to the semicolon-delimited clause the form stands in, so
    ``[ai. X ... paßt lautlich nicht]`` yields no link while
    ``[ai. Y ... gehören nicht dazu; eher < pa. Z < ai. W]`` still links W.
    """
    out = []
    # "ai. pk. tālu" and "ai. pa. danta-" state one form shared by the old and middle
    # stages; the OIA headword follows the second abbreviation, not the first.
    pattern = r"\bai\.\s+(?:(?:pk|pa|mi|ni|ved|apabhr)\.\s+)?((?:\*)?[^\s,;()\]\"]+)"
    for match in re.finditer(pattern, etymology):
        # A cited form can be followed immediately by the author's query mark or by
        # sentence punctuation; neither belongs to the headword being matched.
        token = match.group(1).rstrip("?!.,;:")
        if not token or token.lower() in ETYMON_LABELS:
            continue
        start = clause_start(etymology, match.start())
        clause = clause_of(etymology, match.start()).lower()
        rejected = any(marker.lower() in clause for marker in REJECTIONS)
        out.append((token, rejected, start))
    return out


def split_marks(text: str) -> tuple[str, list[str]]:
    """Peel trailing ``(Sa)``-style source and dialect labels off a printed form."""
    marks: list[str] = []
    known = set(SOURCE_MARKS) | set(DIALECT_MARKS)
    while True:
        match = TRAILING_MARK.search(text)
        if not match:
            break
        parts = [part.strip() for part in match.group(1).replace("/", ",").split(",")]
        if parts and all(part in known for part in parts):
            marks.extend(parts)
            text = (text[: match.start()] + text[match.end():]).strip()
        else:
            break
    return text.strip(), marks


def strip_gloss_marks(gloss: str) -> tuple[str, list[str]]:
    marks: list[str] = []
    known = set(SOURCE_MARKS) | set(DIALECT_MARKS)
    while True:
        match = LEADING_GLOSS_MARK.match(gloss)
        if not match:
            break
        parts = [part.strip() for part in match.group(1).split(",")]
        if parts and all(part in known for part in parts):
            marks.extend(parts)
            gloss = gloss[match.end():].strip()
        else:
            break
    return gloss.strip(), marks


def split_outside_parens(text: str) -> list[str]:
    """Split a printed headword field on commas that are not inside parentheses."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def clean_form(form: str) -> str:
    """Strip printed homonym superscripts and interjection exclamation marks."""
    return HOMONYM.sub("", form).rstrip("!").strip()


def split_units(record: dict) -> list[tuple[str, str, str, str, str, tuple[str, ...]]]:
    """Split one printed record into (form, grammar, gloss_de, gloss_en, relation, marks)."""
    key = (record["Printed_Page"], record["Item"])
    if key in CURATED:
        return list(CURATED[key])

    headword = record["Headword"]
    grammar = record["Grammar"]
    gloss_de, gloss_marks = strip_gloss_marks(record["Gloss_German"])
    gloss_en = record["Gloss_English"]

    units = []
    for piece in split_outside_parens(headword):
        piece = piece.strip()
        if not piece:
            continue
        form, marks = split_marks(piece)
        form = INLINE_FORM_NOTES.get(form, form)
        units.append((form, tuple(marks)))
    out = []
    for index, (form, marks) in enumerate(units):
        relation = "head" if index == 0 else "variant"
        all_marks = tuple(dict.fromkeys((*gloss_marks, *marks)))
        out.append((form, grammar, gloss_de, gloss_en, relation, all_marks))
    return out


def resolve_marks(marks) -> tuple[str, str, list[str]]:
    """Return (language_id, dialect_tag, citation keys) for a form's printed labels."""
    language_id, tag = BASE_LANGUAGE_ID, ""
    citations = []
    for mark in marks:
        if mark in DIALECT_MARKS:
            language_id, tag = DIALECT_MARKS[mark]
        elif mark in SOURCE_MARKS:
            citations.append(SOURCE_MARKS[mark])
    return language_id, tag, list(dict.fromkeys(citations))


def grammar_tags(grammar: str) -> tuple[str, bool]:
    grammar = grammar.strip()
    if not grammar:
        return "", True
    if grammar in GRAMMAR_TAGS:
        return GRAMMAR_TAGS[grammar], True
    return "", False


def read_extract(path: Path = EXTRACT) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing transcription layer {path}; it is the checked-in source of this import"
        )
    rows = []
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("|")
        for line in stream:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("|")
            if len(fields) != len(header):
                raise SystemExit(f"malformed extract row: {line!r}")
            rows.append(dict(zip(header, fields)))
    return rows


def build(records: list[dict]):
    cdial = load_cdial_index()
    forms: list[dict] = []
    audit: list[dict] = []
    unknown_grammar: Counter = Counter()
    by_identity: dict[tuple[str, str, str], dict] = {}

    for record in records:
        page, column, item = record["Printed_Page"], record["Column"], record["Item"]
        pdf_page = int(page) - PRINTED_PAGE_OFFSET
        base_audit = {
            "Snapshot_Date": SNAPSHOT_DATE,
            "Collation_Date": COLLATION_DATE,
            "Printed_Page": page,
            "PDF_Page": str(pdf_page),
            "Column": column,
            "Item": item,
            "List": LIST_NAMES[record["List"]],
            "Letter": record["Letter"],
            "Raw_Headword": record["Headword"],
            "Raw_Grammar": record["Grammar"],
            "Raw_Gloss_German": record["Gloss_German"],
            "Raw_Etymology": record["Etymology"],
            "Record_SHA256": hashlib.sha256(
                "|".join(record[name] for name in record).encode("utf-8")
            ).hexdigest(),
        }
        if record["Grammar"] == "xref":
            audit.append({**base_audit, "Status": "crossref", "Reason": record["Gloss_German"],
                          "Review": record["Editor_Note"]})
            continue

        etymology = record["Etymology"]
        mentions = cited_etyma(etymology) if record["List"] == "ind" else []
        accepted = [token for token, rejected, _ in mentions if not rejected]
        etymon = accepted[0] if accepted else (mentions[0][0] if mentions else "")
        # "ai. X/Y", and equally "< ai. X ... oder < ai. Y" inside one clause, offer two
        # etyma the source does not choose between; both are recorded and neither linked.
        alternatives = [part for part in etymon.split("/") if part] if etymon else []
        if accepted:
            first_clause = next(start for _, rejected, start in mentions if not rejected)
            in_first_clause = [
                token
                for token, rejected, start in mentions
                if not rejected and start == first_clause
            ]
            if len(in_first_clause) > 1:
                alternatives = in_first_clause
                etymon = " / ".join(in_first_clause)
        candidates = (
            [pid for part in alternatives for pid in cdial.get(normalise_etymon(part), [])]
            if accepted else []
        )
        if not mentions:
            parameter_id, match_status = "", "no-etymon"
        elif not accepted:
            parameter_id, match_status = "", "rejected-by-source"
        elif len(alternatives) > 1:
            parameter_id, match_status = "", "source-alternatives"
        elif len(candidates) == 1:
            parameter_id, match_status = candidates[0], "linked"
        elif len(candidates) > 1:
            parameter_id, match_status = "", "ambiguous"
        else:
            parameter_id, match_status = "", "unmatched"

        head_key = ""
        for unit_index, (form, grammar, gloss_de, gloss_en, relation, marks) in enumerate(
            split_units(record), start=1
        ):
            form = clean_form(form)
            language_id, dialect, citations = resolve_marks(marks)
            tags, known = grammar_tags(grammar)
            if not known:
                unknown_grammar[grammar] += 1
            tag_tokens = [token for token in tags.split() if token]
            if form.startswith("-"):
                tag_tokens.append("suffix")
            if form.endswith("-"):
                tag_tokens.append("prefix")
            if dialect:
                tag_tokens.append(dialect)
            if relation == "variant":
                tag_tokens.append("alternate")
            if "?" in etymology:
                tag_tokens.append("uncertain")
            tag_text = " ".join(dict.fromkeys(tag_tokens))

            suffix = "" if relation == "head" else f":{relation}:{unit_index}"
            entry_key = f"{SOURCE_ID}:{record['List']}:{page}:{column}:{item}{suffix}"
            if relation == "head":
                head_key = entry_key

            locator = f"{SOURCE_ID}[p. {page}, col. {1 if column == 'L' else 2}, entry {int(item)}]"
            source = ";".join([locator, *citations])
            row = {
                "Language_ID": language_id,
                "Parameter_ID": parameter_id if record["List"] == "ind" else "",
                "Form": form,
                "Gloss": gloss_en,
                "Native": "",
                "Phonemic": "",
                "Notes": "",
                "Source": source,
                "Cognateset": "",
                "Etymology": f"{LIST_NAMES[record['List']]}: [{etymology}]" if etymology else "",
                "Entry_Key": entry_key,
                "Variant_Of_Key": head_key if relation == "variant" else "",
                "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "",
                "Tags": tag_text,
            }
            unit_audit = {
                **base_audit,
                "Unit_Form": form,
                "Unit_Relation": relation,
                "Status": "ingested",
                "Reason": "",
                "Entry_Key": entry_key,
                "Language_ID": language_id,
                "Tags": tag_text,
                "Gloss_English": gloss_en,
                "Source_Marks": " ".join(marks),
                "Citations": ";".join(citations),
                "Etymon_Cited": etymon,
                "Etymon_Status": match_status,
                "Etymon_Candidates": " ".join(candidates),
                "Parameter_ID": row["Parameter_ID"],
                "Review": record["Editor_Note"],
            }

            identity = (language_id, form, gloss_en)
            existing = by_identity.get(identity)
            if existing is not None:
                merge_row(existing, row)
                if relation == "head":
                    head_key = existing["Entry_Key"]
                unit_audit["Status"] = "merged"
                unit_audit["Reason"] = f"merged into {existing['Entry_Key']}"
                audit.append(unit_audit)
                continue
            by_identity[identity] = row
            forms.append(row)
            audit.append(unit_audit)

    return forms, audit, unknown_grammar


def merge_row(target: dict, extra: dict) -> None:
    """Fold a cross-list repetition of the same lexeme into the row already emitted."""
    sources = [part for part in target["Source"].split(";") if part]
    for part in extra["Source"].split(";"):
        if part and part not in sources:
            sources.append(part)
    target["Source"] = ";".join(sources)
    if extra["Etymology"] and extra["Etymology"] not in target["Etymology"]:
        target["Etymology"] = " | ".join(
            part for part in [target["Etymology"], extra["Etymology"]] if part
        )
    if not target["Parameter_ID"] and extra["Parameter_ID"]:
        target["Parameter_ID"] = extra["Parameter_ID"]
    tags = dict.fromkeys([*target["Tags"].split(), *extra["Tags"].split()])
    target["Tags"] = " ".join(tags)


def write_csv(path: Path, fields: list[str], rows: list[dict], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerows(rows)


def sample_rows(audit: list[dict], size: int = 20) -> list[dict]:
    """Deterministic spread across the appendix for the raw-vs-output audit."""
    ingested = [row for row in audit if row["Status"] == "ingested"]
    if not ingested:
        return []
    step = max(1, len(ingested) // size)
    return [ingested[index * step] for index in range(min(size, len(ingested)))]


def verify_pdf(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        raise SystemExit(f"scan SHA-256 {digest} does not match the collated copy")
    print(f"scan verified: {path} sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="write the installed CSV")
    parser.add_argument("--pdf", type=Path, help="verify the original scan byte-for-byte")
    args = parser.parse_args()

    if args.pdf:
        verify_pdf(args.pdf)

    records = read_extract()
    forms, audit, unknown_grammar = build(records)
    if unknown_grammar:
        raise SystemExit(f"unmapped grammatical labels: {dict(unknown_grammar)}")

    statuses = Counter(row["Status"] for row in audit)
    linked = sum(1 for row in forms if row["Parameter_ID"])
    print(f"raw records: {len(records)}")
    print(f"audit rows:  {len(audit)} {dict(statuses)}")
    print(f"installed:   {len(forms)} rows, {linked} with a CDIAL Parameter_ID")

    if args.install:
        write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
        write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
        write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample_rows(audit), header=True)
        MANIFEST_OUTPUT.write_text(
            json.dumps(
                {
                    "source": SOURCE_ID,
                    "snapshot_date": SNAPSHOT_DATE,
                    "collation_date": COLLATION_DATE,
                    "pdf_sha256": PDF_SHA256,
                    "pdf_pages": PDF_PAGES,
                    "printed_pages": [FIRST_PRINTED_PAGE, LAST_PRINTED_PAGE],
                    "raw_records": len(records),
                    "installed_rows": len(forms),
                    "audit_statuses": dict(statuses),
                    "cdial_linked": linked,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {FORM_OUTPUT}")


if __name__ == "__main__":
    main()
