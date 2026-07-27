"""Turn the reviewed Schmidt OCR sheet into importable Drasi/Brokskat rows.

The OCR sheet is produced by :mod:`extract_schmidt_missing`.  This script only
does mechanical cleanup; the resulting CSV must still be compared with the
rendered table pages before it replaces the committed review copy.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW_TABLE = HERE / "shina.csv"

GLOSS_OVERRIDES = {
    4: "blood", 34: "brother", 35: "brother's wife", 40: "father's brother",
    46: "mother's brother", 53: "wife's brother", 80: "spinning wheel",
    84: "blue sky", 95: "afternoon", 99: "down(hill)",
    105: "sunny side of mountain", 106: "shady side of mountain",
    107: "seasonal migration", 108: "spring (season)", 111: "up(hill)",
    117: "forest", 119: "hill", 123: "spring (of water)", 127: "bear",
    128: "bird", 129: "bull", 131: "cat (m.)", 132: "cat (f.)",
    140: "goat (m.)", 141: "goat (f.)", 146: "louse (nit)",
    147: "mouse, rat", 153: "apple", 155: "bark (of tree)",
    164: "mulberry tree", 172: "to beat", 173: "to bite",
    174: "to burn (intr.)", 175: "to burn (tr.)", 176: "to come",
    177: "to cry", 178: "to die", 179: "to drink", 180: "to eat",
    181: "to fly", 182: "to give", 183: "to go", 184: "to harvest",
    185: "to hear", 186: "to kill", 187: "to know", 188: "to laugh",
    189: "to lie (down)", 190: "to say", 191: "to see", 192: "to sit",
    193: "to sleep", 194: "to stand", 195: "to swim", 196: "to walk",
    197: "to wash (tr.)", 198: "all (countable)",
    199: "all (uncountable)", 230: "right (direction)", 250: "he",
    251: "I", 254: "they (m., dist.)", 255: "they (f., dist.)",
    256: "they (m., prox.)", 257: "they (f., prox.)", 258: "that (m.)",
    259: "that (f.)", 260: "this (m.)", 261: "this (f.)", 265: "who?",
    266: "you (sg.)", 267: "you (pl.)", 268: "camel", 286: "Milky Way",
    291: "rice (paddy)", 292: "rice (cooked)", 297: "to write",
}

# Tesseract merges the weekday subrows because only item 97 is numbered in the
# left margin.  These readings were transcribed directly from p. 265.
TIME_ROWS = [
    ("dr", "dées", "day", "97"),
    ("bro", "dis", "day", "97"),
    ("bro", "'súri", "day", "97"),
    ("dr", "baṭavaár", "Saturday", "97a"),
    ("dr", "adít", "Sunday", "97b"),
    ("dr", "tsádraál", "Monday", "97c"),
    ("dr", "aṅgaáro", "Tuesday", "97d"),
    ("dr", "bóodo", "Wednesday", "97e"),
    ("dr", "brésput", "Thursday", "97f"),
    ("dr", "jumáa~", "Friday", "97g"),
    ("dr", "c̣iírye", "day after tomorrow", "98"),
    ("bro", "c̣e dis", "day after tomorrow", "98"),
]

# Cells where page layout made the generic row-band OCR merge columns/lines,
# plus a handful of specialist glyphs which English OCR cannot represent.
CELL_OVERRIDES = {
    (1, "dr"): "khiṅ, gikhití", (1, "bro"): "caṅ'khoṅ",
    (2, "dr"): "dáae", (7, "bro"): "kry, dut", (15, "dr"): "páa",
    (26, "bro"): "grii, ziṅ'gat",
    (7, "dr"): "mamé", (20, "dr"): "ẓuík", (29, "dr"): "kaṇḍi",
    (40, "dr"): "c̣uṇu babo", (63, "bro"): "bun, kheey",
    (68, "dr"): "mac̣híi", (68, "bro"): "giaá'tsi",
    (69, "dr"): "goóṭ", (70, "dr"): "yap, fiíl", (70, "bro"): "giab, iṣka",
    (71, "dr"): "coṣ", (71, "bro"): "thak'ṣaa",
    (72, "dr"): "thokteé, phyóoṛi, gintí", (72, "bro"): "thok'tse",
    (73, "dr"): "aṇṇáṭi, dut", (73, "bro"): "ɨ'dẓen",
    (78, "dr"): "ḍoṅo, yóo leéc̣i", (78, "bro"): "bo'ṣuṅs",
    (119, "bro"): "caris, nag'lis, zuy", (127, "dr"): "iš",
    (136, "dr"): "ṭhuúli", (138, "dr"): "chímo", (139, "dr"): "priíẓo",
    (144, "dr"): "ašup", (151, "dr"): "lamúṭi", (163, "dr"): "makái",
    (171, "dr"): "byéi", (177, "dr"): "roóno", (196, "dr"): "yazoóno",
    (209, "dr"): "c̣eék", (210, "dr"): "paziléi", (224, "dr"): "naáo",
    (225, "dr"): "náu", (234, "dr"): "ṭíino", (236, "dr"): "ṣóoi",
    (243, "dr"): "sáas", (251, "dr"): "moh", (252, "dr"): "nuṣ",
    (260, "dr"): "anúh, áa", (264, "dr"): "jok",
    (275, "dr"): "buxáar, tsat", (289, "dr"): "hattáa",
    (291, "dr"): "dayóo", (292, "dr"): "brím",
    (149, "dr"): "lac̣", (149, "bro"): "nílo", (177, "bro"): "rus",
    (207, "bro"): "as'toṣ", (219, "bro"): "šo",
    (233, "bro"): "sat'tóoṣ", (240, "dr"): "c̣óoi",
    (250, "dr"): "j̣o", (255, "dr"): "j̣o, paraáo",
    (270, "dr"): "a~i~c̣i",
}

ANIMAL_SUBROWS = [
    ("dr", "eš", "ewe", "149a"), ("bro", "eey", "ewe", "149a"),
    ("dr", "karéelo", "ram", "149b"), ("bro", "chur'di", "ram", "149b"),
]


def clean_gloss(item: int, value: str) -> str:
    if item in GLOSS_OVERRIDES:
        return GLOSS_OVERRIDES[item]
    value = re.sub(r"\b\d{1,3}[a-z]?[.,)]?\b", "", value)
    value = re.sub(r"\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|VHT|XT)\.?\b", "", value)
    value = " ".join(value.replace("—", " ").replace("_", " ").split()).strip(" .,;")
    # OCR occasionally reverses short wrapped labels.
    reversals = {
        "tree) bark (of": "bark (of tree)", "rat mouse": "mouse, rat",
        "drink to": "to drink", "give to": "to give", "harvest to": "to harvest",
        "know to": "to know", "sit to": "to sit", "swim to": "to swim",
        "write to": "to write", "(paddy) rice": "rice (paddy)",
        "(cooked) rice": "rice (cooked)", "pl. you": "you (pl.)",
    }
    return reversals.get(value, value)


def clean_cell(value: str) -> list[str]:
    value = value.replace("‘", "'").replace("’", "'").replace("—", " ")
    value = value.replace("_", " ").replace("©", " ").replace("§", "š")
    if re.search(r"\b(?:data\s+no|no\s+data)\b", value, re.I):
        return []
    value = re.sub(r"\bpl\.?\??\b", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" ,.;?")
    values = []
    for form in value.split(","):
        form = form.strip(" .;?")
        if not form or form in {"-", "–"}:
            continue
        form = form.replace("S", "š").replace("Z", "ẓ")
        form = form.replace("0", "o").replace("6", "o").replace("4", "a")
        form = form.replace("33", "ɨɨ").replace("3", "ɨ")
        form = re.sub(r"(?<=\w)1(?=\w)", "'", form)
        form = form.replace("1", "'").replace("”", "").strip()
        values.append(form)
    return values


def rough(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z]+", "", value.replace("š", "s").replace("ẓ", "z"))


def source_forms() -> dict[str, list[str]]:
    forms: dict[str, list[str]] = defaultdict(list)
    with RAW_TABLE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for language in ("gil", "koh", "gur", "Astor"):
                for form in row[language].split(","):
                    form = re.sub(r"\s*\([^)]*\)\s*$", "", form).strip()
                    if form:
                        forms[row["Gloss"]].append(form)
    return forms


def existing_keys() -> set[tuple[str, str, str]]:
    keys = set()
    with RAW_TABLE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for language in ("dr", "bro"):
                for form in row[language].split(","):
                    form = re.sub(r"\s*\([^)]*\)\s*$", "", form).strip()
                    if form:
                        keys.add((language, row["Gloss"], rough(form)))
    return keys


def restore_known_spelling(
    form: str, gloss: str, language: str, known: dict[str, list[str]]
) -> str:
    target = rough(form)
    if not target:
        return form
    choices = [(difflib.SequenceMatcher(None, target, rough(candidate)).ratio(), candidate)
               for candidate in known.get(gloss, [])]
    if not choices:
        return form
    score, candidate = max(choices)
    # Only copy a source spelling for near-identical cognates.  This restores
    # accents/underdots lost by OCR without erasing genuine dialect differences.
    threshold = 0.78 if language == "dr" and len(target) == len(rough(candidate)) else 0.94
    return candidate if score >= threshold else form


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    known = source_forms()
    already_present = existing_keys()
    output = []
    with args.review.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = int(row["Item"])
            if item in {97, 98}:
                continue
            gloss = clean_gloss(item, row["Gloss_OCR"])
            for language, field in (("dr", "Drasi_OCR"), ("bro", "Brokskat_OCR")):
                cell = CELL_OVERRIDES.get((item, language), row[field])
                for form in clean_cell(cell):
                    if (item, language) not in CELL_OVERRIDES:
                        form = restore_known_spelling(form, gloss, language, known)
                    if (language, gloss, rough(form)) in already_present:
                        continue
                    output.append({
                        "Language_ID": language,
                        "Form": form,
                        "Gloss": gloss,
                        "Notes": f"Table 2 item {item}",
                    })
    output.extend(
        {"Language_ID": language, "Form": form, "Gloss": gloss,
         "Notes": f"Table 2 item {item}"}
        for language, form, gloss, item in TIME_ROWS + ANIMAL_SUBROWS
        if (language, gloss, rough(form)) not in already_present
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Language_ID", "Form", "Gloss", "Notes"])
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
