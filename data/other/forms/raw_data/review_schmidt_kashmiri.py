"""Turn Table 3 OCR cells into the page-review worksheet.

Run this only when redoing the visual review.  ``schmidt_kashmiri.csv`` is the
checked artifact consumed by the database build.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


GLOSS_OVERRIDES = {
    "34": "brother", "35": "brother's wife", "39": "father's brother",
    "44": "mother's brother", "49": "wife's brother", "55": "arrow",
    "70": "irrigation channel", "84": "sky, blue", "95": "afternoon",
    "98": "day after tomorrow", "105": "sunny side of mountain",
    "106": "shady side of mountain", "107": "seasonal migration",
    "108": "spring (season)", "111": "uphill", "118": "highest summer pasture",
    "119": "hill", "123": "spring (of water)", "125": "ant",
    "131": "cat m.", "132": "cat f.", "140": "goat m.", "141": "goat f.",
    "146": "louse (nit)", "147": "mouse, rat", "149a": "ewe", "149b": "ram",
    "153": "apple", "155": "bark (of tree)", "164": "mulberry tree",
    "172": "to beat", "173": "to bite", "174": "to burn i.",
    "175": "to burn t.", "176": "to come", "177": "to cry", "178": "to die",
    "179": "to drink", "180": "to eat", "181": "to fly", "182": "to give",
    "183": "to go", "184": "to harvest", "185": "to hear", "186": "to kill",
    "187": "to know", "188": "to laugh", "189": "to lie (down)",
    "190": "to say", "191": "to see", "192": "to sit", "193": "to sleep",
    "194": "to stand", "195": "to swim", "196": "to walk", "197": "to wash t.",
    "198": "all (sārā)", "250": "he", "251": "I", "254": "they m. far",
    "255": "they f. far", "256": "they m. near", "257": "they f. near",
    "258": "that (thing)", "259": "that (person)", "260": "this (thing)",
    "261": "this (person)", "266": "you sg.", "267": "you pl.",
}

SECTION_TEXT = re.compile(
    r"(?:II\. Terms for kin|III\. Human artifacts|IV\. The sky, weather|"
    r"V\. Time and space|VII\. Animals|VIII\. The plant word|X\. Adjectives|"
    r"XI\. Pronouns etc\.)",
    re.IGNORECASE,
)


def clean_gloss(key: str, value: str) -> str:
    if key in GLOSS_OVERRIDES:
        return GLOSS_OVERRIDES[key]
    value = SECTION_TEXT.sub("", value)
    value = re.sub(r"\b\d{1,3}[a-z]?[.,I]?\b", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .,")
    return value


def clean_form(value: str) -> str:
    # Corrections common to the OCR font confusion, checked against the pages.
    value = value.replace("”", "ʰ").replace("®", "ʰ")
    value = value.replace("™", "ʲ").replace("£", "f.")
    value = re.sub(r"(?<=[ptkbdgčšts])\?(?=\b|[ei])", "ʰ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,|")
    return unicodedata.normalize("NFC", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "Item": row["Key"],
                    "Gloss": clean_gloss(row["Key"], row["Gloss_OCR"]),
                    "Kashmiri": clean_form(row["Kashmiri_OCR"]),
                    "Kishtawari": clean_form(row["Kishtawari_OCR"]),
                    "Poguli": clean_form(row["Poguli_OCR"]),
                    "Siraji": clean_form(row["Siraji_OCR"]),
                    "Page": row["Page"],
                }
            )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
