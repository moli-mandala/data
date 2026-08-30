#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 115--119."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_115_119_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "nasalization, aspiration, vowel length, continuations, and page/column "
    "breaks rechecked at 800 dpi; text scaffold neither supplied nor verified "
    "any reading"
)
SITES = {
    "BAI": ("Bhumij", "Baigodia", True),
    "CHA": ("Bhumij", "Champi", True),
    "DIG": ("Bhumij", "Dighinuasahi", True),
    "DUM": ("Bhumij", "Dumadie", True),
    "LAD": ("Bhumij", "Ladhiramsai", True),
    "MAD": ("Bhumij", "Madhupur", True),
    "MOH": ("Bhumij", "Mohuldiha", True),
    "MUN": ("Bhumij", "Munduy", True),
    "POD": ("Bhumij", "Podadiha", True),
    "UDA": ("Bhumij/Mundari", "Udala", True),
    "MCH": ("Mundari", "Chalagi", False),
    "MDI": ("Mundari", "Dictionary", False),
    "MDH": ("Mundari", "Dhungarisai", False),
    "MJH": ("Mundari", "Jharmunda", False),
    "HDI": ("Ho", "Dillisore", False),
    "SDI": ("Santali", "Dictionary", False),
    "SNA": ("Santali", "Nayarangamotia", False),
    "ORI": ("Oriya", "Cuttack", False),
}
ITEM119_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD"}

DATA = {
    115: ("boy", {
        "BAI": ("hon koɖɑ", "1"), "CHA": ("koɖɑ hone", "1"),
        "DIG": ("huɖiɲhõn", "1"), "DUM": ("kodɑ hone", "1"),
        "LAD": ("koɾɑ hɔn", "1"), "MAD": ("kodɑ hon", "1"),
        "MOH": ("huɖiɲkoɖɑ", "1"), "MUN": ("koɖɑhõn", "1"),
        "POD": ("kuɖɑ hon", "1"), "UDA": ("kodɑ hon", "1"),
        "MCH": ("dɦɑŋgeɾɑ", "2"), "MDI": ("hon", "1"),
        "MDH": ("kodɑ hon", "1"), "MJH": ("koɖɑhõn", "1"),
        "HDI": ("kɔwɑhõn", "1"), "SDI": ("koɾɑ", "1"),
        "SNA": ("giɖɾə", "3"), "ORI": ("pilɑʔ", "4"),
    }),
    116: ("girl", {
        "BAI": ("kui hon", "1"), "CHA": ("kuɖi hone", "1"),
        "DIG": ("huɖiɲkuɖijon", "4"), "DUM": ("kudi hone", "1"),
        "LAD": ("kuɾi hon", "1"), "MAD": ("kudi hon", "1"),
        "MOH": ("kuɖihɔne", "1"), "MUN": ("kuɖihõn", "1"),
        "POD": ("kuɖi hõn", "1"), "UDA": ("kudi hon", "1"),
        "MCH": ("dɦɑŋgiɾi", "2"), "MDI": ("kuɾi hon", "1"),
        "MDH": ("kudi hon", "1"), "MJH": ("kuɖihõn", "1"),
        "HDI": ("kuihõn", "1"), "SDI": ("kuɾi", "1"),
        "SNA": ("kuɖigiɖɾə", "3"), "ORI": ("dʒio pilɑʔ", "5"),
    }),
    117: ("day", {
        "BAI": ("siŋgi", "1"), "CHA": ("siŋgi", "1"),
        "DIG": ("siŋki", "1"), "DUM": ("siŋgi", "1"),
        "LAD": ("siŋgi", "1"), "MAD": ("siŋgi", "1"),
        "MOH": ("ɖin", "2"), "MUN": ("siŋki", "1"),
        "POD": ("siŋgi", "1"), "UDA": ("siŋgi", "1"),
        "MCH": ("siŋgi", "1"), "MDI": ("ɖin | hulɑŋ", "2 | 3"),
        "MDH": ("siŋgi", "1"), "MJH": ("siŋki", "1"),
        "HDI": ("siŋki", "1"), "SDI": ("hilok", "4"),
        "SNA": ("siŋ", "1"), "ORI": ("ɖino", "2"),
    }),
    118: ("night", {
        "BAI": ("niɖe", "1"), "CHA": ("niɖɑ", "1"),
        "DIG": ("ejub", "3"), "DUM": ("nĩɖe", "1"),
        "LAD": ("niɖʌ", "1"), "MAD": ("niɖe", "1"),
        "MOH": ("ejub", "3"), "MUN": ("niɖɑ", "1"),
        "POD": ("niɖe", "1"), "UDA": ("niɖɑ", "1"),
        "MCH": ("niɖə", "1"), "MDI": ("niɖɑ", "1"),
        "MDH": ("niɖɑ", "1"), "MJH": ("ejub", "3"),
        "HDI": ("niɖe", "1"), "SDI": ("nindɑ | nindɑ", "1 | 2"),
        "SNA": ("ɲinʈiɾ", "2"), "ORI": ("ɾɑt̪i", "4"),
    }),
    119: ("morning", {
        "BAI": ("seʈɑʔɑ", "1"), "CHA": ("siʈɑʔ", "1"),
        "DIG": ("sɛʈɑ", "1"), "DUM": ("siʈɑʔɑ", "1"),
        "LAD": ("siʈːɑ", "1"), "MAD": ("seʈɑʔ", "1"),
        "MOH": ("sɛʈɑ", "1"), "MUN": ("sɛʈɑ", "1"),
        "POD": ("siʈɑʔɑ", "1"), "UDA": ("seʈɑʔ", "1"),
        "MCH": ("sɑʈɑʔɑ", "1"), "MDI": ("setɑ | idɑŋ", "1 | 2"),
        "MDH": ("seʈɑʔ", "1"), "MJH": ("siʈɑɑʔ", "1"),
        "HDI": ("siʈɑɑʔ", "1"), "SDI": ("setɑk", "1"),
        "SNA": ("sɛʈɑ", "1"), "ORI": ("səkɑɭə", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {115, 116}:
        return "56", "51", "right"
    if item in {117, 118} or (item == 119 and code in ITEM119_LEFT):
        return "57", "52", "left"
    return "57", "52", "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": "attested", "Confidence": "high",
                "Uncertainty": "", "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 5 * 18 == 90
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 93
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
