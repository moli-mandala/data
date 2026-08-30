#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 150--154."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_150_154_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with vowel quality, "
    "nasalization, dental marks, continuations, and the page break rechecked at "
    "800 dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM151_PAGE63 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}

DATA = {
    150: ("red", {
        "BAI": ("ɾɑŋgɑ", "2"), "CHA": ("ɾɑŋgɑ", "2"),
        "DIG": ("ɾenkɑ", "2"), "DUM": ("ɑɾɑʔɑ", "1"),
        "LAD": ("ɑɾɑ", "1"), "MAD": ("ɑɾɑʔ", "1"),
        "MOH": ("eɾɑ", "1"), "MUN": ("ɾenkɑ", "2"),
        "POD": ("ʌɾɑʔɑ", "1"), "UDA": ("ɾɑŋgɑ", "2"),
        "MCH": ("ɑɾɑʔɑ", "1"), "MDI": ("ɑɾɑ", "1"),
        "MDH": ("ɾɑŋgɑ", "2"), "MJH": ("eɾɑɑʔ", "1"),
        "HDI": ("eɾɑ", "1"), "SDI": ("ɑɾɑk", "1"),
        "SNA": ("eɾɑ", "1"), "ORI": ("ɾoŋgo | nɑli", "2 | 3"),
    }),
    151: ("one", {
        "BAI": ("mien", "1"), "CHA": ("mõe", "1"),
        "DIG": ("mɔjõ", "1"), "DUM": ("mudʒeʔd", "2"),
        "LAD": ("mijʌnd̪ʔ", "1"), "MAD": ("mudʒet", "2"),
        "MOH": ("mudʒit̪", "2"), "MUN": ("moʔj", "1"),
        "POD": ("mudʒed", "2"), "UDA": ("mojet", "1"),
        "MCH": ("mojɔn", "1"), "MDI": ("miɑd̪", "1"),
        "MDH": ("ek", "4"), "MJH": ("mijɑn", "1"),
        "HDI": ("mijɛn", "1"), "SDI": ("mit", "3"),
        "SNA": ("mit̪ɑŋ", "3"), "ORI": ("eko", "4"),
    }),
    152: ("two", {
        "BAI": ("bɑɾie", "1"), "CHA": ("bɑɾie", "1"),
        "DIG": ("beɾijɑ", "1"), "DUM": ("bɑɾie", "1"),
        "LAD": ("bɑɾiʌ", "1"), "MAD": ("bɑɾijɑ", "1"),
        "MOH": ("bɑɾijə", "1"), "MUN": ("beɾijɑ", "1"),
        "POD": ("bɑɾie", "1"), "UDA": ("bɑɾiɑ", "1"),
        "MCH": ("bɑɾije", "1"), "MDI": ("bɑɾ | bɑɾiɑ", "1 | 1"),
        "MDH": ("duɾi", "2"), "MJH": ("beɾijɑ", "1"),
        "HDI": ("bɑɾijə", "1"), "SDI": ("bɑɾ", "1"),
        "SNA": ("beɾijɑ", "1"), "ORI": ("duɾi", "2"),
    }),
    153: ("three", {
        "BAI": ("ɑpie", "1"), "CHA": ("ɑpie", "1"),
        "DIG": ("epijɑ", "1"), "DUM": ("ʌpie", "1"),
        "LAD": ("ʌpiʌ", "1"), "MAD": ("epije", "1"),
        "MOH": ("əpijə", "1"), "MUN": ("epijɑ", "1"),
        "POD": ("ʌpie", "1"), "UDA": ("ɑpijɑ", "1"),
        "MCH": ("ɑpije", "1"), "MDI": ("ɑpie", "1"),
        "MDH": ("t̪in", "2"), "MJH": ("epijɑ", "1"),
        "HDI": ("epijə", "1"), "SDI": ("ped", "1"),
        "SNA": ("pijɑ", "1"), "ORI": ("t̪ini", "2"),
    }),
    154: ("four", {
        "BAI": ("upunie", "1"), "CHA": ("upunijɑ", "1"),
        "DIG": ("upənijɑ", "1"), "DUM": ("upunie", "1"),
        "LAD": ("upuniɛ", "1"), "MAD": ("upunije", "1"),
        "MOH": ("upənijə", "1"), "MUN": ("upunijə", "1"),
        "POD": ("upunie", "1"), "UDA": ("upuniɑ", "1"),
        "MCH": ("opuɲie", "1"), "MDI": ("upun", "1"),
        "MDH": ("tʃɑɾ", "3"), "MJH": ("upənijɑ", "1"),
        "HDI": ("upunijə", "1"), "SDI": ("pon", "2"),
        "SNA": ("ponijɑ", "2"), "ORI": ("tʃɑɾi", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 150 or (item == 151 and code in ITEM151_PAGE63):
        return "63", "58", "right"
    if item in {151, 152, 153}:
        return "64", "59", "left"
    return "64", "59", "right"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 92
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Target"] == "yes"
    ) == 50
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
