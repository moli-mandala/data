#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 155--159."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_155_159_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with nasalization, "
    "dental marks, vowel quality, continuations, and the page break rechecked "
    "at 800 dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM156_PAGE64 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA", "MCH"}
BLANKS = {(158, "MJH")}

DATA = {
    155: ("five", {
        "BAI": ("monejɑ", "1"), "CHA": ("mõnojɑ", "1"),
        "DIG": ("mõnejɑ", "1"), "DUM": ("mõnẽɑ", "1"),
        "LAD": ("moniʌ", "1"), "MAD": ("moneɑ", "1"),
        "MOH": ("mõnəjɑ", "1"), "MUN": ("mõnəjɑ", "1"),
        "POD": ("monejɑ", "1"), "UDA": ("moneɑ", "1"),
        "MCH": ("monejɑ", "1"), "MDI": ("monɾeɑ", "1"),
        "MDH": ("pɑ̃tʃ", "2"), "MJH": ("mõnəjɑ", "1"),
        "HDI": ("mũje", "1"), "SDI": ("mõɾẽ", "1"),
        "SNA": ("mone", "1"), "ORI": ("pɑntʃə", "2"),
    }),
    156: ("six", {
        "BAI": ("tʃo", "2"), "CHA": ("t̪uɾijɑ", "1"),
        "DIG": ("t̪uɾijɑ", "1"), "DUM": ("t̪uɾie", "1"),
        "LAD": ("t̪uɾiʌ", "1"), "MAD": ("tʃe", "2"),
        "MOH": ("t̪uɾijɑ", "1"), "MUN": ("t̪uɾijɑ", "1"),
        "POD": ("t̪uɾie", "1"), "UDA": ("tʃhe", "2"),
        "MCH": ("t̪uɾijɑ", "1"), "MDI": ("t̪uɾiɑ | t̪uɾuiɑ", "1 | 1"),
        "MDH": ("tʃhe", "2"), "MJH": ("t̪uɾijɑ", "1"),
        "HDI": ("t̪uɾiɑ", "1"), "SDI": ("t̪uɾui", "1"),
        "SNA": ("t̪uɾuj", "1"), "ORI": ("tʃhə", "2"),
    }),
    157: ("seven", {
        "BAI": ("sɑt̪", "2"), "CHA": ("sɑt̪", "2"),
        "DIG": ("sɑt̪", "2"), "DUM": ("sʌt̪", "2"),
        "LAD": ("ɑjɛ", "1"), "MAD": ("sɑt̪", "2"),
        "MOH": ("sɑt̪", "2"), "MUN": ("sɑt̪", "2"),
        "POD": ("sɑt̪", "2"), "UDA": ("sɑt̪", "2"),
        "MCH": ("ejeː", "1"), "MDI": ("eɑ", "1"),
        "MDH": ("sɑt̪", "2"), "MJH": ("sɑt̪", "2"),
        "HDI": ("ɑje", "1"), "SDI": ("eɑe", "1"),
        "SNA": ("sɑt̪", "2"), "ORI": ("sɑt̪o", "2"),
    }),
    158: ("eight", {
        "BAI": ("ɑt", "2"), "CHA": ("ɑt", "2"),
        "DIG": ("ɑto", "2"), "DUM": ("ɑt", "2"),
        "LAD": ("ilijʌ", "1"), "MAD": ("ɑt", "2"),
        "MOH": ("ɑtɑ", "2"), "MUN": ("ɑtɑ", "2"),
        "POD": ("ɑt", "2"), "UDA": ("ɑt", "2"),
        "MCH": ("iɾɑlije", "1"), "MDI": ("iɾɑliɑ | iɾiliɑ", "1 | 1"),
        "MDH": ("ɑt", "2"), "MJH": ("", ""),
        "HDI": ("iɾlijə", "1"), "SDI": ("iɾɑɭ", "1"),
        "SNA": ("ɑto", "2"), "ORI": ("ɑtho", "2"),
    }),
    159: ("nine", {
        "BAI": ("no", "2"), "CHA": ("no", "2"),
        "DIG": ("nõ", "2"), "DUM": ("nʌ", "2"),
        "LAD": ("ɑɾijʌ", "1"), "MAD": ("no", "2"),
        "MOH": ("no", "2"), "MUN": ("notɑ", "2"),
        "POD": ("no", "2"), "UDA": ("nɛ", "2"),
        "MCH": ("ɑɾeje", "1"), "MDI": ("ɑɾeɑ", "1"),
        "MDH": ("nɛ", "2"), "MJH": ("ɑɾeje", "1"),
        "HDI": ("eɾijɑ", "1"), "SDI": ("ɑɾe", "1"),
        "SNA": ("nõ", "2"), "ORI": ("nɑo", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 155 or (item == 156 and code in ITEM156_PAGE64):
        return "64", "59", "right"
    if item in {156, 157, 158}:
        return "65", "60", "left"
    return "65", "60", "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            source_blank = (item, code) in BLANKS
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if source_blank else "attested",
                "Confidence": "high",
                "Uncertainty": "source explicitly prints '0 no entry'" if source_blank else "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 5 * 18 == 90
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 91
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows
        if row["Review_Status"] == "attested" and row["Target"] == "yes"
    ) == 50
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
