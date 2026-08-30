#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 125--129."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_125_129_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental and "
    "retroflex marks, vowel quality, nasalization, length, continuations, and "
    "the page break rechecked at 800 dpi; OCR/PDF text neither supplied nor "
    "verified any reading"
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
ITEM126_PAGE58 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN"}
BLANKS = {(125, "MDI")}

DATA = {
    125: ("week", {
        "BAI": ("hʌpt̪ɑ", "1"), "CHA": ("hʌpt̪ɑ", "1"),
        "DIG": ("ɛt̪əuɑɾi", "3"), "DUM": ("hɑt", "2"),
        "LAD": ("hɑpt̪ʌ", "1"), "MAD": ("mot hept̪ɑ", "1"),
        "MOH": ("hept̪ɑ", "1"), "MUN": ("moɾhɑt", "2"),
        "POD": ("hɑpt̪ɑ | hɑt", "1 | 2"),
        "UDA": ("sɛpt̪ɑ | hɑt", "1 | 2"),
        "MCH": ("ɛt̪ɑuɾi", "3"), "MDI": ("", ""),
        "MDH": ("sɛpt̪ɑ | hɑt", "1 | 2"), "MJH": ("hept̪ɑ", "1"),
        "HDI": ("sɛpt̪ɑ", "1"), "SDI": ("hɑpt̪ɑ", "1"),
        "SNA": ("sɛpt̪ɑ", "1"), "ORI": ("səpt̪ɑhə", "1"),
    }),
    126: ("month", {
        "BAI": ("tʃɑndu", "1"), "CHA": ("tʃʌndup", "1"),
        "DIG": ("moʔtʃɑt̪u", "1"), "DUM": ("tʃʌndu", "1"),
        "LAD": ("tʃɑnd̪uʔ", "1"), "MAD": ("mot tʃɛndu", "1"),
        "MOH": ("moʔtʃɑt̪u", "1"), "MUN": ("mɔʔtʃɑnt̪u", "1"),
        "POD": ("tʃɑndu", "1"), "UDA": ("tʃɑndu", "1"),
        "MCH": ("mid̪tʃɑnɖuʔu", "1"), "MDI": ("tʃɑnɖu", "1"),
        "MDH": ("tʃɑndu", "1"), "MJH": ("mebinɑ", "3"),
        "HDI": ("mitʃɑntu", "1"), "SDI": ("tʃɑndo", "1"),
        "SNA": ("tʃɑnʈo", "1"), "ORI": ("mɑsoʔ", "4"),
    }),
    127: ("year", {
        "BAI": ("siɾmɑ", "1"), "CHA": ("siɾmɑ", "1"),
        "DIG": ("mɔsiɾmɑ", "1"), "DUM": ("siɾmɑ", "1"),
        "LAD": ("siɾmɑ", "1"), "MAD": ("mot siɾmɑ", "1"),
        "MOH": ("mɔsiɾmɑ", "1"), "MUN": ("mɔsiɾmɑ", "1"),
        "POD": ("siɾumʌ", "1"), "UDA": ("siɾmɑ", "1"),
        "MCH": ("siɾmɑ", "1"), "MDI": ("siɾmɑ", "1"),
        "MDH": ("siɾmɑ", "1"), "MJH": ("bɛɾɑs", "2"),
        "HDI": ("bɔɾso", "2"), "SDI": ("seɾmɑ | botʃhoɾ", "1 | 3"),
        "SNA": ("siɾmɑ", "1"), "ORI": ("bɔɾsə", "2"),
    }),
    128: ("old", {
        "BAI": ("puɾnɑ", "1"), "CHA": ("puɾnɑ", "1"),
        "DIG": ("puɾuɳɑ", "1"), "DUM": ("puɾnɑ", "1"),
        "LAD": ("puɾnːɑ", "1"), "MAD": ("puɾnɑhɑ", "1"),
        "MOH": ("puɾuɳɑ", "1"), "MUN": ("puɾuɳɑ", "1"),
        "POD": ("puɾne", "1"), "UDA": ("puɾnɑ", "1"),
        "MCH": ("puɾɳɑ", "1"), "MDI": ("puɾɑnɑ", "1"),
        "MDH": ("puɾnɑ", "1"), "MJH": ("mɑɾi", "2"),
        "HDI": ("pɑpɑɾi", "3"), "SDI": ("mɑɾe", "2"),
        "SNA": ("mɑɾe", "2"), "ORI": ("poɾuɳɑ", "1"),
    }),
    129: ("new", {
        "BAI": ("nɑmɑ", "1"), "CHA": ("nɑmɑ", "1"),
        "DIG": ("nɑ̃uɑ̃", "1"), "DUM": ("nɑmɑ", "1"),
        "LAD": ("nɑuwʌ", "1"), "MAD": ("nɑmɑhɑ", "1"),
        "MOH": ("nemɑ", "1"), "MUN": ("nemɑ", "1"),
        "POD": ("nʌmɑ", "1"), "UDA": ("nɑwɑ", "1"),
        "MCH": ("nɑuɑ̃", "1"), "MDI": ("nɑwɑ", "1"),
        "MDH": ("nɑmɑ | nɑwɑ", "1 | 1"), "MJH": ("neuɑ̃", "1"),
        "HDI": ("nemɑ", "1"), "SDI": ("nɑwɑ", "1"),
        "SNA": ("neuɑ", "1"), "ORI": ("nuɑ̃", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 125 or (item == 126 and code in ITEM126_PAGE58):
        return "58", "53", "right"
    if item in {126, 127, 128}:
        return "59", "54", "left"
    return "59", "54", "right"


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
    ) == 94
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
