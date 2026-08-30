#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 204--208."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_204_208_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 800-dpi rendered PDF page with vowel "
    "qualities, glottal stops, nasals, source length colons, column continuation, "
    "and separately numbered responses checked in tight source-image crops; "
    "OCR/PDF text neither supplied nor verified any reading"
)
SITES = {
    "BAI": ("Bhumij", "Baigodia", True), "CHA": ("Bhumij", "Champi", True),
    "DIG": ("Bhumij", "Dighinuasahi", True), "DUM": ("Bhumij", "Dumadie", True),
    "LAD": ("Bhumij", "Ladhiramsai", True), "MAD": ("Bhumij", "Madhupur", True),
    "MOH": ("Bhumij", "Mohuldiha", True), "MUN": ("Bhumij", "Munduy", True),
    "POD": ("Bhumij", "Podadiha", True), "UDA": ("Bhumij/Mundari", "Udala", True),
    "MCH": ("Mundari", "Chalagi", False), "MDI": ("Mundari", "Dictionary", False),
    "MDH": ("Mundari", "Dhungarisai", False), "MJH": ("Mundari", "Jharmunda", False),
    "HDI": ("Ho", "Dillisore", False), "SDI": ("Santali", "Dictionary", False),
    "SNA": ("Santali", "Nayarangamotia", False), "ORI": ("Oriya", "Cuttack", False),
}
DATA = {
    204: ("you (2nd sg, formal)", {
        "BAI": ("ɑben", "1"), "CHA": ("ɑben", "1"), "DIG": ("", ""),
        "DUM": ("ɑben", "1"), "LAD": ("ɑbɛn", "1"), "MAD": ("ɑben", "1"),
        "MOH": ("ɑben", "1"), "MUN": ("ɑben", "1"), "POD": ("ʌben", "1"),
        "UDA": ("ɑbin", "1"), "MCH": ("ɑm", "2"), "MDI": ("", ""),
        "MDH": ("ɑbin", "1"), "MJH": ("ɑben", "1"), "HDI": ("ɑben", "1"),
        "SDI": ("ɑben", "1"), "SNA": ("ɑbiŋ", "1"), "ORI": ("ɑponõ", "3"),
    }),
    205: ("he (3rd sg, masculine)", {
        "BAI": ("ɑʔe", "2"), "CHA": ("ɑʔt̪", "2"), "DIG": ("", ""),
        "DUM": ("iniʔi", "1"), "LAD": ("iniʔ", "1"), "MAD": ("ini", "1"),
        "MOH": ("ini", "1"), "MUN": ("ini", "1"), "POD": ("ini", "1"),
        "UDA": ("ɑe", "2"), "MCH": ("ini", "1"), "MDI": ("ɑe", "2"),
        "MDH": ("ɑe", "2"), "MJH": ("ini", "1"), "HDI": ("ini", "1"),
        "SDI": ("uni", "1"), "SNA": ("uni", "1"), "ORI": ("se", "3"),
    }),
    206: ("she (3rd sg, feminine)", {
        "BAI": ("ɑʔe", "2"), "CHA": ("ɑtʔ", "2"), "DIG": ("", ""),
        "DUM": ("iniʔi", "1"), "LAD": ("iniʔ", "1"), "MAD": ("ini", "1"),
        "MOH": ("ini", "1"), "MUN": ("ini", "1"), "POD": ("ini", "1"),
        "UDA": ("ɑeʔ", "2"), "MCH": ("iniʔi", "1"), "MDI": ("", ""),
        "MDH": ("ɑeʔ", "2"), "MJH": ("ini", "1"), "HDI": ("ini", "1"),
        "SDI": ("uni", "1"), "SNA": ("uni", "1"), "ORI": ("se", "3"),
    }),
    207: ("we (1st pl, inclusive)", {
        "BAI": ("ɑbu", "1"), "CHA": ("ɑbu", "1"), "DIG": ("", ""),
        "DUM": ("ɑbu", "1"), "LAD": ("ɑbu", "1"), "MAD": ("ɑbu", "1"),
        "MOH": ("ɛle", "2"), "MUN": ("ɛbu", "1"), "POD": ("ʌbu", "1"),
        "UDA": ("ɑbu", "1"), "MCH": ("ɑle", "2"), "MDI": ("ɑbu", "1"),
        "MDH": ("ɑbu", "1"), "MJH": ("ɛbu", "1"), "HDI": ("ɛbu", "1"),
        "SDI": ("ɑbo", "1"), "SNA": ("ɛle", "2"), "ORI": ("ɑme | ɑmpe", "3 | 3"),
    }),
    208: ("we (1st pl, exclusive)", {
        "BAI": ("ɑle", "1"), "CHA": ("ɑle", "1"), "DIG": ("", ""),
        "DUM": ("ɑle", "1"), "LAD": ("ɑlːe", "1"), "MAD": ("ɑle", "1"),
        "MOH": ("ɛle", "1"), "MUN": ("ɛpe", "2"), "POD": ("ʌle", "1"),
        "UDA": ("ɑle", "1"), "MCH": ("ɑle", "1"), "MDI": ("ɑle", "1"),
        "MDH": ("ɑle", "1"), "MJH": ("ɛle", "1"), "HDI": ("ɛle", "1"),
        "SDI": ("ɑle", "1"), "SNA": ("ɛlege", "1"), "ORI": ("ɑme | ɑmpe", "2 | 2"),
    }),
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def column_for(item, index):
    if item in {204, 205}:
        return "left"
    if item == 206:
        return "left" if index < 11 else "right"
    return "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for index, (code, (language, site, target)) in enumerate(SITES.items()):
            form, labels = cells[code]
            source_blank = not form
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": "75",
                "Printed_Page": "70", "Column": column_for(item, index),
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if source_blank else "attested",
                "Confidence": "high",
                "Uncertainty": "source explicitly prints '0 no entry'" if source_blank else "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-29",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 5 * 18 == 90
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 7
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested") == 85
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 45
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
