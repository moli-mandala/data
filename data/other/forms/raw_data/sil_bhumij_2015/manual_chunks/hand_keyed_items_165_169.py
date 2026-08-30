#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 165--169."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_165_169_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental marks, "
    "vowel quality, nasals, continuations, and the page break rechecked at 800 "
    "dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM166_PAGE66 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA", "MCH"}
BLANKS = {(168, "DUM")}

DATA = {
    165: ("who?", {
        "BAI": ("okoe", "1"), "CHA": ("okoe", "1"),
        "DIG": ("ɔkɑje", "1"), "DUM": ("okoe", "1"),
        "LAD": ("okoe", "1"), "MAD": ("okoe", "1"),
        "MOH": ("ɔkɑj", "1"), "MUN": ("ɔkoje", "1"),
        "POD": ("okojɑ", "1"), "UDA": ("okoe", "1"),
        "MCH": ("ɔkoj", "1"), "MDI": ("okoe", "1"),
        "MDH": ("okoe", "1"), "MJH": ("ɔkɑje", "1"),
        "HDI": ("ɔkɑj", "1"), "SDI": ("okoe", "1"),
        "SNA": ("ɔkɑj", "1"), "ORI": ("kie", "1"),
    }),
    166: ("what?", {
        "BAI": ("kɑnɑ", "2"), "CHA": ("tʃikɑnɑ", "1"),
        "DIG": ("kenɑ", "2"), "DUM": ("tʃiɑ", "1"),
        "LAD": ("kɑɲɑ", "2"), "MAD": ("tʃikɑnɑ", "1"),
        "MOH": ("tʃikɛnɑ", "1"), "MUN": ("tʃijɑ", "1"),
        "POD": ("tʃiem", "1"), "UDA": ("kɑnɑ", "2"),
        "MCH": ("tʃɑnɑʔɑ", "1"), "MDI": ("tʃinɑ", "1"),
        "MDH": ("kɑnɑ", "2"), "MJH": ("tʃinɑ", "1"),
        "HDI": ("tʃinɑ", "1"), "SDI": ("tʃeʔt", "1"),
        "SNA": ("tʃeɾ", "1"), "ORI": ("kɔnɔʔ", "2"),
    }),
    167: ("where?", {
        "BAI": ("okot̪e", "1"), "CHA": ("okɑ", "1"),
        "DIG": ("ɔkuɑɾe", "1"), "DUM": ("okosɑ", "1"),
        "LAD": ("okonɾe", "1"), "MAD": ("okot̪e", "1"),
        "MOH": ("ɔkot̪ɑɾe", "1"), "MUN": ("ɔkowɑ", "1"),
        "POD": ("okosɑ", "1"), "UDA": ("oksɑi", "1"),
        "MCH": ("okot̪ɑʔ", "1"), "MDI": ("okonɾeko", "1"),
        "MDH": ("oksɑi", "1"), "MJH": ("ɔkot̪e", "1"),
        "HDI": ("ɔkonpɑ", "1"), "SDI": ("okɑ", "1"),
        "SNA": ("ɔkɑɾe", "1"), "ORI": ("keuntɑɾe | kuɑde", "2 | 2"),
    }),
    168: ("when?", {
        "BAI": ("tʃimt̪em", "1"), "CHA": ("tʃumt̪ɑ", "1"),
        "DIG": ("tʃimt̪ɑŋ", "1"), "DUM": ("", ""),
        "LAD": ("tʃimt̪ʌŋ", "1"), "MAD": ("tʃimt̪e", "1"),
        "MOH": ("tʃimt̪ɑ", "1"), "MUN": ("tʃimt̪u", "1"),
        "POD": ("tʃimt̪e", "1"), "UDA": ("tʃimt̪ɑŋ", "1"),
        "MCH": ("tʃimt̪ɑŋ", "1"), "MDI": ("tʃimt̪ɑ", "1"),
        "MDH": ("tʃimt̪ɑŋ", "1"), "MJH": ("tʃimtɑn", "1"),
        "HDI": ("tʃuile", "2"), "SDI": ("tisɾe | khɑn", "3 | 4"),
        "SNA": ("t̪iso", "3"), "ORI": ("kebe", "5"),
    }),
    169: ("how many?", {
        "BAI": ("tʃinɑŋ", "1"), "CHA": ("tʃimɑʔ", "1"),
        "DIG": ("tʃiminɑŋ", "1"), "DUM": ("tʃimɑʔ", "1"),
        "LAD": ("tʃiminʌŋ", "1"), "MAD": ("tʃimu", "1"),
        "MOH": ("tʃimu", "1"), "MUN": ("tʃimu", "1"),
        "POD": ("tʃimin", "1"), "UDA": ("tʃimnɑŋ", "1"),
        "MCH": ("tʃiminɑŋ", "1"), "MDI": ("tʃimin", "1"),
        "MDH": ("tʃimnɑŋ", "1"), "MJH": ("tʃiminɑŋgi", "1"),
        "HDI": ("tʃiminɑŋ", "1"), "SDI": ("tinɑk", "2"),
        "SNA": ("t̪inɛŋ", "2"), "ORI": ("ket̪e", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 165 or (item == 166 and code in ITEM166_PAGE66):
        return "66", "61", "right"
    if item in {166, 167, 168}:
        return "67", "62", "left"
    return "67", "62", "right"


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
    ) == 49
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
