#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 65--69."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_065_069_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with page-break "
    "and retroflex-lateral cells enlarged separately; text scaffold not "
    "accepted without cell visual match"
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
ITEM68_LEFT = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH", "MJH",
}
BLANKS = {(68, "MDI"), (68, "SDI")}

DATA = {
    65: ("flower", {
        "BAI": ("boʔɑ", "1"), "CHA": ("bɑhɑ", "1"),
        "DIG": ("bɑʔ", "1"), "DUM": ("bʌhɑ", "1"),
        "LAD": ("bɑː", "1"), "MAD": ("bɑhɑ", "1"),
        "MOH": ("behɑ", "1"), "MUN": ("bɑ", "1"),
        "POD": ("bɑhɑ", "1"), "UDA": ("bɑhɑ", "1"),
        "MCH": ("behɑ", "1"), "MDI": ("bɑhɑ", "1"),
        "MDH": ("bɑhɑ", "1"), "MJH": ("behɑ", "1"),
        "HDI": ("bɑ", "1"), "SDI": ("bɑhɑ", "1"),
        "SNA": ("behɑ", "1"), "ORI": ("phulo", "2"),
    }),
    66: ("fruit", {
        "BAI": ("dʒʌʔo", "1"), "CHA": ("dʒo", "1"),
        "DIG": ("dʒo", "1"), "DUM": ("dʒʌ", "1"),
        "LAD": ("dʒo", "1"), "MAD": ("dʒɛ", "1"),
        "MOH": ("dʒo", "1"), "MUN": ("dʒo", "1"),
        "POD": ("dʒo", "1"), "UDA": ("dʒoʔ", "1"),
        "MCH": ("dʒo", "1"), "MDI": ("dʒo", "1"),
        "MDH": ("dʒoʔ", "1"), "MJH": ("dʒo", "1"),
        "HDI": ("dʒo", "1"), "SDI": ("dʒo", "1"),
        "SNA": ("dʒo", "1"), "ORI": ("pholo", "2"),
    }),
    67: ("mango", {
        "BAI": ("uli", "1"), "CHA": ("uli", "1"),
        "DIG": ("uli", "1"), "DUM": ("uli", "1"),
        "LAD": ("uli", "1"), "MAD": ("uli", "1"),
        "MOH": ("uli", "1"), "MUN": ("uli", "1"),
        "POD": ("uli", "1"), "UDA": ("uɭi", "1"),
        "MCH": ("uli", "1"), "MDI": ("uli", "1"),
        "MDH": ("uɭi", "1"), "MJH": ("uli", "1"),
        "HDI": ("uli", "1"), "SDI": ("ul", "1"),
        "SNA": ("uɭ", "1"), "ORI": ("ɑmbo", "2"),
    }),
    68: ("banana", {
        "BAI": ("kɑd̪ɑlɑ", "1"), "CHA": ("kɑd̪ɑl", "1"),
        "DIG": ("ked̪ɑlɑ", "1"), "DUM": ("kʌd̪ɑl", "1"),
        "LAD": ("kʌd̪ɑlɑ", "1"), "MAD": ("ked̪eɭ", "1"),
        "MOH": ("ked̪eɭ", "1"), "MUN": ("ked̪eɭ", "1"),
        "POD": ("kɑd̪ɑl", "1"), "UDA": ("ked̪ɑlɑ", "1"),
        "MCH": ("ked̪eɭ", "1"), "MDI": ("", ""),
        "MDH": ("ked̪ɑlɑ", "1"), "MJH": ("kɑd̪eɭ", "1"),
        "HDI": ("ked̪eɭ", "1"), "SDI": ("", ""),
        "SNA": ("kɑiɾɑ", "1"), "ORI": ("kodoli", "1"),
    }),
    69: ("wheat", {
        "BAI": ("gʌm", "1"), "CHA": ("gʌhʌm", "1"),
        "DIG": ("gɔhɔmo", "1"), "DUM": ("gʌhʌm", "1"),
        "LAD": ("gohom", "1"), "MAD": ("gehɛm", "1"),
        "MOH": ("gɔlɔm", "1"), "MUN": ("gom", "1"),
        "POD": ("gohom", "1"), "UDA": ("gohom", "1"),
        "MCH": ("gɔhɔm", "1"), "MDI": ("gohom", "1"),
        "MDH": ("gohom", "1"), "MJH": ("gɔhɔmo", "1"),
        "HDI": ("gom", "1"), "SDI": ("guhum", "1"),
        "SNA": ("gomo", "1"), "ORI": ("gohomõ", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 65 or (item == 66 and code == "BAI"):
        return "46", "41", "right"
    if item in {66, 67} or (item == 68 and code in ITEM68_LEFT):
        return "47", "42", "left"
    return "47", "42", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 88
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
