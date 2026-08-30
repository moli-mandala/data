#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 45--49."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_045_049_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dense page "
    "breaks and glyphs enlarged separately; text scaffold not accepted without "
    "cell visual match"
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
ITEM45_P42 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI",
}
ITEM48_LEFT = {"BAI", "CHA", "DIG", "DUM"}
BLANKS = {(45, "SDI"), (48, "BAI"), (49, "SDI")}

DATA = {
    45: ("rain", {
        "BAI": ("gɑmɑ", "1"), "CHA": ("gʌmɑ", "1"),
        "DIG": ("gɑmɑ", "1"), "DUM": ("gɑmɑ", "1"),
        "LAD": ("gɔmʌ", "1"), "MAD": ("gɛmɑ", "1"),
        "MOH": ("gɑmɑ", "1"), "MUN": ("gɑmɑ", "1"),
        "POD": ("gʌmɑ", "1"),
        "UDA": ("d̪ɑʔɑʔ gɑmɑ | d̪ɑʔɑʔ | d̪ɑʔɑʔ gɑmɑ", "1 | 2 | 2"),
        "MCH": ("d̪ɑʔɑʔ", "2"), "MDI": ("dʒɑɾgi", "3"),
        "MDH": ("d̪ɑʔɑʔ gɑmɑ | d̪ɑʔɑʔ | d̪ɑʔɑʔ gɑmɑ", "1 | 2 | 2"),
        "MJH": ("d̪ɑʔɑʔ", "2"), "HDI": ("gɛmɑ", "1"),
        "SDI": ("", ""), "SNA": ("d̪ɑ", "2"),
        "ORI": ("boɾosɑ", "4"),
    }),
    46: ("water", {
        "BAI": ("d̪ɑʔɑ", "1"), "CHA": ("d̪ɑʔ", "1"),
        "DIG": ("d̪ɑʔɑʔ", "1"), "DUM": ("d̪ɑʔ", "1"),
        "LAD": ("d̪ɑʔʌ", "1"), "MAD": ("d̪ɑʔ", "1"),
        "MOH": ("d̪ɑʔ", "1"), "MUN": ("d̪ɑ", "1"),
        "POD": ("d̪ɑʔɑ", "1"), "UDA": ("d̪ɑʔɑ", "1"),
        "MCH": ("d̪ɑʔɑʔ", "1"), "MDI": ("d̪ɑ", "1"),
        "MDH": ("d̪ɑʔɑ", "1"), "MJH": ("d̪ɑʔɑʔ", "1"),
        "HDI": ("d̪ɑɑʔ", "1"), "SDI": ("dɑk", "1"),
        "SNA": ("d̪ɑ", "1"), "ORI": ("pɑni", "2"),
    }),
    47: ("river", {
        "BAI": ("gɑd̪ɑ", "1"), "CHA": ("gʌd̪ɑ", "1"),
        "DIG": ("ged̪ɑ", "1"), "DUM": ("gɑd̪ɑ", "1"),
        "LAD": ("gɑd̪ʌ", "1"), "MAD": ("gɑd̪ɑ", "1"),
        "MOH": ("ged̪ɑ", "1"), "MUN": ("ged̪ɑ", "1"),
        "POD": ("gʌdɑ", "1"), "UDA": ("gɑd̪ɑ", "1"),
        "MCH": ("gɑd̪ɑ", "1"), "MDI": ("gɑd̪ɑ", "1"),
        "MDH": ("gɑd̪ɑ", "1"), "MJH": ("nɑi", "2"),
        "HDI": ("ged̪ɑ", "1"), "SDI": ("gɑd̪ɑ", "1"),
        "SNA": ("ged̪ɑ", "1"), "ORI": ("nod̪i", "2"),
    }),
    48: ("cloud", {
        "BAI": ("", ""), "CHA": ("ɾembil", "1"),
        "DIG": ("ɾimɑl", "1"), "DUM": ("ɾemil", "1"),
        "LAD": ("ɾɛmbʔil", "1"), "MAD": ("ɾemil", "1"),
        "MOH": ("ɾimbil", "1"), "MUN": ("ɾimil", "1"),
        "POD": ("ɾimil", "1"), "UDA": ("ɾimil", "1"),
        "MCH": ("ɾimbil", "1"), "MDI": ("ɾimil", "1"),
        "MDH": ("ɾimbil", "1"), "MJH": ("ɾimil", "1"),
        "HDI": ("ɾimil", "1"), "SDI": ("ɾimil | lɑhɾɑ", "1 | 2"),
        "SNA": ("ɾəblɑ", "3"), "ORI": ("megɦo", "4"),
    }),
    49: ("lightning", {
        "BAI": ("gɦʌdɑgɑti", "3"), "CHA": ("bidʒʌli", "2"),
        "DIG": ("hitʃiɾ", "1"), "DUM": ("bidʒili", "2"),
        "LAD": ("bidʒʌlɑu", "2"), "MAD": ("bidʒli", "2"),
        "MOH": ("bidʒili", "2"), "MUN": ("bidʒili", "2"),
        "POD": ("itʃiɾ t̪ɑdɑ | bidʒlo", "1 | 2"),
        "UDA": ("hitʃiɾ", "1"), "MCH": ("hitʃiɾ", "1"),
        "MDI": ("hitʃiɾ | t̪heɾ", "1 | 4"),
        "MDH": ("hitʃiɾ", "1"), "MJH": ("bidʒili", "2"),
        "HDI": ("bidʒili", "2"), "SDI": ("", ""),
        "SNA": ("bidʒili", "2"), "ORI": ("bidʒuli", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 45 and code in ITEM45_P42:
        return "42", "37", "right"
    if item == 45 and code == "MDH":
        return "42-43", "37-38", "right/left"
    if item in {45, 46, 47} or (item == 48 and code in ITEM48_LEFT):
        return "43", "38", "left"
    return "43", "38", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 3
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
