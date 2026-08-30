#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 10--14."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_010_014_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages; "
    "text scaffold not accepted without cell visual match"
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
FIRST_TEN = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}

DATA = {
    10: ("tongue", {
        "BAI": ("ɑlɑŋ", "1"), "CHA": ("ɑlʌŋ", "1"),
        "DIG": ("ɐlɑŋ", "1"), "DUM": ("ɑlɑŋ", "1"),
        "LAD": ("leʔe", "2"), "MAD": ("ɑlɑŋ", "1"),
        "MOH": ("ɐlɑŋ", "1"), "MUN": ("lɛlej", "2"),
        "POD": ("ʌlɑŋ", "1"), "UDA": ("ɑlɑŋ", "1"),
        "MCH": ("lɛʔje", "2"), "MDI": ("ɑlɑŋ | leʔe", "1 | 2"),
        "MDH": ("ɑlɑŋ", "1"), "MJH": ("ɐlɑŋ", "1"),
        "HDI": ("lɛʔje", "2"), "SDI": ("ɑlɑŋ", "1"),
        "SNA": ("ɐlɑŋ", "1"), "ORI": ("dʒibɦə", "3"),
    }),
    11: ("breast", {
        "BAI": ("", "0"), "CHA": ("t̪uwɑ", "3"),
        "DIG": ("kuɭɐm", "1"), "DUM": ("nunu", "2"),
        "LAD": ("nunu", "2"), "MAD": ("nunu", "2"),
        "MOH": ("kuɖɑm", "1"), "MUN": ("", "0"),
        "POD": ("nunu", "2"), "UDA": ("nunu", "2"),
        "MCH": ("kuɖɑm", "1"), "MDI": ("nunu", "2"),
        "MDH": ("nunu", "2"), "MJH": ("nunu", "2"),
        "HDI": ("kujem", "1"), "SDI": ("koɾɑm | nunu", "1 | 2"),
        "SNA": ("", "0"), "ORI": ("tʃɑt̪i", "4"),
    }),
    12: ("belly", {
        "BAI": ("lɑi", "1"), "CHA": ("lɑi", "1"),
        "DIG": ("lɑʔi", "1"), "DUM": ("leʔ", "1"),
        "LAD": ("lɑʔi", "1"), "MAD": ("lɑi", "1"),
        "MOH": ("lɑi", "1"), "MUN": ("lɑi", "1"),
        "POD": ("lei", "1"), "UDA": ("lɑiʔ", "1"),
        "MCH": ("lɑʔi", "1"), "MDI": ("lɑi", "1"),
        "MDH": ("lɑiʔ", "1"), "MJH": ("lɑheʔ", "1"),
        "HDI": ("lɑi", "1"), "SDI": ("lɑʔe | dodʒok", "1 | 2"),
        "SNA": ("lɑʔ", "1"), "ORI": ("pet̪t̪o", "3"),
    }),
    13: ("arm", {
        "BAI": ("t̪i", "1"), "CHA": ("t̪i", "1"),
        "DIG": ("t̪i", "1"), "DUM": ("supu", "2"),
        "LAD": ("t̪i", "1"), "MAD": ("supu", "2"),
        "MOH": ("t̪i", "1"), "MUN": ("t̪i", "1"),
        "POD": ("supu", "2"), "UDA": ("ti supu", "2"),
        "MCH": ("t̪iʔi", "1"), "MDI": ("t̪i", "1"),
        "MDH": ("ti supu", "2"), "MJH": ("t̪iʔi", "1"),
        "HDI": ("t̪i", "1"), "SDI": ("t̪i | sopo", "1 | 2"),
        "SNA": ("t̪i", "1"), "ORI": ("hɑʈo", "3"),
    }),
    14: ("elbow", {
        "BAI": ("gonti", "2"), "CHA": ("ukɑʔ", "1"),
        "DIG": ("uk | ukɑ", "1 | 1"), "DUM": ("ukʌʔ", "1"),
        "LAD": ("ukːɑ", "1"), "MAD": ("mukɑ", "1"),
        "MOH": ("uk", "1"), "MUN": ("ukɑ", "1"),
        "POD": ("uke", "1"), "UDA": ("ukɑʔ", "1"),
        "MCH": ("ukɑ", "1"), "MDI": ("ukɑ", "1"),
        "MDH": ("ukɑʔ", "1"), "MJH": ("ukɑ", "1"),
        "HDI": ("uke", "1"), "SDI": ("mokɑ", "1"),
        "SNA": ("mukɑ", "1"), "ORI": ("koini", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 10 and code in FIRST_TEN:
        return "35", "30", "right"
    if item in {10, 11, 12}:
        return "36", "31", "left"
    return "36", "31", "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            source_blank = item == 11 and code in {"BAI", "MUN", "SNA"}
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
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
