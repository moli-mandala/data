#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 80--84."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_080_084_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflex, "
    "nasal, continuation-line, and page/column-break cells rechecked at 800 "
    "dpi; text scaffold not accepted without cell visual match"
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
ITEM81_PAGE49 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH", "MJH", "HDI",
}
ITEM84_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD"}
BLANKS = {(80, "SDI"), (81, "MDI")}

DATA = {
    80: ("tomato", {
        "BAI": ("bilɑʈi beŋgedɑ", "1"), "CHA": ("bilɑʈi", "1"),
        "DIG": ("bɛlɑʈi", "1"), "DUM": ("bilɑʈi", "1"),
        "LAD": ("bilɑʈi", "1"), "MAD": ("bilɑʈi beŋgɑd", "1"),
        "MOH": ("bilɑʈi", "1"), "MUN": ("bilɑʈi", "1"),
        "POD": ("bilɑʈi", "1"), "UDA": ("bilɑʈi beŋgɑd", "1"),
        "MCH": ("dʒolbɛʈɑ", "3"), "MDI": ("bilɑiʈi", "1"),
        "MDH": ("bilɑʈi beŋgɑd", "1"), "MJH": ("bilɛʈi | pɛʈɛl", "1 | 2"),
        "HDI": ("bilɛʈi", "1"), "SDI": ("", ""),
        "SNA": ("bilɑʈi bɛŋkɑɭ", "1"), "ORI": ("bilɑʈi", "1"),
    }),
    81: ("cabbage", {
        "BAI": ("bondɑ kobi", "2"), "CHA": ("potʌm kopi", "1"),
        "DIG": ("pɔʈɔŋkobi", "1"), "DUM": ("pʌtʌɾ kobi", "1"),
        "LAD": ("pɛʈom kobi", "1"), "MAD": ("potom kobi", "1"),
        "MOH": ("pɔʈɔŋkobi", "1"), "MUN": ("pɔʈoŋkobi", "1"),
        "POD": ("potom kobi", "1"), "UDA": ("potɛm kobi", "1"),
        "MCH": ("pɔʈoŋkobi", "1"), "MDI": ("", ""),
        "MDH": ("potɛm kobi", "1"), "MJH": ("pɔʈɔŋkobi", "1"),
        "HDI": ("pɔʈom kobi", "1"), "SDI": ("kubi ɑɾɑk", "3"),
        "SNA": ("pɔʈoŋkobi", "1"), "ORI": ("bənd̪hɑ kobi", "2"),
    }),
    82: ("oil", {
        "BAI": ("sunum", "1"), "CHA": ("sunum", "1"),
        "DIG": ("sunum", "1"), "DUM": ("sunum", "1"),
        "LAD": ("sunum", "1"), "MAD": ("sunum", "1"),
        "MOH": ("sunum", "1"), "MUN": ("sunum", "1"),
        "POD": ("sunum", "1"), "UDA": ("sunum", "1"),
        "MCH": ("sunum", "1"), "MDI": ("sunum", "1"),
        "MDH": ("sunum", "1"), "MJH": ("sunum", "1"),
        "HDI": ("sunum", "1"), "SDI": ("sunum", "1"),
        "SNA": ("sunum", "1"), "ORI": ("t̪elo", "2"),
    }),
    83: ("salt", {
        "BAI": ("buluŋ", "1"), "CHA": ("buluŋ", "1"),
        "DIG": ("bulum", "1"), "DUM": ("buluŋ", "1"),
        "LAD": ("bʌluŋ", "1"), "MAD": ("buluŋ", "1"),
        "MOH": ("bulum", "1"), "MUN": ("buluŋ", "1"),
        "POD": ("buluŋ", "1"), "UDA": ("buluŋ", "1"),
        "MCH": ("buluŋ", "1"), "MDI": ("buluŋ", "1"),
        "MDH": ("buluŋ", "1"), "MJH": ("buluŋ", "1"),
        "HDI": ("buluŋ", "1"), "SDI": ("buluŋ", "1"),
        "SNA": ("buluŋ", "1"), "ORI": ("luɳə | nũno", "2 | 3"),
    }),
    84: ("meat", {
        "BAI": ("tʃilu", "1"), "CHA": ("dʒilu", "1"),
        "DIG": ("dʒilu", "1"), "DUM": ("dzilu", "1"),
        "LAD": ("dʒilu", "1"), "MAD": ("dʒilu", "1"),
        "MOH": ("dʒilu", "1"), "MUN": ("dʒilu", "1"),
        "POD": ("dʒilu", "1"), "UDA": ("dʒilu", "1"),
        "MCH": ("dʒilu", "1"), "MDI": ("dʒilu", "1"),
        "MDH": ("dʒilu", "1"), "MJH": ("mɑs", "3"),
        "HDI": ("dʒilu", "1"), "SDI": ("beɾel dʒel | dʒel", "1 | 1"),
        "SNA": ("dʒil", "1"), "ORI": ("mɑŋtso", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 80 or (item == 81 and code in ITEM81_PAGE49):
        return "49", "44", "right"
    if item in {81, 82, 83} or (item == 84 and code in ITEM84_LEFT):
        return "50", "45", "left"
    return "50", "45", "right"


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
    ) == 91
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
