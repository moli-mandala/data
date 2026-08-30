#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 90--94."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_090_094_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "rhotics, nasalization, continuation lines, and page/column breaks "
    "rechecked at 800 dpi; text scaffold not accepted without cell visual match"
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
ITEM91_PAGE51 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH", "MJH", "HDI", "SDI",
}
ITEM94_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH"}

DATA = {
    90: ("water buffalo", {
        "BAI": ("keɖɑ", "1"), "CHA": ("kiɖɑ | mɔ̃s", "1 | 2"),
        "DIG": ("mũisi", "2"), "DUM": ("kiɖɑ", "1"),
        "LAD": ("kɛɖɑ", "1"), "MAD": ("mɔ̃isi", "2"),
        "MOH": ("mɔisi", "2"), "MUN": ("keɖɑ", "1"),
        "POD": ("keɖɑ | mɔ̃isi", "1 | 2"), "UDA": ("mɔ̃isi", "2"),
        "MCH": ("keɖɑ", "1"), "MDI": ("keɖɑ | birkeɾɑ", "1 | 4"),
        "MDH": ("keɖɑ", "1"), "MJH": ("kɛɖɑ", "1"),
        "HDI": ("keɖɑ", "1"), "SDI": ("kɑɖɑ | bitkil", "1 | 3"),
        "SNA": ("kɛɖɑ", "1"), "ORI": ("moiʃɑ", "2"),
    }),
    91: ("milk", {
        "BAI": ("ʈuwɑ", "1"), "CHA": ("ʈuwɑ", "1"),
        "DIG": ("ʈowɑ", "1"), "DUM": ("ʈuɑ", "1"),
        "LAD": ("ʈowʌ", "1"), "MAD": ("ʈowɑ", "1"),
        "MOH": ("ʈɔwɑ", "1"), "MUN": ("ʈɔwɑ", "1"),
        "POD": ("ʈowɑ", "1"), "UDA": ("ʈowɑ", "1"),
        "MCH": ("ʈɔwɑ", "1"), "MDI": ("ʈoɑ", "1"),
        "MDH": ("ʈowɑ", "1"), "MJH": ("ʈɔwɑ", "1"),
        "HDI": ("ʈuwɑ", "1"), "SDI": ("ʈoɑ", "1"),
        "SNA": ("ʈowɑ", "1"), "ORI": ("khiro", "2"),
    }),
    92: ("horns", {
        "BAI": ("ɖiriŋ", "1"), "CHA": ("ɖiriŋ", "1"),
        "DIG": ("ɖiriŋ", "1"), "DUM": ("ɖiriŋ", "1"),
        "LAD": ("ɖiriŋ", "1"), "MAD": ("ɖiriŋ", "1"),
        "MOH": ("ɖiriŋ", "1"), "MUN": ("ɖiriŋ", "1"),
        "POD": ("ɖiriŋ", "1"), "UDA": ("ɖiriŋ", "1"),
        "MCH": ("ɖiriŋ", "1"), "MDI": ("ɖiriŋ", "1"),
        "MDH": ("ɖiriŋ", "1"), "MJH": ("ɖiriŋ", "1"),
        "HDI": ("ɖiriŋ", "1"), "SDI": ("siŋgɑ | ɖɑbe", "2 | 3"),
        "SNA": ("ɖɛreŋ", "1"), "ORI": ("siŋgə", "2"),
    }),
    93: ("tail", {
        "BAI": ("tʃɑʔtlɑm", "1"), "CHA": ("tʃʌʔtlʌm", "1"),
        "DIG": ("tʃʌʔlom", "1"), "DUM": ("tʃʌʔlom", "1"),
        "LAD": ("tʃɑlʌm", "1"), "MAD": ("tʃɑtlɛm", "1"),
        "MOH": ("tʃɑʔilom", "1"), "MUN": ("tʃɛʔlom", "1"),
        "POD": ("tʃʌʔlom", "1"), "UDA": ("tʃɑtlɛm", "1"),
        "MCH": ("tʃɑʔlom", "1"), "MDI": ("tʃɑʔlʌm", "1"),
        "MDH": ("tʃɑtlɛm", "1"), "MJH": ("tʃɛʔlom", "1"),
        "HDI": ("tʃɑʔlom", "1"), "SDI": ("tʃɑnɖbol", "2"),
        "SNA": ("tʃɑlom", "1"), "ORI": ("lɑndʒo", "3"),
    }),
    94: ("goat", {
        "BAI": ("meɾɑm", "1"), "CHA": ("meɾʌm", "1"),
        "DIG": ("meɾɔm", "1"), "DUM": ("meɾom", "1"),
        "LAD": ("meɾom", "1"), "MAD": ("mɛɾom", "1"),
        "MOH": ("meɾɔm", "1"), "MUN": ("mɛɾom", "1"),
        "POD": ("meɾʌm", "1"), "UDA": ("mɛɾom", "1"),
        "MCH": ("meɾɔm", "1"), "MDI": ("meɾom", "1"),
        "MDH": ("mɛɾom", "1"), "MJH": ("meɾom", "1"),
        "HDI": ("mɛɾɔm", "1"), "SDI": ("meɾom", "1"),
        "SNA": ("mɛɾom", "1"), "ORI": ("tʃheɭi", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 90 or (item == 91 and code in ITEM91_PAGE51):
        return "51", "46", "right"
    if item in {91, 92, 93} or (item == 94 and code in ITEM94_LEFT):
        return "52", "47", "left"
    return "52", "47", "right"


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
    assert sum(
        len(row["Manual_Transcription"].split(" | ")) for row in rows
    ) == 95
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
