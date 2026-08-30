#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 85--89."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_085_089_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "laterals, rhotics, length, continuation lines, and the column break "
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
ITEM89_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD"}

DATA = {
    85: ("fat", {
        "BAI": ("iʈil", "1"), "CHA": ("iʈil", "1"),
        "DIG": ("iʈil", "1"), "DUM": ("iʈil", "1"),
        "LAD": ("iʈil", "1"), "MAD": ("iʈil", "1"),
        "MOH": ("iʈil", "1"), "MUN": ("iʈil", "1"),
        "POD": ("iʈil", "1"), "UDA": ("iʈil", "1"),
        "MCH": ("iʈil", "1"), "MDI": ("iʈil", "1"),
        "MDH": ("iʈil", "1"), "MJH": ("iʈil", "1"),
        "HDI": ("iʈil", "1"), "SDI": ("iʈil", "1"),
        "SNA": ("iʈil", "1"), "ORI": ("tʃəɾbi", "2"),
    }),
    86: ("fish", {
        "BAI": ("hai", "1"), "CHA": ("hai", "1"),
        "DIG": ("hai", "1"), "DUM": ("hʌi", "1"),
        "LAD": ("hai", "1"), "MAD": ("hɛi", "1"),
        "MOH": ("hɛi", "1"), "MUN": ("hai", "1"),
        "POD": ("hei", "1"), "UDA": ("hai", "1"),
        "MCH": ("hai", "1"), "MDI": ("hai", "1"),
        "MDH": ("hai", "1"), "MJH": ("hʌku", "2"),
        "HDI": ("hʌku", "2"), "SDI": ("hɑku", "2"),
        "SNA": ("hɛku", "2"), "ORI": ("mɑːtʃo", "3"),
    }),
    87: ("chicken", {
        "BAI": ("sim", "1"), "CHA": ("sim", "1"),
        "DIG": ("sim", "1"), "DUM": ("sim", "1"),
        "LAD": ("sim", "1"), "MAD": ("sim", "1"),
        "MOH": ("sim", "1"), "MUN": ("sim", "1"),
        "POD": ("sim", "1"), "UDA": ("sim", "1"),
        "MCH": ("sim", "1"), "MDI": ("sim", "1"),
        "MDH": ("sim", "1"), "MJH": ("sim", "1"),
        "HDI": ("sim", "1"), "SDI": ("sim", "1"),
        "SNA": ("sim", "1"), "ORI": ("kukudɑ", "2"),
    }),
    88: ("egg", {
        "BAI": ("peʈɑɭi", "1"), "CHA": ("pedʌo", "5"),
        "DIG": ("pɛʈɑɖi", "1"), "DUM": ("bitʃʌɭi", "1"),
        "LAD": ("ʌɳɖʌ", "4"), "MAD": ("pitheɭu", "1"),
        "MOH": ("pɛʈɑɭu", "1"), "MUN": ("bili", "3"),
        "POD": ("bitʃɑɖi", "1"), "UDA": ("peʈɑɭi", "1"),
        "MCH": ("dʒeɾɔm", "2"), "MDI": ("dʒɑɾom | bili", "2 | 3"),
        "MDH": ("peʈɑɭi", "1"), "MJH": ("dʒeɾɑm", "2"),
        "HDI": ("dʒeɾɑm", "2"), "SDI": ("bele", "3"),
        "SNA": ("bili", "3"), "ORI": ("oɳɖɑ", "4"),
    }),
    89: ("cow", {
        "BAI": ("gai", "1"), "CHA": ("gʌi", "1"),
        "DIG": ("gaj", "1"), "DUM": ("gʌi", "1"),
        "LAD": ("gundi", "2"), "MAD": ("gɛi", "1"),
        "MOH": ("gej", "1"), "MUN": ("gundi", "2"),
        "POD": ("gei | uɾi", "1 | 3"), "UDA": ("gai", "1"),
        "MCH": ("gai", "1"), "MDI": ("gai | gundi", "1 | 2"),
        "MDH": ("gai", "1"), "MJH": ("gundi", "2"),
        "HDI": ("uɾiiʔ", "3"), "SDI": ("gai | dɑŋgri", "1 | 4"),
        "SNA": ("gej", "1"), "ORI": ("gai", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {85, 86}:
        return "50", "45", "right"
    if item in {87, 88} or (item == 89 and code in ITEM89_LEFT):
        return "51", "46", "left"
    return "51", "46", "right"


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
    ) == 94
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
