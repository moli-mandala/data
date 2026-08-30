#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 60--64."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_060_064_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with item 60 and "
    "dense root variants rechecked at 800 dpi; text scaffold not accepted "
    "without cell visual match"
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
ITEM63_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}

DATA = {
    60: ("gold", {
        "BAI": ("sonɑ", "1"), "CHA": ("sonɑ", "1"),
        "DIG": ("sɛmɔŋɔm", "2"), "DUM": ("sonɑ", "1"),
        "LAD": ("sɑ̃məɾom", "2"), "MAD": ("sonɑ", "1"),
        "MOH": ("sənɑ", "1"), "MUN": ("sənɑ", "1"),
        "POD": ("sonɑ", "1"), "UDA": ("sunɑ", "1"),
        "MCH": ("sənɑ", "1"), "MDI": ("sɑmɾom", "2"),
        "MDH": ("sunɑ", "1"), "MJH": ("sənɑ", "1"),
        "HDI": ("sənɑ", "1"),
        "SDI": ("sonɑ | sɑmɑɾom", "1 | 2"),
        "SNA": ("sunɑ", "1"), "ORI": ("sunːɑ", "1"),
    }),
    61: ("tree", {
        "BAI": ("d̪ɑɾu", "1"), "CHA": ("d̪ɑɾu", "1"),
        "DIG": ("d̪ɑɾu", "1"), "DUM": ("d̪ʌɾu", "1"),
        "LAD": ("d̪ɑɾu", "1"), "MAD": ("d̪eɾu", "1"),
        "MOH": ("d̪eɾu", "1"), "MUN": ("d̪ɑɾu", "1"),
        "POD": ("d̪ɑɾi", "1"), "UDA": ("d̪ɑɾu", "1"),
        "MCH": ("d̪ɑɾu", "1"), "MDI": ("d̪ɑɾu", "1"),
        "MDH": ("d̪ɑɾu", "1"), "MJH": ("d̪ɑɾu", "1"),
        "HDI": ("d̪ɑɾu", "1"), "SDI": ("d̪ɑɾe", "1"),
        "SNA": ("d̪eɾe", "1"), "ORI": ("gɑtʃhɑ", "2"),
    }),
    62: ("leaf", {
        "BAI": ("sɑkɑm", "1"), "CHA": ("sikɑm", "1"),
        "DIG": ("sɛkɑm", "1"), "DUM": ("sikɑm", "1"),
        "LAD": ("sɑkɑm", "1"), "MAD": ("sekɑm", "1"),
        "MOH": ("sikɑm", "1"), "MUN": ("sikɑm", "1"),
        "POD": ("sikɑm", "1"), "UDA": ("sɑkɑm", "1"),
        "MCH": ("sɛkɑm", "1"), "MDI": ("sɑkɑm", "1"),
        "MDH": ("sɑkɑm", "1"), "MJH": ("sɛkɑm", "1"),
        "HDI": ("sɛkɛm", "1"), "SDI": ("sɑkɑm", "1"),
        "SNA": ("sɛkɑm", "1"), "ORI": ("potɾo", "2"),
    }),
    63: ("root", {
        "BAI": ("ɾeʔn", "1"), "CHA": ("ɾeʔt", "1"),
        "DIG": ("ɾeʔ", "1"), "DUM": ("ɾeheʔt", "1"),
        "LAD": ("ɾɛd̪ʔ", "1"), "MAD": ("ɾehet", "1"),
        "MOH": ("ɾeʔ", "1"), "MUN": ("ɾeʔɾ", "1"),
        "POD": ("ɾed̪ʔ", "1"), "UDA": ("ɾeʔt", "1"),
        "MCH": ("ɾeʔ", "1"), "MDI": ("dʒeɾ", "2"),
        "MDH": ("ɾeʔt", "1"),
        "MJH": ("ɾeʔheʔ | ɾeʔɾ", "1 | 1"),
        "HDI": ("tʃeɾoɾeʔ", "2"), "SDI": ("ɾehetʔ", "1"),
        "SNA": ("ɾehet", "1"), "ORI": ("tʃeɾo", "2"),
    }),
    64: ("thorn", {
        "BAI": ("dʒɑnum", "1"), "CHA": ("dʒɑnum", "1"),
        "DIG": ("dʒenum", "1"), "DUM": ("dʒʌnum", "1"),
        "LAD": ("dʒɑnum", "1"), "MAD": ("dʒenum", "1"),
        "MOH": ("dʒenum", "1"), "MUN": ("dʒenum", "1"),
        "POD": ("dʒɑnum", "1"), "UDA": ("dʒɑnum", "1"),
        "MCH": ("dʒɑnum", "1"), "MDI": ("dʒɑnum", "1"),
        "MDH": ("dʒɑnum", "1"), "MJH": ("dʒenum", "1"),
        "HDI": ("dʒenum", "1"), "SDI": ("dʒɑnum", "1"),
        "SNA": ("dʒenum", "1"), "ORI": ("kont̪ɑ", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 60:
        return "45", "40", "right"
    if item in {61, 62} or (item == 63 and code in ITEM63_LEFT):
        return "46", "41", "left"
    return "46", "41", "right"


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
    ) == 92
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
