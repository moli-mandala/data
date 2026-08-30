#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 25--29."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_025_029_hand_keyed.tsv"
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
ITEM25_P38 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA", "MCH", "MDI", "MDH"}
ITEM28_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD"}

DATA = {
    25: ("village", {
        "BAI": ("hɑt̪u", "1"), "CHA": ("hʌt̪u", "1"),
        "DIG": ("hɐt̪u", "1"), "DUM": ("hʌt̪u", "1"),
        "LAD": ("hɑ̃t̪u", "1"), "MAD": ("hɐt̪u", "1"),
        "MOH": ("hɐt̪u", "1"), "MUN": ("hɐt̪u", "1"),
        "POD": ("hʌt̪u", "1"), "UDA": ("hɑt̪u", "1"),
        "MCH": ("hɐt̪u", "1"), "MDI": ("hɑt̪u | ɖi", "1 | 2"),
        "MDH": ("hɑt̪u", "1"), "MJH": ("hɐt̪u", "1"),
        "HDI": ("hɐt̪u", "1"), "SDI": ("ɑto", "1"),
        "SNA": ("ɑt̪u", "1"), "ORI": ("gɾɑːmõ", "3"),
    }),
    26: ("house", {
        "BAI": ("oɖɑʔɑ", "1"), "CHA": ("oɖɑʔ", "1"),
        "DIG": ("ɔɽɑ", "1"), "DUM": ("oɖʔɑ", "1"),
        "LAD": ("oɽɑ", "1"), "MAD": ("oɖɑʔ", "1"),
        "MOH": ("ɔɽɑ", "1"), "MUN": ("ɔɽɑ", "1"),
        "POD": ("oɖɑʔɑ", "1"), "UDA": ("oɖɑʔ", "1"),
        "MCH": ("ɔɽɑ", "1"), "MDI": ("oɽɑ", "1"),
        "MDH": ("oɖɑʔ", "1"), "MJH": ("ɔɽɑ | ʋɑɑʔ", "1 | 2"),
        "HDI": ("ʋɑɑʔ", "2"), "SDI": ("oɾɑk", "1"),
        "SNA": ("ɔɖɑ", "1"), "ORI": ("gɦoɽo", "3"),
    }),
    27: ("roof", {
        "BAI": ("d̪ɑbeɑ", "4"), "CHA": ("sʌɖʌmi", "1"),
        "DIG": ("sɑɖimɑ", "1"), "DUM": ("d̪ɑlʌb", "3"),
        "LAD": ("sɑɾimɑ", "1"), "MAD": ("sedmi", "1"),
        "MOH": ("mut̪uɭ", "6"), "MUN": ("tʃɐnt̪ɑi", "2"),
        "POD": ("sʌɖimi t̪ed", "1"), "UDA": ("sɑɾimɑ", "1"),
        "MCH": ("sɑɽmi", "1"), "MDI": ("sɑɽɑmi", "1"),
        "MDH": ("tʃɑt", "2"), "MJH": ("khɐpɾɑ", "5"),
        "HDI": ("sidimɑ", "1"), "SDI": ("tʃɑl", "2"),
        "SNA": ("sɛɖim", "1"), "ORI": ("tʃhɑːto", "2"),
    }),
    28: ("door", {
        "BAI": ("silipiŋ", "2"), "CHA": ("tɑti", "1"),
        "DIG": ("silpiŋ", "2"), "DUM": ("tʌti", "1"),
        "LAD": ("silpiŋ", "2"), "MAD": ("tɐti", "1"),
        "MOH": ("d̪uʋɑɾ", "3"), "MUN": ("ʈɐʈi", "1"),
        "POD": ("tɑti", "1"), "UDA": ("duɑɾ", "3"),
        "MCH": ("d̪uʋɑɾ", "3"), "MDI": ("duɑɾ", "3"),
        "MDH": ("duɑɾ", "3"), "MJH": ("ʈɐʈɾɑ", "1"),
        "HDI": ("ʈɐʈi", "1"), "SDI": ("silpiŋ | kɑpɑt", "2 | 4"),
        "SNA": ("silpiŋ", "2"), "ORI": ("kɔbɑtɔ", "4"),
    }),
    29: ("firewood", {
        "BAI": ("sɑːn", "1"), "CHA": ("sɑn", "1"),
        "DIG": ("sɑn", "1"), "DUM": ("sɑhʌn", "1"),
        "LAD": ("sɑŋ", "1"), "MAD": ("sɑhɑn", "1"),
        "MOH": ("dʒulsɐhɑn", "1"), "MUN": ("sɑn", "1"),
        "POD": ("sɑn", "1"), "UDA": ("sɑhɑn", "1"),
        "MCH": ("sɑn", "1"), "MDI": ("sɑhɑn", "1"),
        "MDH": ("sɑhɑn", "1"), "MJH": ("sɐhɐn", "1"),
        "HDI": ("sɑn", "1"), "SDI": ("sɑhɑn", "1"),
        "SNA": ("sɐhɑn", "1"), "ORI": ("kɑːto", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 25 and code in ITEM25_P38:
        return "38", "33", "right"
    if item in {25, 26, 27} or (item == 28 and code in ITEM28_LEFT):
        return "39", "34", "left"
    return "39", "34", "right"


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
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
