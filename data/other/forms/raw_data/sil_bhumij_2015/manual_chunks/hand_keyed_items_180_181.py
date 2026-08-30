#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 180--181."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_180_181_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with continuation "
    "lines, dental marks, and the page break rechecked at 800 dpi; OCR/PDF text "
    "neither supplied nor verified any reading"
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

DATA = {
    180: ("many", {
        "BAI": ("dʒɑtkɑ", "7"), "CHA": ("bedʒɑjŋ", "1"),
        "DIG": ("sɑŋki", "2"), "DUM": ("bedʒʌŋ | d̪eheɾ", "1 | 4"),
        "LAD": ("sɑŋgi", "2"), "MAD": ("bidʒeŋ", "1"),
        "MOH": ("tʃimin", "5"), "MUN": ("bidʒɑŋ", "1"),
        "POD": ("bidʒen | puɾe", "1 | 3"), "UDA": ("sɑŋgi", "2"),
        "MCH": ("puɾɑʔɑ", "3"),
        "MDI": ("d̪ɦeɾ | ɑn hut | isu", "4 | 9 | 10"),
        "MDH": ("sɑŋgi", "2"), "MJH": ("t̪himbɑ", "6"),
        "HDI": ("puɾejə", "3"), "SDI": ("ɑemɑ | ɑdi", "11 | 12"),
        "SNA": ("dɦɛheɾ", "4"), "ORI": ("bohut", "13"),
    }),
    181: ("all", {
        "BAI": ("dʒʌt̪ɔ", "1"), "CHA": ("dʒʌt̪ʌ", "1"),
        "DIG": ("dʒot̪o", "1"), "DUM": ("dʒʌt̪o", "1"),
        "LAD": ("sʌbin", "2"), "MAD": ("dʒʌnt̪o", "1"),
        "MOH": ("dʒɔt̪uɑ", "1"), "MUN": ("dʒot̪o", "1"),
        "POD": ("dʒet̪e", "1"), "UDA": ("dʒot̪o", "1"),
        "MCH": ("sobenɑʔɑ", "2"), "MDI": ("soben", "2"),
        "MDH": ("dʒot̪o", "1"), "MJH": ("t̪himbɑgi", "3"),
        "HDI": ("səbɛn", "2"), "SDI": ("dʒot̪o | sɑnɑm", "1 | 4"),
        "SNA": ("dʒɔt̪o", "1"), "ORI": ("sobu", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = (
                ("69", "64", "right") if item == 180
                else ("70", "65", "left")
            )
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
    assert len(rows) == 2 * 18 == 36
    assert sum(
        len(row["Manual_Transcription"].split(" | ")) for row in rows
    ) == 42
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Target"] == "yes"
    ) == 22
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
