#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 120--124."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_120_124_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental "
    "marks, aspiration, vowel quality, repeated responses, continuations, "
    "and the page break rechecked at 800 dpi; text scaffold neither supplied "
    "nor verified any reading"
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
    120: ("noon", {
        "BAI": ("t̪ikin", "1"), "CHA": ("dɦupʌɾ", "2"),
        "DIG": ("t̪ikin", "1"), "DUM": ("t̪ikin", "1"),
        "LAD": ("t̪ikin", "1"), "MAD": ("t̪ikin", "1"),
        "MOH": ("t̪ikin", "1"), "MUN": ("t̪ɑɾɑsiŋki", "3"),
        "POD": ("t̪ikin", "1"), "UDA": ("t̪ikin", "1"),
        "MCH": ("t̪ikin", "1"), "MDI": ("t̪ikin", "1"),
        "MDH": ("t̪ikin", "1"), "MJH": ("t̪ikin", "1"),
        "HDI": ("t̪ikin", "1"), "SDI": ("t̪ikin", "1"),
        "SNA": ("t̪ikin", "1"), "ORI": ("məd̪ɦjɑnə", "4"),
    }),
    121: ("evening/afternoon", {
        "BAI": ("ɑjub", "1"), "CHA": ("siŋgʌd", "3"),
        "DIG": ("t̪ɑɾɑsiŋ", "2"), "DUM": ("ɑjub", "1"),
        "LAD": ("ɑijupsɑ", "1"), "MAD": ("ɑjub", "1"),
        "MOH": ("t̪ɑɾɑsiŋ", "2"), "MUN": ("t̪ɑɾsiŋ", "2"),
        "POD": ("ʌub siŋgi | ʌub siŋgi", "1 | 3"), "UDA": ("ɑjub", "1"),
        "MCH": ("t̪ɑɾsiŋ", "2"), "MDI": ("ɑjub", "1"),
        "MDH": ("ɑjub", "1"), "MJH": ("t̪ɑɾɑsiŋki", "2"),
        "HDI": ("t̪ɑɾɑsiŋki", "2"), "SDI": ("ɑjup", "1"),
        "SNA": ("t̪ɑɾɑsiŋ", "2"), "ORI": ("sənd̪ɦjɑ", "4"),
    }),
    122: ("yesterday", {
        "BAI": ("holɑ", "1"), "CHA": ("holɑ", "1"),
        "DIG": ("hɔlɑ", "1"), "DUM": ("holɑ", "1"),
        "LAD": ("holɑ", "1"), "MAD": ("holɑ", "1"),
        "MOH": ("hɔlɑ", "1"), "MUN": ("hɔlɑ", "1"),
        "POD": ("holɑ", "1"), "UDA": ("holo", "1"),
        "MCH": ("holɑ", "1"), "MDI": ("holɑ", "1"),
        "MDH": ("holɑ", "1"), "MJH": ("hɔlɑ", "1"),
        "HDI": ("hɔlɑ", "1"), "SDI": ("holɑ", "1"),
        "SNA": ("hɔlɑ", "1"), "ORI": ("kɑli", "3"),
    }),
    123: ("today", {
        "BAI": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "CHA": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "DIG": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "DUM": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "LAD": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MAD": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MOH": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MUN": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "POD": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "UDA": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MCH": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MDI": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MDH": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "MJH": ("isiŋ", "2"),
        "HDI": ("t̪isiŋ | t̪isiŋ", "1 | 2"),
        "SDI": ("t̪ehen", "1"), "SNA": ("t̪ɛheŋ", "1"),
        "ORI": ("ɑdʒi", "3"),
    }),
    124: ("tomorrow", {
        "BAI": ("gɑpɑ", "1"), "CHA": ("gʌpɑ", "1"),
        "DIG": ("gepɑ", "1"), "DUM": ("gʌpɑ", "1"),
        "LAD": ("gɔpɑ", "1"), "MAD": ("gepɑ", "1"),
        "MOH": ("gɑpɑ", "1"), "MUN": ("gepɑ", "1"),
        "POD": ("gʌpɑ", "1"), "UDA": ("gepɑ", "1"),
        "MCH": ("gɑppɑ", "1"), "MDI": ("gɑpɑ", "1"),
        "MDH": ("gepɑ", "1"), "MJH": ("gɑpɑ", "1"),
        "HDI": ("gepɑ", "1"), "SDI": ("gɑpɑ", "1"),
        "SNA": ("gepɑ", "1"), "ORI": ("ɑsont̪ɑ kɑli", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {120, 121}:
        return "57", "52", "right"
    if item in {122, 123}:
        return "58", "53", "left"
    return "58", "53", "right"


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
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 105
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
