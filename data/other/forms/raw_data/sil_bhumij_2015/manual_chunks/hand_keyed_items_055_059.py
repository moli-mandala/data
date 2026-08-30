#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 55--59."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_055_059_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with nasalized and "
    "dental glyphs rechecked at 800 dpi; text scaffold not accepted without "
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
ITEM55_P44 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH", "MJH", "HDI", "SDI",
}
ITEM58_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH"}

DATA = {
    55: ("fire", {
        "BAI": ("seŋgel", "1"), "CHA": ("seŋgel", "1"),
        "DIG": ("seŋkel", "1"), "DUM": ("seŋgel", "1"),
        "LAD": ("siŋgel", "1"), "MAD": ("seŋgel", "1"),
        "MOH": ("seŋkel", "1"), "MUN": ("seŋkel", "1"),
        "POD": ("seŋgel", "1"), "UDA": ("seŋgel", "1"),
        "MCH": ("seŋkel", "1"), "MDI": ("seŋgel", "1"),
        "MDH": ("seŋgel", "1"), "MJH": ("seŋkel", "1"),
        "HDI": ("seŋkel", "1"), "SDI": ("seŋgel", "1"),
        "SNA": ("seŋkel", "1"), "ORI": ("nĩːɑ", "2"),
    }),
    56: ("smoke", {
        "BAI": ("sukul", "1"), "CHA": ("sukul", "1"),
        "DIG": ("sukul", "1"), "DUM": ("sukul", "1"),
        "LAD": ("sukuɾ", "1"), "MAD": ("sukul", "1"),
        "MOH": ("sukul", "1"), "MUN": ("sukul", "1"),
        "POD": ("sukul", "1"), "UDA": ("sukul", "1"),
        "MCH": ("sukul", "1"),
        "MDI": ("sukul | dɦuŋgiɑ", "1 | 3"),
        "MDH": ("sukul", "1"), "MJH": ("sukul", "1"),
        "HDI": ("mɔ̃ʔoʔ", "4"),
        "SDI": ("dɦũɑ̃ | dɦuŋgiɑ", "2 | 3"),
        "SNA": ("d̪ũɑ̃", "2"), "ORI": ("dɦuɑ̃", "2"),
    }),
    57: ("ash", {
        "BAI": ("t̪ʌɾe", "1"), "CHA": ("t̪oɾetʔ", "1"),
        "DIG": ("t̪ɔɾoj", "1"), "DUM": ("t̪oɾʌʔt̪", "1"),
        "LAD": ("t̪oɾoʔɛ", "1"), "MAD": ("t̪oɾet", "1"),
        "MOH": ("t̪ɔɾoj", "1"), "MUN": ("t̪ɔɾoj", "1"),
        "POD": ("t̪oɾodʒʔ", "1"), "UDA": ("t̪eɾneʔ", "1"),
        "MCH": ("t̪ɔɾoʔe", "1"), "MDI": ("t̪oɾoe", "1"),
        "MDH": ("t̪eɾneʔ", "1"), "MJH": ("t̪ɔɾe", "1"),
        "HDI": ("t̪ɔɾoj", "1"), "SDI": ("t̪oɾotʃʔ", "1"),
        "SNA": ("t̪ɔɾoj", "1"), "ORI": ("pɑ̃usə", "2"),
    }),
    58: ("mud", {
        "BAI": ("lʌsʌʔʌ", "1"), "CHA": ("kɑd̪ʌm", "2"),
        "DIG": ("hesɑ", "3"), "DUM": ("losoʔn", "1"),
        "LAD": ("losʌdʔ", "1"), "MAD": ("lɛsɛt", "1"),
        "MOH": ("lɔso", "1"), "MUN": ("lɔsot̪", "1"),
        "POD": ("losod | kɑd̪om", "1 | 2"),
        "UDA": ("loseeʔ", "1"), "MCH": ("lɔsot̪", "1"),
        "MDI": ("losod", "1"), "MDH": ("loseeʔ", "1"),
        "MJH": ("lɔso", "1"), "HDI": ("lɔsot̪", "1"),
        "SDI": ("losot", "1"), "SNA": ("hesɑ", "3"),
        "ORI": ("kɑd̪uə", "2"),
    }),
    59: ("dust", {
        "BAI": ("dɦuɾɑ", "1"), "CHA": ("dɦuɾɑ", "1"),
        "DIG": ("d̪ud̪ɑ", "1"), "DUM": ("dɦulʌ", "1"),
        "LAD": ("duɾɑʔ", "1"), "MAD": ("dɦule", "1"),
        "MOH": ("git̪il", "3"), "MUN": ("d̪ud̪ɑ", "1"),
        "POD": ("d̪ud̪e", "1"), "UDA": ("dɦulɑ", "1"),
        "MCH": ("d̪ɑud̪ɑ", "1"), "MDI": ("gɑɾdɑ", "2"),
        "MDH": ("dɦulɑ", "1"), "MJH": ("gund̪ɑ", "2"),
        "HDI": ("d̪uli", "1"), "SDI": ("dɦuɾi", "1"),
        "SNA": ("d̪ud̪i", "1"), "ORI": ("dɦuli", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 55 and code in ITEM55_P44:
        return "44", "39", "right"
    if item in {55, 56, 57} or (item == 58 and code in ITEM58_LEFT):
        return "45", "40", "left"
    return "45", "40", "right"


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
    ) == 93
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
