#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 100--104."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_100_104_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "nasalization, dental marks, continuations, small-cap vowels, and the "
    "page/column break rechecked at 800 dpi; text scaffold neither supplied "
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
ITEM104_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}

DATA = {
    100: ("spider", {
        "BAI": ("binɖiri", "1"), "CHA": ("ʈʌɳʈulɑ", "2"),
        "DIG": ("binʈiri", "1"), "DUM": ("binɖi", "1"),
        "LAD": ("biŋɖiri", "1"), "MAD": ("ʈeɳʈulbɑhɑ", "2"),
        "MOH": ("binʈirɑm", "1"), "MUN": ("ʈeɳʈulɑ", "2"),
        "POD": ("ʈɑɳʈɑle", "2"), "UDA": ("bindri", "1"),
        "MCH": ("binʈirɑm", "1"), "MDI": ("bindrɑm", "1"),
        "MDH": ("bindri", "1"), "MJH": ("binʈirɑm", "1"),
        "HDI": ("binʈirɑm", "1"), "SDI": ("bindi", "1"),
        "SNA": ("binʈɪ", "1"), "ORI": ("buɖɦiɑɳi", "3"),
    }),
    101: ("name", {
        "BAI": ("nuʈum", "1"), "CHA": ("numu", "2"),
        "DIG": ("luʈum", "1"), "DUM": ("numu", "2"),
        "LAD": ("nuʈum", "1"), "MAD": ("numu", "2"),
        "MOH": ("nuʈum", "1"), "MUN": ("numu", "2"),
        "POD": ("numu", "2"), "UDA": ("luʈum", "1"),
        "MCH": ("nuʈum", "1"), "MDI": ("nuʈum | num", "1 | 2"),
        "MDH": ("luʈum", "1"), "MJH": ("nuʈum", "1"),
        "HDI": ("nuʈum", "1"), "SDI": ("nuʈum", "1"),
        "SNA": ("ɲuʈum", "1"), "ORI": ("nɑːmɔ", "2"),
    }),
    102: ("man", {
        "BAI": ("hɔɾo", "1"), "CHA": ("hoɖo", "1"),
        "DIG": ("koɖɑhõn", "2"), "DUM": ("hʌɖʌ", "1"),
        "LAD": ("hoɾo", "1"), "MAD": ("hoɖo", "1"),
        "MOH": ("hɔɖo", "1"), "MUN": ("hɔɖo", "1"),
        "POD": ("hoɖo", "1"), "UDA": ("hoɖo", "1"),
        "MCH": ("koɖɑ", "2"), "MDI": ("hoɾo | koɾɑ", "1 | 2"),
        "MDH": ("hoɖo", "1"), "MJH": ("hɔɖo", "1"),
        "HDI": ("ho", "1"), "SDI": ("hoɾ", "1"),
        "SNA": ("hoɖ", "1"), "ORI": ("moniʃo", "3"),
    }),
    103: ("woman", {
        "BAI": ("kuɖi hon", "1"), "CHA": ("iɾɑ", "2"),
        "DIG": ("kuɖihõn", "1"), "DUM": ("kuɖi", "1"),
        "LAD": ("kuɾi", "1"), "MAD": ("iɾɑ", "2"),
        "MOH": ("kuɖiʔjeɾɑ", "1"), "MUN": ("jeɾɑ", "2"),
        "POD": ("iɾɑ", "2"), "UDA": ("kuɖi hoɖo", "1"),
        "MCH": ("kuɖi", "1"), "MDI": ("kuɾi", "1"),
        "MDH": ("kuɖi hon", "1"), "MJH": ("kuɖi", "1"),
        "HDI": ("jeɾɑ", "2"), "SDI": ("mɑedʒiu", "3"),
        "SNA": ("kuɖiɑpon", "1"), "ORI": ("st̪ɾi", "4"),
    }),
    104: ("child", {
        "BAI": ("hon", "1"), "CHA": ("hone", "1"),
        "DIG": ("hõn", "1"), "DUM": ("koɖɑ", "3"),
        "LAD": ("hɔn", "1"), "MAD": ("koɖɑ", "3"),
        "MOH": ("hɔne", "1"), "MUN": ("hɔne", "1"),
        "POD": ("hone", "1"), "UDA": ("hon", "1"),
        "MCH": ("hõn", "1"), "MDI": ("hon", "1"),
        "MDH": ("hon", "1"), "MJH": ("hõn", "1"),
        "HDI": ("hõn", "1"), "SDI": ("giɖɾɑ", "2"),
        "SNA": ("giɖɾə", "2"), "ORI": ("pilːɑ", "4"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {100, 101}:
        return "53", "48", "right"
    if item in {102, 103} or (item == 104 and code in ITEM104_LEFT):
        return "54", "49", "left"
    return "54", "49", "right"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 92
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
