#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 105--109."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_105_109_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "nasal places, aspiration, length, continuations, and the page/column "
    "break rechecked at 800 dpi; text scaffold neither supplied nor verified "
    "any reading"
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
ITEM109_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA", "MCH"}
BLANKS = {(107, "MDI")}

DATA = {
    105: ("father", {
        "BAI": ("ɑbɑ", "1"), "CHA": ("bɑbu", "1"),
        "DIG": ("ɑbɑ", "1"), "DUM": ("bɑ", "1"),
        "LAD": ("ɑbɑ", "1"), "MAD": ("bɛbu", "1"),
        "MOH": ("bɑbu", "1"), "MUN": ("bɑbu", "1"),
        "POD": ("bɑbu", "1"), "UDA": ("ɑbɑ", "1"),
        "MCH": ("ɑbɑ", "1"), "MDI": ("ɑbɑ | ɑpu", "1 | 1"),
        "MDH": ("ɑbɑ", "1"), "MJH": ("ɑbɑ", "1"),
        "HDI": ("ɑpuŋ", "1"), "SDI": ("ɑpɑ | bɑ", "1 | 1"),
        "SNA": ("bɑbɑ", "1"), "ORI": ("bɑpɑ", "1"),
    }),
    106: ("mother", {
        "BAI": ("mɑ", "1"), "CHA": ("mɑɳ", "1"),
        "DIG": ("mɑ", "1"), "DUM": ("mɑŋ", "1"),
        "LAD": ("mɑː", "1"), "MAD": ("mɑ", "1"),
        "MOH": ("mɑŋ", "1"), "MUN": ("mɑŋ", "1"),
        "POD": ("mɑŋ", "1"), "UDA": ("mɑ", "1"),
        "MCH": ("ɪŋgɑ", "3"), "MDI": ("eŋgɑ", "3"),
        "MDH": ("mɑ", "1"), "MJH": ("ummɑ", "2"),
        "HDI": ("ɪŋkɑ", "3"), "SDI": ("eŋgɑ | ɑyo", "3 | 4"),
        "SNA": ("ɑjo", "4"), "ORI": ("mɑʔ", "1"),
    }),
    107: ("older brother", {
        "BAI": ("mɑɾɑɳɖɑɖɑ", "1"), "CHA": ("ɖɑɖɑ", "1"),
        "DIG": ("ɖɑɖɑ", "1"), "DUM": ("mɑɾɑɳɖɑɖɑ", "1"),
        "LAD": ("ɖɑɖɑ", "1"), "MAD": ("ɖɑɖɑ", "1"),
        "MOH": ("meɾɑɳɖɑɖɑ", "1"), "MUN": ("meɾɑɳɖɑɖɑ", "1"),
        "POD": ("ɖɑɖɑ", "1"), "UDA": ("ɖɑɖɑ", "1"),
        "MCH": ("meɾɑɳɖɑɖɑ", "1"), "MDI": ("", ""),
        "MDH": ("mɑɾɑɲhɑgɑ", "1"), "MJH": ("ɖɑɖɑ", "1"),
        "HDI": ("ɖɑɖɑ", "1"), "SDI": ("ɖɑɖɑ", "1"),
        "SNA": ("məɾɑɳɖɑɖɑ", "1"), "ORI": ("nõnɑʔ", "2"),
    }),
    108: ("younger brother", {
        "BAI": ("bɑbu", "6"), "CHA": ("hon hʌgɑ", "1"),
        "DIG": ("huɖiɲhɑgɑ", "1"), "DUM": ("huɖiɲboko", "4"),
        "LAD": ("boko", "2"), "MAD": ("huɖiɲɳi", "5"),
        "MOH": ("huɖiɲbɦɑi", "3"), "MUN": ("huɖiɲbɦɑi", "3"),
        "POD": ("boko", "2"), "UDA": ("huɖiɳɖɑɖɑ", "7"),
        "MCH": ("huɖiɲhɑgɑ", "1"), "MDI": ("hɑgɑ", "1"),
        "MDH": ("huɖiɲhɑgɑ", "1"), "MJH": ("hɛgɑ", "1"),
        "HDI": ("unɖi", "5"), "SDI": ("bokot koɾɑ", "2"),
        "SNA": ("hepen bɑbu", "6"), "ORI": ("tʃhoʈɑ bɦɑi", "3"),
    }),
    109: ("older sister", {
        "BAI": ("ɖɑi", "1"), "CHA": ("ɖiɖi", "3"),
        "DIG": ("ɖɑi", "1"), "DUM": ("nɑnɑ", "2"),
        "LAD": ("ɖɑi", "1"), "MAD": ("nɑnɑ", "2"),
        "MOH": ("ɖiɖi", "3"), "MUN": ("nenɑ", "2"),
        "POD": ("nɑnɑ", "2"), "UDA": ("meɾɑɳɖɑi", "1"),
        "MCH": ("meɾɑɳɖɑi", "1"), "MDI": ("nɑnɑ", "2"),
        "MDH": ("meɾɑɳɖɑi", "1"), "MJH": ("nenɑ", "2"),
        "HDI": ("ɖɑi", "1"), "SDI": ("ɖɑi | ɑdʒi", "1 | 4"),
        "SNA": ("məɾɛm ɖɑi", "1"), "ORI": ("nɑnːi | ɖiɖi", "2 | 3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {105, 106}:
        return "54", "49", "right"
    if item in {107, 108} or (item == 109 and code in ITEM109_LEFT):
        return "55", "50", "left"
    return "55", "50", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 94
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
