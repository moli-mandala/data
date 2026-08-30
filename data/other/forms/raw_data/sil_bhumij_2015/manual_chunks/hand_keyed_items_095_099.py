#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 95--99."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_095_099_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "nasalization, glottalization, continuation lines, and page/column breaks "
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
ITEM99_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}
BLANKS = {(97, "MUN"), (97, "MDI")}

DATA = {
    95: ("dog", {
        "BAI": ("seʈɑ", "1"), "CHA": ("seʈɑ", "1"),
        "DIG": ("sɛʈɑʔ", "1"), "DUM": ("seʈɑ", "1"),
        "LAD": ("sɛʈɑ", "1"), "MAD": ("sɛʈɑ", "1"),
        "MOH": ("sɛʈɑ", "1"), "MUN": ("sɛʈɑ", "1"),
        "POD": ("siʈɑ", "1"), "UDA": ("seʈɑ", "1"),
        "MCH": ("seʈɑ", "1"), "MDI": ("seʈɑ", "1"),
        "MDH": ("seʈɑ", "1"), "MJH": ("sɛʈɑ", "1"),
        "HDI": ("siʈɑ", "1"), "SDI": ("seʈɑ", "1"),
        "SNA": ("sɛʈɑ", "1"), "ORI": ("kukuɾɑ", "2"),
    }),
    96: ("snake", {
        "BAI": ("biŋ", "1"), "CHA": ("biŋ", "1"),
        "DIG": ("biŋ", "1"), "DUM": ("biŋ", "1"),
        "LAD": ("biŋ", "1"), "MAD": ("biŋ", "1"),
        "MOH": ("biŋ", "1"), "MUN": ("biŋ", "1"),
        "POD": ("biŋ", "1"), "UDA": ("biŋ", "1"),
        "MCH": ("biŋ", "1"), "MDI": ("biŋ", "1"),
        "MDH": ("biŋ", "1"), "MJH": ("biŋ", "1"),
        "HDI": ("biŋ", "1"), "SDI": ("biŋ | kɑl", "1 | 2"),
        "SNA": ("biŋ", "1"), "ORI": ("sɑpo", "3"),
    }),
    97: ("monkey", {
        "BAI": ("gɑɭi", "1"), "CHA": ("gɑɖi", "1"),
        "DIG": ("gɑɖi", "1"), "DUM": ("gʌɭi", "1"),
        "LAD": ("gɑɖi | hanumɑn", "1 | 2"), "MAD": ("gɛɖi", "1"),
        "MOH": ("hɑɳu", "2"), "MUN": ("", ""),
        "POD": ("gɑɖi", "1"), "UDA": ("gɑɖi", "1"),
        "MCH": ("gɑɖi", "1"), "MDI": ("", ""),
        "MDH": ("gɑɖi", "1"), "MJH": ("seɾɑ", "3"),
        "HDI": ("gai", "1"), "SDI": ("gɑ̃ɾĩ", "1"),
        "SNA": ("heɳu", "2"), "ORI": ("mɑŋkəɾə", "4"),
    }),
    98: ("mosquito", {
        "BAI": ("sikɳi", "1"), "CHA": ("luʈi", "2"),
        "DIG": ("sikini", "1"), "DUM": ("siknĩ", "1"),
        "LAD": ("sikəɳi", "1"), "MAD": ("sikɳi", "1"),
        "MOH": ("luʈi", "2"), "MUN": ("luʈi", "2"),
        "POD": ("sikini", "1"), "UDA": ("sikiɳi", "1"),
        "MCH": ("sikiɳi", "1"), "MDI": ("bɦusɾi", "4"),
        "MDH": ("sikiɳi", "1"), "MJH": ("sikɳi", "1"),
        "HDI": ("sikiŋ", "1"), "SDI": ("sikɾĩtʃ", "1"),
        "SNA": ("guɖu", "3"), "ORI": ("motʃhɑ", "5"),
    }),
    99: ("ant", {
        "BAI": ("mũʔi", "1"), "CHA": ("muːi", "1"),
        "DIG": ("mui", "1"), "DUM": ("moʔt", "1"),
        "LAD": ("mũi", "1"), "MAD": ("muit", "1"),
        "MOH": ("mui", "1"), "MUN": ("mui", "1"),
        "POD": ("muiʔ", "1"), "UDA": ("muʔi", "1"),
        "MCH": ("muʔi", "1"), "MDI": ("mui", "1"),
        "MDH": ("muʔi", "1"), "MJH": ("mui", "1"),
        "HDI": ("mũi", "1"), "SDI": ("mutʃʔ", "2"),
        "SNA": ("muŋ", "1"), "ORI": ("mɑtʃi", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {95, 96}:
        return "52", "47", "right"
    if item in {97, 98} or (item == 99 and code in ITEM99_LEFT):
        return "53", "48", "left"
    return "53", "48", "right"


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
    ) == 90
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
