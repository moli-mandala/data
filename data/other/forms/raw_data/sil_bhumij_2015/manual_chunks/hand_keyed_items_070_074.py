#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 70--74."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_070_074_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with difficult "
    "retroflex, nasal, and page-break cells rechecked at 800/1600 dpi; text "
    "scaffold not accepted without cell visual match"
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
ITEM71_PAGE47 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD"}
BLANKS = {
    (70, "SNA"), (70, "ORI"),
    (73, "DIG"), (73, "MOH"), (73, "MDI"), (73, "MJH"),
    (73, "SDI"), (73, "SNA"),
    (74, "MDI"),
}

DATA = {
    70: ("millet", {
        "BAI": ("dʒʌndʒʌdɑ", "3"), "CHA": ("gʌŋgɑi", "1"),
        "DIG": ("t̪eɾbudʒ", "6"), "DUM": ("gʌŋgɑi", "1"),
        "LAD": ("gʌŋgɑi", "1"), "MAD": ("gʌŋgɑi", "1"),
        "MOH": ("dʒɑne", "2"), "MUN": ("dʒɔnɑri", "2"),
        "POD": ("gʌŋgɑi", "1"), "UDA": ("gɑŋgɑi", "1"),
        "MCH": ("geŋgɑ", "1"), "MDI": ("kode", "5"),
        "MDH": ("gɑŋgɑi", "1"), "MJH": ("dʒɔneheɾ", "2"),
        "HDI": ("gɛlɛgeŋgɛi", "7"), "SDI": ("gundli", "4"),
        "SNA": ("", ""), "ORI": ("", ""),
    }),
    71: ("rice", {
        "BAI": ("tʃʌuli", "1"), "CHA": ("tʃʌuli", "1"),
        "DIG": ("tʃɑʊuli", "1"), "DUM": ("tʃʌuli", "1"),
        "LAD": ("tʃɑuli", "1"), "MAD": ("tʃɛuli", "1"),
        "MOH": ("tʃɑuli", "1"), "MUN": ("tʃɑuli", "1"),
        "POD": ("tʃɑuli", "1"), "UDA": ("mɑɳɖi", "2"),
        "MCH": ("tʃɑʊuli", "1"), "MDI": ("tʃɑuli", "1"),
        "MDH": ("mɑɳɖi", "2"), "MJH": ("tʃʌuli", "1"),
        "HDI": ("tʃɑuli", "1"), "SDI": ("tʃɑoli", "1"),
        "SNA": ("tʃɑʊele", "1"), "ORI": ("tʃɑwulo", "1"),
    }),
    72: ("potato", {
        "BAI": ("sɑŋgɑ", "2"), "CHA": ("ɑlu", "1"),
        "DIG": ("sɛŋkɑ", "2"), "DUM": ("golɑɭui", "1"),
        "LAD": ("ɑlu", "1"), "MAD": ("sɑŋgɑ", "2"),
        "MOH": ("ɑlu", "1"), "MUN": ("ɑlu", "1"),
        "POD": ("golɑɭu", "1"), "UDA": ("sɑŋgɑ", "2"),
        "MCH": ("ɑlu", "2"), "MDI": ("ɑlu", "1"),
        "MDH": ("sɑŋgɑ", "1"), "MJH": ("ɑlu", "2"),
        "HDI": ("ɑlu", "1"), "SDI": ("ɑlu", "1"),
        "SNA": ("ɑlu", "1"), "ORI": ("ɑɭu", "1"),
    }),
    73: ("eggplant", {
        "BAI": ("bejgɑdɑ", "1"), "CHA": ("biŋgɑd", "1"),
        "DIG": ("", ""), "DUM": ("biŋgɑɭ", "1"),
        "LAD": ("biŋgəɾɑ", "1"), "MAD": ("beŋgɑd", "1"),
        "MOH": ("", ""), "MUN": ("bɛŋgɑɾ", "1"),
        "POD": ("biŋgɑd", "1"), "UDA": ("biŋgɑd", "1"),
        "MCH": ("bɛŋgɑdɑ", "1"), "MDI": ("", ""),
        "MDH": ("biŋgɑd", "1"), "MJH": ("", ""),
        "HDI": ("biŋkɑ", "1"), "SDI": ("", ""),
        "SNA": ("", ""), "ORI": ("bɑiŋgoɾõ", "1"),
    }),
    74: ("groundnut", {
        "BAI": ("bɑɖɑm", "1"), "CHA": ("bʌɖɑm", "1"),
        "DIG": ("bɛɖɑm", "1"), "DUM": ("bʌɖɑm", "1"),
        "LAD": ("bʌɖʌm", "1"), "MAD": ("bɑɖɑm", "1"),
        "MOH": ("bɛɖɑm", "1"), "MUN": ("bɛɖɑm", "1"),
        "POD": ("bɑɖɑm", "1"), "UDA": ("bɑɖɑm", "1"),
        "MCH": ("bɛɖɑm", "1"), "MDI": ("", ""),
        "MDH": ("bɑɖɑm", "1"), "MJH": ("muɸuli", "2"),
        "HDI": ("bɛɖɛm", "1"), "SDI": ("bɑɖɑm", "1"),
        "SNA": ("bɛɖɑm", "1"), "ORI": ("tʃinbɑɖɑm", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 70 or (item == 71 and code in ITEM71_PAGE47):
        return "47", "42", "right"
    if item in {71, 72, 73}:
        return "48", "43", "left"
    return "48", "43", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 9
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 81
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
