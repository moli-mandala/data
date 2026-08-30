#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 145--149."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_145_149_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental and "
    "retroflex marks, vowel quality, continuations, and the page break "
    "rechecked at 800 dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM146_PAGE62 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD"}
BLANKS = {(145, "MDI")}

DATA = {
    145: ("light", {
        "BAI": ("ɾʌbɑl", "1"), "CHA": ("ɾʌbɑl", "1"),
        "DIG": ("ɾebɑl", "1"), "DUM": ("ɾʌbɑl", "1"),
        "LAD": ("ɾʌbɑl", "1"), "MAD": ("ɾebɑl", "1"),
        "MOH": ("ɾebɑl", "1"), "MUN": ("lɑisɑ", "2"),
        "POD": ("lʌbɑɾ | ɾʌbɑl", "1 | 1"), "UDA": ("ɾebɑl", "1"),
        "MCH": ("ɾebɑl", "1"), "MDI": ("", ""),
        "MDH": ("ɾebɑl", "1"), "MJH": ("huɖiŋ", "3"),
        "HDI": ("lesɑ", "2"), "SDI": ("ɾɑwɑl | mɑɾsɑl", "1 | 4"),
        "SNA": ("ɾeʋɑl", "1"), "ORI": ("hɑlukɑ", "5"),
    }),
    146: ("above", {
        "BAI": ("tʃet̪ɑn", "1"), "CHA": ("tʃit̪ɑn", "1"),
        "DIG": ("tʃɛt̪n", "1"), "DUM": ("tʃilʌn", "1"),
        "LAD": ("tʃit̪ɑn", "1"), "MAD": ("tʃit̪ɑn", "1"),
        "MOH": ("tʃɛt̪n", "1"), "MUN": ("tʃɛt̪n", "1"),
        "POD": ("tʃit̪ɑn", "1"), "UDA": ("tʃet̪ɑn", "1"),
        "MCH": ("tʃɛt̪n", "1"), "MDI": ("tʃet̪ɑn", "1"),
        "MDH": ("tʃet̪ɑn", "1"), "MJH": ("tʃɛt̪n", "1"),
        "HDI": ("tʃɛt̪n", "1"), "SDI": ("tʃet̪ɑn | tʃot", "1 | 2"),
        "SNA": ("tʃɛt̪n", "1"), "ORI": ("upoɾo", "3"),
    }),
    147: ("below", {
        "BAI": ("lʌt̪ɑɾ", "1"), "CHA": ("lʌt̪ɑɾ", "1"),
        "DIG": ("lɛt̪ɛɾ", "1"), "DUM": ("lʌt̪ɑɾ", "1"),
        "LAD": ("lɑt̪ɑɾ", "1"), "MAD": ("lɛt̪ɑɾ", "1"),
        "MOH": ("lɛt̪ɑɾ", "1"), "MUN": ("lɛt̪ɑɾ", "1"),
        "POD": ("lʌt̪ɑɾ", "1"), "UDA": ("lɛt̪ɑɾ", "1"),
        "MCH": ("lɑt̪ɑɾ", "1"), "MDI": ("lɑt̪ɑɾ", "1"),
        "MDH": ("lɛt̪ɑɾ", "1"), "MJH": ("lɛt̪ɑɾ", "1"),
        "HDI": ("lɛt̪ɑɾ", "1"), "SDI": ("lɑt̪ɑɾ | phed", "1 | 2"),
        "SNA": ("lɛt̪ɑɾ", "1"), "ORI": ("t̪ələ", "3"),
    }),
    148: ("white", {
        "BAI": ("pundi", "1"), "CHA": ("pundi", "1"),
        "DIG": ("phuɳɖi", "1"), "DUM": ("pundi", "1"),
        "LAD": ("pundi", "1"), "MAD": ("pundi", "1"),
        "MOH": ("puɳɖi", "1"), "MUN": ("phuɳɖi", "1"),
        "POD": ("pundi", "1"), "UDA": ("pundi", "1"),
        "MCH": ("puɳɖi", "1"), "MDI": ("puɳɖi", "1"),
        "MDH": ("phuɳɖi", "1"), "MJH": ("puɳɖi", "1"),
        "HDI": ("puɳɖi", "1"), "SDI": ("poɳɖ", "1"),
        "SNA": ("puɳɖ", "1"), "ORI": ("ɖholɑ", "2"),
    }),
    149: ("black", {
        "BAI": ("hend̪e", "1"), "CHA": ("hend̪e", "1"),
        "DIG": ("hɛnt̪e", "1"), "DUM": ("hend̪e", "1"),
        "LAD": ("heɲd̪ɛ", "1"), "MAD": ("hend̪e", "1"),
        "MOH": ("hɛnt̪e", "1"), "MUN": ("hɛnt̪e", "1"),
        "POD": ("heɲd̪e", "1"), "UDA": ("hend̪e", "1"),
        "MCH": ("hɛnd̪e", "1"), "MDI": ("hend̪e", "1"),
        "MDH": ("hend̪e", "1"), "MJH": ("hɛnt̪e", "1"),
        "HDI": ("hɛnt̪e", "1"), "SDI": ("hend̪e", "1"),
        "SNA": ("hɛnt̪e", "1"), "ORI": ("koɭɑʔ", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 145 or (item == 146 and code in ITEM146_PAGE62):
        return "62", "57", "right"
    if item in {146, 147, 148}:
        return "63", "58", "left"
    return "63", "58", "right"


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
    ) == 93
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows
        if row["Review_Status"] == "attested" and row["Target"] == "yes"
    ) == 51
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
