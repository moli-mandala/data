#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 15--19."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_015_019_hand_keyed.tsv"
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
FIRST_NINE = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD"}

DATA = {
    15: ("palm", {
        "BAI": ("t̪ɑlkɑ", "1"), "CHA": ("t̪ʌlkɑ", "1"),
        "DIG": ("t̪ɐlkɑ", "1"), "DUM": ("t̪i t̪ɑlkɑ", "1"),
        "LAD": ("t̪i t̪ɛlkɑ", "1"), "MAD": ("t̪it̪ophɑ", "2"),
        "MOH": ("t̪ɐlkɑ", "1"), "MUN": ("t̪ɐlkɑ", "1"),
        "POD": ("t̪ʌlʌkɑ t̪i", "1"), "UDA": ("t̪i t̪ɑlkɑ", "1"),
        "MCH": ("t̪ɐlkɑ", "1"), "MDI": ("t̪i t̪ɑlkɑ", "1"),
        "MDH": ("t̪i t̪ɑlkɑ", "1"), "MJH": ("t̪ɐlkɑ", "1"),
        "HDI": ("t̪ɐlkɑ", "1"), "SDI": ("t̪ɑlkɑ", "1"),
        "SNA": ("t̪ɐlkɑ", "1"), "ORI": ("toɭohɑto | pɑpuli", "3 | 4"),
    }),
    16: ("finger", {
        "BAI": ("ɑŋguɭi", "1"), "CHA": ("ɑŋguɖi", "1"),
        "DIG": ("ɖɑɖo", "3"), "DUM": ("ɑŋɭi", "1"),
        "LAD": ("ɖɑd̪o", "3"), "MAD": ("ɑŋguɭi", "1"),
        "MOH": ("ɐŋkiɖi", "1"), "MUN": ("ɑŋkuɽ", "1"),
        "POD": ("ʌŋgiɖi", "1"), "UDA": ("ɖɑɖo", "3"),
        "MCH": ("kɐʈuʔu", "2"), "MDI": ("kɑʈu", "2"),
        "MDH": ("ɖɑɖo", "3"), "MJH": ("ɐŋkʈi", "1"),
        "HDI": ("ɐŋkuɖi", "1"), "SDI": ("kɑʈup", "2"),
        "SNA": ("kəʈuʔp", "2"), "ORI": ("ɑŋguɭi", "1"),
    }),
    17: ("fingernail", {
        "BAI": ("nokhʌ", "3"), "CHA": ("sɑɾsɑɾ", "1"),
        "DIG": ("sɐɾseɾ", "1"), "DUM": ("sɑɾsɑɾ", "1"),
        "LAD": ("sɑɾsɑɾ", "1"), "MAD": ("sɑɾsɑɾ", "1"),
        "MOH": ("sɐɾɑɾ", "1"), "MUN": ("sɐɾsɑɾ", "1"),
        "POD": ("sʌɾsʌɾ", "1"), "UDA": ("sɐɾsɐɾ", "1"),
        "MCH": ("sɐɾseɾ", "1"), "MDI": ("ɾɑmɑ", "2"),
        "MDH": ("sɐɾsɐɾ", "1"), "MJH": ("ɾɑmɑ", "2"),
        "HDI": ("sɐɾɑɾ", "1"), "SDI": ("t̪i ɾɑmɑ", "2"),
        "SNA": ("ɾəmɑ", "2"), "ORI": ("noːkho", "3"),
    }),
    18: ("leg", {
        "BAI": ("kɑtɑ", "1"), "CHA": ("kɑtɑ", "1"),
        "DIG": ("kɐʈɑ", "1"), "DUM": ("kɑtɑ", "1"),
        "LAD": ("kɑrʈɑ", "1"), "MAD": ("kɐʈɑ", "1"),
        "MOH": ("kɐʈɑ", "1"), "MUN": ("kɐʈɑ", "1"),
        "POD": ("kʌtɑ", "1"), "UDA": ("kɑʈɑ", "1"),
        "MCH": ("kɐʈɑ", "1"), "MDI": ("kɑʈɑ", "1"),
        "MDH": ("kɑʈɑ", "1"), "MJH": ("kɐʈɑ", "1"),
        "HDI": ("kɐʈɑ", "1"), "SDI": ("dʒɑŋgɑ", "2"),
        "SNA": ("dʒəŋgɑ", "2"), "ORI": ("gudo", "3"),
    }),
    19: ("skin", {
        "BAI": ("hɑɾtɑ", "1"), "CHA": ("ũɾ", "2"),
        "DIG": ("hɐɾt̪ɑ", "1"), "DUM": ("hɑɾt̪ɑ", "1"),
        "LAD": ("ũɾ", "2"), "MAD": ("hɑɾtɑ", "1"),
        "MOH": ("hɐɾt̪ɑ", "1"), "MUN": ("uɾ", "2"),
        "POD": ("hʌɾt̪ɑ", "1"), "UDA": ("hɐɾt̪ɑ", "1"),
        "MCH": ("uɾ", "2"), "MDI": ("hɑɾtɑ | ũɾ", "1 | 2"),
        "MDH": ("hɐɾt̪ɑ", "1"), "MJH": ("uhuɾ", "2"),
        "HDI": ("", "0"), "SDI": ("hɑɾtɑ", "1"),
        "SNA": ("hɐɾt̪ɑ", "1"), "ORI": ("tʃɑɾəmõ", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 15 and code in FIRST_NINE:
        return "36", "31", "right"
    if item in {15, 16, 17} or (item == 18 and code == "BAI"):
        return "37", "32", "left"
    return "37", "32", "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            source_blank = item == 19 and code == "HDI"
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
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
