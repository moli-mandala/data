#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 160--164."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_160_164_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflex and "
    "dental marks, vowel quality, continuations, and the page break rechecked "
    "at 800 dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM161_PAGE65 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}
BLANKS = {
    (160, "MJH"),
    (161, "MDI"), (161, "MJH"),
    (162, "MDI"), (162, "MJH"),
    (163, "MJH"),
}

DATA = {
    160: ("ten", {
        "BAI": ("ɖɑs", "2"), "CHA": ("ɖɑs", "2"),
        "DIG": ("ɖəs", "2"), "DUM": ("ɖʌs", "2"),
        "LAD": ("gelijʌ", "1"), "MAD": ("ɖɛs", "2"),
        "MOH": ("ɖəs", "2"), "MUN": ("ɖəstɑ", "2"),
        "POD": ("ɖos", "2"), "UDA": ("ɖɛs", "2"),
        "MCH": ("geleje", "1"), "MDI": ("gel | geleɑ", "1 | 1"),
        "MDH": ("ɖɛs", "2"), "MJH": ("", ""),
        "HDI": ("gelijɑ", "1"), "SDI": ("gel", "1"),
        "SNA": ("ɖəso", "2"), "ORI": ("ɖɑso", "2"),
    }),
    161: ("eleven", {
        "BAI": ("egɑɾ", "2"), "CHA": ("gʌɾɑ", "2"),
        "DIG": ("ɛgɑɾo", "2"), "DUM": ("egɑɾ", "2"),
        "LAD": ("gel mijʌd̪ʔ", "1"), "MAD": ("egɑɾ", "2"),
        "MOH": ("ɛgɑɾo", "2"), "MUN": ("ɛgɑɾotɑ", "2"),
        "POD": ("eg gɑɾo", "2"), "UDA": ("egɑɾ", "2"),
        "MCH": ("gelbɑɾijɑ", "1"), "MDI": ("", ""),
        "MDH": ("egɑɾ", "2"), "MJH": ("", ""),
        "HDI": ("gelmiɑ", "1"), "SDI": ("gel mit", "1"),
        "SNA": ("ɛgɑɾo", "2"), "ORI": ("egɑɾo", "2"),
    }),
    162: ("twelve", {
        "BAI": ("bɑɾ", "2"), "CHA": ("bɑɾɑ", "2"),
        "DIG": ("bɑɾo", "2"), "DUM": ("bɑɾ", "2"),
        "LAD": ("gel bɑɾiʌ", "1"), "MAD": ("bɑɾ", "2"),
        "MOH": ("bɑɾo", "2"), "MUN": ("bɑɾotɑ", "2"),
        "POD": ("bɑɾo", "2"), "UDA": ("bɑɾ", "2"),
        "MCH": ("gelbɑɾijɑ", "1"), "MDI": ("", ""),
        "MDH": ("bɑɾ", "2"), "MJH": ("", ""),
        "HDI": ("gelbɑɾijə", "1"), "SDI": ("gel bɑɾeɑ", "1"),
        "SNA": ("bɑɾo", "2"), "ORI": ("bɑɾo", "2"),
    }),
    163: ("twenty", {
        "BAI": ("kudie", "2"), "CHA": ("bis", "3"),
        "DIG": ("kudije", "2"), "DUM": ("kodie", "2"),
        "LAD": ("hɛsi", "1"), "MAD": ("mot hisi", "1"),
        "MOH": ("kudije | bis", "2 | 3"), "MUN": ("monisi", "1"),
        "POD": ("hisi", "1"), "UDA": ("mot hisi | kodi", "1 | 2"),
        "MCH": ("mid̪isi", "1"), "MDI": ("hisi", "1"),
        "MDH": ("mot hisi", "1"), "MJH": ("", ""),
        "HDI": ("hisi", "1"), "SDI": ("isi", "1"),
        "SNA": ("kudije", "2"), "ORI": ("koɾie", "2"),
    }),
    164: ("one hundred", {
        "BAI": ("mitʔ sə", "1"), "CHA": ("mot sʌ", "1"),
        "DIG": ("soj", "2"), "DUM": ("mod sʌ", "1"),
        "LAD": ("monʔɛ hɛsi", "1"), "MAD": ("mot so", "1"),
        "MOH": ("mɔʔso", "1"), "MUN": ("moneisi", "1"),
        "POD": ("mod so", "1"), "UDA": ("moʔt sɑ", "1"),
        "MCH": ("mod̪ehisi", "1"), "MDI": ("mid sɑe", "1"),
        "MDH": ("moʔt sɑ", "1"), "MJH": ("miʔsou", "1"),
        "HDI": ("miʔso", "1"), "SDI": ("sɑe", "2"),
        "SNA": ("soje", "2"), "ORI": ("eko sɑho", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 160 or (item == 161 and code in ITEM161_PAGE65):
        return "65", "60", "right"
    if item in {161, 162, 163}:
        return "66", "61", "left"
    return "66", "61", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 6
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 87
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows
        if row["Review_Status"] == "attested" and row["Target"] == "yes"
    ) == 52
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
