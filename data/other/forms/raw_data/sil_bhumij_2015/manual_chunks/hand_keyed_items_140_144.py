#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 140--144."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_140_144_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental and "
    "retroflex marks, nasalization, length, continuations, and the page break "
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
ITEM141_PAGE61 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH"}

DATA = {
    140: ("near", {
        "BAI": ("dʒʌpɑ", "1"), "CHA": ("sube", "2"),
        "DIG": ("dʒɛbɑʔ", "1"), "DUM": ("subʌ", "2"),
        "LAD": ("dʒʌpɑ", "1"), "MAD": ("sube", "2"),
        "MOH": ("subɑɾe", "2"), "MUN": ("dʒɛbɑʔ", "1"),
        "POD": ("sube", "2"), "UDA": ("dʒopɑ", "1"),
        "MCH": ("nɑɳe", "4"), "MDI": ("nipɑt̪", "5"),
        "MDH": ("dʒopɑ", "1"), "MJH": ("nɑɳe", "4"),
        "HDI": ("dʒɛbɑʔ", "1"), "SDI": ("soɾ", "3"),
        "SNA": ("suɾ", "3"), "ORI": ("pɑkːo", "7"),
    }),
    141: ("far", {
        "BAI": ("sɑŋgi", "1"), "CHA": ("sɑŋgin", "1"),
        "DIG": ("peɾkɑ", "2"), "DUM": ("sɑŋgiŋ", "1"),
        "LAD": ("sʌŋgin", "1"), "MAD": ("sɛŋgin", "1"),
        "MOH": ("sɑŋkiŋ", "1"), "MUN": ("sɑŋkiŋ", "1"),
        "POD": ("sʌŋgiŋ", "1"), "UDA": ("sɑŋgin", "1"),
        "MCH": ("sɑŋgiŋ", "1"), "MDI": ("sɑŋin", "1"),
        "MDH": ("sɑŋgin", "1"), "MJH": ("sɑŋkiŋ", "1"),
        "HDI": ("sɑɳiŋ", "1"), "SDI": ("sɑŋgin", "1"),
        "SNA": ("dʒel", "3"), "ORI": ("ɖuɾo", "4"),
    }),
    142: ("big", {
        "BAI": ("mɑɾɑŋ", "1"), "CHA": ("mʌɾʌŋ", "1"),
        "DIG": ("meɾɑŋ", "1"), "DUM": ("mɑɾɑŋ", "1"),
        "LAD": ("mʌɾɑŋ", "1"), "MAD": ("mɑɾɑŋ", "1"),
        "MOH": ("meɾɑŋ", "1"), "MUN": ("meɾɑŋ", "1"),
        "POD": ("mʌɾʌŋ", "1"), "UDA": ("mɑɾɑŋ", "1"),
        "MCH": ("meɾɑŋ", "1"), "MDI": ("mɑɾɑŋ", "1"),
        "MDH": ("mɑɾɑŋ", "1"), "MJH": ("meɾɑŋ", "1"),
        "HDI": ("meɾɑŋ", "1"), "SDI": ("mɑɾɑn", "1"),
        "SNA": ("meɾɑŋ", "1"), "ORI": ("boɽo", "2"),
    }),
    143: ("small", {
        "BAI": ("huɖiŋ", "1"), "CHA": ("huɖiŋ", "1"),
        "DIG": ("huɾiŋ", "1"), "DUM": ("huɖiŋ", "1"),
        "LAD": ("hoɾiŋ", "1"), "MAD": ("huɖiŋ", "1"),
        "MOH": ("hunʈiŋ", "1"), "MUN": ("huɾiŋ", "1"),
        "POD": ("huɖiŋ", "1"), "UDA": ("huɖiŋ", "1"),
        "MCH": ("huɾiŋ", "1"), "MDI": ("huɾiŋ", "1"),
        "MDH": ("huɖiŋ", "1"), "MJH": ("huɾiŋ", "1"),
        "HDI": ("huɾiŋ", "1"), "SDI": ("huɖiŋ | kɑtitʃʔ", "1 | 2"),
        "SNA": ("hɔpon", "1"), "ORI": ("sɑnõ", "3"),
    }),
    144: ("heavy", {
        "BAI": ("hɑmbɑl", "1"), "CHA": ("hʌmbɑl", "1"),
        "DIG": ("hembel", "1"), "DUM": ("hʌmbɑl", "1"),
        "LAD": ("hɑmbʌl", "1"), "MAD": ("hembɑlɑ", "1"),
        "MOH": ("t̪egɑɖɑ", "2"), "MUN": ("hembel", "1"),
        "POD": ("hʌmbɑlɑ", "1"), "UDA": ("hɑmbɑl", "1"),
        "MCH": ("hembɑl", "1"), "MDI": ("hɑmbɑl", "1"),
        "MDH": ("hɑmbɑl", "1"), "MJH": ("hembɑl", "1"),
        "HDI": ("hembɑl", "1"), "SDI": ("hɑmɑl", "1"),
        "SNA": ("hemɑl", "1"), "ORI": ("bɦɑɾi", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 140 or (item == 141 and code in ITEM141_PAGE61):
        return "61", "56", "right"
    if item in {141, 142, 143}:
        return "62", "57", "left"
    return "62", "57", "right"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 91
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
