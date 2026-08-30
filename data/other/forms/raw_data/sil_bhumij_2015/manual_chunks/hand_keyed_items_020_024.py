#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 20--24."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_020_024_hand_keyed.tsv"
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
ITEM20_P37 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA", "MCH", "MDI", "MDH"}
ITEM23_LEFT = {"BAI", "CHA", "DIG", "DUM"}

DATA = {
    20: ("bone", {
        "BAI": ("dʒɑŋ", "1"), "CHA": ("dʒɑŋ", "1"),
        "DIG": ("dʒɐŋ", "1"), "DUM": ("dʒɑŋ", "1"),
        "LAD": ("dʒɑŋ", "1"), "MAD": ("dʒɑŋ", "1"),
        "MOH": ("dʒɑŋ", "1"), "MUN": ("dʒɑŋ", "1"),
        "POD": ("dʒɑŋ", "1"), "UDA": ("dʒɑŋ", "1"),
        "MCH": ("dʒɐŋ", "1"), "MDI": ("dʒɑŋ", "1"),
        "MDH": ("dʒɑŋ", "1"), "MJH": ("dʒɑŋ", "1"),
        "HDI": ("", "0"), "SDI": ("dʒɑŋ", "1"),
        "SNA": ("dʒɑŋ", "1"), "ORI": ("hɑːdo", "2"),
    }),
    21: ("heart", {
        "BAI": ("", "0"), "CHA": ("mɑjʌm oɖɑʔ", "2"),
        "DIG": ("dʒibon", "1"), "DUM": ("mʌjʌm kundi", "5"),
        "LAD": ("mɑjɑm oɽɑ", "2"), "MAD": ("mɑjɐm kundi", "5"),
        "MOH": ("kɔldʒɑ", "3"), "MUN": ("kɔldʒɑ", "3"),
        "POD": ("mɑjɑm oɖɑʔ", "2"), "UDA": ("mɐjɐm bhundɑɾ", "6"),
        "MCH": ("dʒibon", "1"), "MDI": ("dʒi | bukɑ", "1 | 4"),
        "MDH": ("mɐjɐm bhundɑɾ", "6"), "MJH": ("kɔldʒɑ", "3"),
        "HDI": ("dʒibɔn", "1"), "SDI": ("boko | ontoɾ", "4 | 7"),
        "SNA": ("kɔldʒɑ", "3"), "ORI": ("həɾud̪ɑio", "8"),
    }),
    22: ("blood", {
        "BAI": ("mɑjʌm", "1"), "CHA": ("mɑjʌm", "1"),
        "DIG": ("mɐjɑm", "1"), "DUM": ("mɑjʌm", "1"),
        "LAD": ("mɑjʌ", "1"), "MAD": ("mɑjɐm", "1"),
        "MOH": ("mɐjɑm", "1"), "MUN": ("mɐjɔm", "1"),
        "POD": ("mʌjʌm", "1"), "UDA": ("mɑjɐm", "1"),
        "MCH": ("mɐjɔm", "1"), "MDI": ("mɑjom | ɾokot", "1 | 2"),
        "MDH": ("mɑjɐm", "1"), "MJH": ("mɐjɔm", "1"),
        "HDI": ("mɐjɐm", "1"), "SDI": ("mɑ̃jɑ̃m", "1"),
        "SNA": ("mɑjɑm", "1"), "ORI": ("ɾɑkto", "2"),
    }),
    23: ("urine", {
        "BAI": ("duki", "1"), "CHA": ("dʌɖo", "2"),
        "DIG": ("nono", "3"), "DUM": ("dʌdʌ", "2"),
        "LAD": ("duki", "1"), "MAD": ("dɐdɐ", "2"),
        "MOH": ("ɖoɖo", "2"), "MUN": ("ɖuki", "1"),
        "POD": ("", "0"), "UDA": ("dɐɖo", "2"),
        "MCH": ("ɖuki", "1"), "MDI": ("ɖuki | ɖoɖo", "1 | 2"),
        "MDH": ("dɐɖo", "2"), "MJH": ("ɖukid̪ɑʔɑʔ", "1"),
        "HDI": ("ɖuki", "1"), "SDI": ("ɑɖoeɑk", "4"),
        "SNA": ("", "0"), "ORI": ("mut̪t̪o", "5"),
    }),
    24: ("feces", {
        "BAI": ("dʒɦʌɖɑ", "2"), "CHA": ("iːiʔ", "1"),
        "DIG": ("", "0"), "DUM": ("eʔ", "1"),
        "LAD": ("iʔi", "1"), "MAD": ("iʔi", "1"),
        "MOH": ("iʔ", "1"), "MUN": ("", "0"),
        "POD": ("iʔi", "1"), "UDA": ("iʔi", "1"),
        "MCH": ("iʔ", "1"), "MDI": ("eee | idʒɦ", "1 | 3"),
        "MDH": ("iʔi", "1"), "MJH": ("iʔ", "1"),
        "HDI": ("iʔiʔ", "1"), "SDI": ("dʒidʒɑ | itʃʔ", "2 | 3"),
        "SNA": ("", "0"), "ORI": ("dʒɑɽɑ", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 20 and code in ITEM20_P37:
        return "37", "32", "right"
    if item in {20, 21, 22} or (item == 23 and code in ITEM23_LEFT):
        return "38", "33", "left"
    return "38", "33", "right"


def build_rows():
    rows = []
    blanks = {(20, "HDI"), (21, "BAI"), (23, "POD"), (23, "SNA"),
              (24, "DIG"), (24, "MUN"), (24, "SNA")}
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            source_blank = (item, code) in blanks
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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 7
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
