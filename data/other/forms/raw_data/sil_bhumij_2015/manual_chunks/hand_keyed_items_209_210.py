#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 209--210."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_209_210_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 800-dpi rendered final PDF page and tight "
    "source-image crops, with vowel qualities, engmas, glottal stops, source "
    "length colons, unlabeled continuation variants, and separately numbered "
    "responses checked cell by cell; OCR/PDF text neither supplied nor verified "
    "any reading"
)
SITES = {
    "BAI": ("Bhumij", "Baigodia", True), "CHA": ("Bhumij", "Champi", True),
    "DIG": ("Bhumij", "Dighinuasahi", True), "DUM": ("Bhumij", "Dumadie", True),
    "LAD": ("Bhumij", "Ladhiramsai", True), "MAD": ("Bhumij", "Madhupur", True),
    "MOH": ("Bhumij", "Mohuldiha", True), "MUN": ("Bhumij", "Munduy", True),
    "POD": ("Bhumij", "Podadiha", True), "UDA": ("Bhumij/Mundari", "Udala", True),
    "MCH": ("Mundari", "Chalagi", False), "MDI": ("Mundari", "Dictionary", False),
    "MDH": ("Mundari", "Dhungarisai", False), "MJH": ("Mundari", "Jharmunda", False),
    "HDI": ("Ho", "Dillisore", False), "SDI": ("Santali", "Dictionary", False),
    "SNA": ("Santali", "Nayarangamotia", False), "ORI": ("Oriya", "Cuttack", False),
}
DATA = {
    209: ("you (2nd pl)", {
        "BAI": ("ɑpe", "1"), "CHA": ("ɑpe", "1"), "DIG": ("", ""),
        "DUM": ("ɑpe", "1"), "LAD": ("ɑpeʔ", "1"), "MAD": ("ɑpe", "1"),
        "MOH": ("ɛpe", "1"), "MUN": ("ɑŋku", "2"), "POD": ("ʌpe", "1"),
        "UDA": ("ɑpe", "1"), "MCH": ("ɑpe", "1"), "MDI": ("ɑpeɑ", "1"),
        "MDH": ("ɑpe", "1"), "MJH": ("ɛpe", "1"), "HDI": ("ɛpe", "1"),
        "SDI": ("ɑpe", "1"), "SNA": ("ɛpe", "1"), "ORI": ("ɑponõ", "3"),
    }),
    210: ("they (3rd pl)", {
        "BAI": ("ɑko", "1"), "CHA": ("ɑko", "1"), "DIG": ("", ""),
        "DUM": ("inku", "1"), "LAD": ("ɑko", "1"), "MAD": ("ɑko", "1"),
        "MOH": ("inku", "1"), "MUN": ("ɛko", "1"), "POD": ("ʌko", "1"),
        "UDA": ("ɑko", "1"), "MCH": ("inku", "1"), "MDI": ("ɑkiŋ", "1"),
        "MDH": ("ɑko | ɑko", "1 | 1"), "MJH": ("hɑŋku", "1"),
        "HDI": ("ɛko", "1"), "SDI": ("onko", "1"),
        "SNA": ("uŋkin | uŋkuʔko", "1 | 1"), "ORI": ("se mɑnːe", "2"),
    }),
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            source_blank = not form
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": "76",
                "Printed_Page": "71", "Column": "left" if item == 209 else "right",
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if source_blank else "attested",
                "Confidence": "high",
                "Uncertainty": "source explicitly prints '0 no entry'" if source_blank else "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-29",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 2 * 18 == 36
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested") == 36
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 18
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
