#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 201."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_201_201_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with the column "
    "continuation, dental marks, palatal nasals, and length marks rechecked in "
    "800-dpi source-image crops; OCR/PDF text neither supplied nor verified any "
    "reading"
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
    "BAI": ("lelim, lelkiɑ", "1"),
    "CHA": ("nelem, nellejɑm", "1"),
    "DIG": ("", ""),
    "DUM": ("nelem, nellejɑ", "1"),
    "LAD": ("lelːime, lelːid̪ɑ", "1"),
    "MAD": ("nelit̪me, nelkijɑt", "1"),
    "MOH": ("nenelɑ, nelɛijɑ", "1"),
    "MUN": ("nelt̪enɑ, ɑinelkid̪ɑi", "1"),
    "POD": ("nelem, nellejɑe", "1"),
    "UDA": ("lelem, lelkid̪ɑ", "1"),
    "MCH": ("lel, lelkid̪ɑi", "1"),
    "MDI": ("lel, nel", "1"),
    "MDH": ("lelem, lelkid̪ɑ", "1"),
    "MJH": ("lɛlɛjeʔ, lɛlket̪e", "1"),
    "HDI": ("nɛlt̪enɑ, nɛlkidɑ", "1"),
    "SDI": ("nel", "1"),
    "SNA": ("ɲɛɲelkɛnɑj, ɲelkijɑj", "1"),
    "ORI": ("dekho", "2"),
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def build_rows():
    assert set(DATA) == set(SITES)
    rows = []
    for index, (code, (language, site, target)) in enumerate(SITES.items()):
        form, labels = DATA[code]
        source_blank = code == "DIG"
        row = {
            "Item": "201", "Gloss": "look!, he saw", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "74",
            "Printed_Page": "69", "Column": "left" if index < 9 else "right",
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
    assert len(rows) == 18
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested") == 17
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 9
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
