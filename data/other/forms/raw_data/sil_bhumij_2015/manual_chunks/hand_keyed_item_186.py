#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 186."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_186_186_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with repeated "
    "dental marks, glottal stops, and continuation lines rechecked at 800 dpi; "
    "OCR/PDF text neither supplied nor verified any reading"
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
    "BAI": ("t̪it̪ɑŋt̪aiken", "1"),
    "CHA": ("t̪it̪ɑŋt̪ɑdʒi t̪aikenɑ", "1"),
    "DIG": ("", ""),
    "DUM": ("t̪it̪ɑŋt̪e t̪aikenɑ", "1"),
    "LAD": ("t̪ɛt̪ɑŋt̪aikinʌ", "1"),
    "MAD": ("t̪it̪ɑŋt̪e t̪aikenɑ", "1"),
    "MOH": ("t̪ɛt̪ɑŋt̪edʒijɑ, hɔlɑ t̪ɛt̪ɑŋt̪edʒijɑ", "1"),
    "MUN": ("t̪ɛt̪ɑŋt̪ɑdʒijɑ, t̪ɛt̪ɑŋt̪aikenɑ", "1"),
    "POD": ("t̪it̪ɑŋt̪ɑdʒie", "1"),
    "UDA": ("t̪it̪ɑŋt̪aikenɑ", "1"),
    "MCH": ("t̪ɛt̪ɑŋdʒɑʔɑje, t̪ɛt̪ɑŋliʔɑ", "1"),
    "MDI": ("t̪ɛt̪ɑŋ", "1"),
    "MDH": ("t̪it̪ɑŋt̪aikenɑ", "1"),
    "MJH": ("t̪ɛt̪ɑŋɔt̪ɛne, t̪ɛt̪ɑŋɔt̪ɛine", "1"),
    "HDI": ("t̪ɛt̪ɑŋit̪ɛnɑ, t̪ɛt̪ɑŋlije", "1"),
    "SDI": ("t̪ɛt̪ɑn", "1"),
    "SNA": ("t̪ɛt̪ɑŋikɑnɑ, t̪ɛt̪ɑŋlid̪ijɑ", "1"),
    "ORI": ("soso həlɑʔ", "2"),
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
    for code, (language, site, target) in SITES.items():
        form, labels = DATA[code]
        source_blank = code == "DIG"
        row = {
            "Item": "186", "Gloss": "he is, he was thirsty", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "71",
            "Printed_Page": "66", "Column": "left",
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
