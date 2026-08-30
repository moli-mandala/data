#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 197."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_197_197_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with glottal "
    "stops, dental marks, and separately numbered responses rechecked in tight "
    "source-image crops; OCR/PDF text neither supplied nor verified any reading"
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
    "BAI": ("senem, senojɑnɑ", "1"),
    "CHA": ("senoʔme, senlinɑje", "1"),
    "DIG": ("", ""),
    "DUM": ("senoʔme, senlinɑ", "1"),
    "LAD": ("senoʔome, senodʒɑnʌ", "1"),
    "MAD": ("senome, senjɑnɑe", "1"),
    "MOH": ("senom, sendʒenɑ", "1"),
    "MUN": ("d̪olɑ, sindʒenɑ", "1"),
    "POD": ("senme, sendʒɑnɑ", "1"),
    "UDA": ("senoʔme, senoʔjɑnɑ", "1"),
    "MCH": ("sen, senoʔodʒɑnɑ", "1"),
    "MDI": ("sen", "1"),
    "MDH": ("senoʔme, senoʔjɑnɑ", "1"),
    "MJH": ("dʒu, senojenɑ", "1"),
    "HDI": ("dʒu, senojenɑ", "1"),
    "SDI": ("sen | tʃɑlɑkʔ", "1 | 2"),
    "SNA": ("tʃɛlɑinɑj, tʃɛlɑpe", "2"),
    "ORI": ("dʒibɑ", "3"),
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
            "Item": "197", "Gloss": "go!, he went", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "73",
            "Printed_Page": "68", "Column": "right",
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
               if row["Review_Status"] == "attested") == 18
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 9
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
