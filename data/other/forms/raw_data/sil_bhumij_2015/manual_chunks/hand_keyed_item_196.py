#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 196."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_196_196_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with the "
    "column continuation, dental marks, and separately numbered responses "
    "rechecked in tight source-image crops; OCR/PDF text neither supplied nor "
    "verified any reading"
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
    "BAI": ("dɦɑud̪em, dɦɑukid̪ɑ", "2"),
    "CHA": ("niɾeme, niɾlejɑe", "1"),
    "DIG": ("", ""),
    "DUM": ("niɾeme, niɾlinɑ", "1"),
    "LAD": ("niɾʌme, niɾled̪ɑ", "1"),
    "MAD": ("niɾeme, niɾkijet", "1"),
    "MOH": ("niɾme, niɾkijɑ", "1"),
    "MUN": ("d̪iɾime, niɾkid̪ɑi", "1"),
    "POD": ("niɾem, niɾlidɑ", "1"),
    "UDA": ("dɦɑodum, dɦɑodkidɑ", "2"),
    "MCH": ("niɾ, niɾdʒɑnɑ", "1"),
    "MDI": ("niɾ | dɑuɾi", "1 | 2"),
    "MDH": ("dɦɑodum, dɦɑodkidɑ", "2"),
    "MJH": ("niɾkɛd̪ɑi, niɾəme", "1"),
    "HDI": ("niɾme, niɾkid̪ɑ", "1"),
    "SDI": ("niɾ | dɑɾ", "1 | 2"),
    "SNA": ("d̪ɛd̪pee, d̪ɛd̪kijɑ", "3"),
    "ORI": ("douuibɑ", "2"),
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
            "Item": "196", "Gloss": "run!, he ran", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "73",
            "Printed_Page": "68", "Column": "left" if index < 15 else "right",
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
               if row["Review_Status"] == "attested") == 19
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 9
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
