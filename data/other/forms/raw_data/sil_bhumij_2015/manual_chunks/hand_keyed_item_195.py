#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 195."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_195_195_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with repeated "
    "numbered responses, dental marks, and the printed uncertainty qualifier "
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
    "BAI": ("send̪ogome, senkid̪ɑ", "1"),
    "CHA": ("senoʔme, senlinɑjɑ", "1"),
    "DIG": ("", ""),
    "DUM": ("senoʔme, senlene", "1"),
    "LAD": ("sɛnodʒɑnʌ, dolɑ", "1"),
    "MAD": ("seneme, senkijet̪", "1"),
    "MOH": ("sen, edʒ senlɛnɑ", "1"),
    "MUN": ("d̪olɑŋ, sentʃenɑ | d̪olɑŋ, sentʃenɑ", "1 | 2"),
    "POD": ("senem, sendʒɑnɑ", "1"),
    "UDA": ("senem, ɑeʔ senkid̪ɑ", "1"),
    "MCH": ("tɑhɑlne, honoɾt̪idʒɑnɑ", "4"),
    "MDI": ("", ""),
    "MDH": ("senem, ɑeʔ senkid̪ɑ", "1"),
    "MJH": ("d̪olɑ, senket̪eɾ | d̪olɑ, senket̪eɾ", "1 | 2"),
    "HDI": ("senime, senojinɑ", "1"),
    "SDI": ("dɑɾɑ | tɑɾɑm", "3 | 3"),
    "SNA": ("d̪elɑŋ, tʃelɑwenɑj", "2"),
    "ORI": ("tʃɑlibɑʔ", "5"),
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
        source_blank = code in {"DIG", "MDI"}
        uncertainty = ""
        confidence = "high"
        if source_blank:
            uncertainty = "source explicitly prints '0 no entry'"
        elif code == "LAD":
            uncertainty = "source appends '(?)' after dolɑ; printed form itself is legible"
            confidence = "medium"
        row = {
            "Item": "195", "Gloss": "walk!, he walked", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "73",
            "Printed_Page": "68", "Column": "left",
            "Manual_Transcription": form, "Source_Cognate_Labels": labels,
            "Review_Status": "source_blank" if source_blank else "attested",
            "Confidence": confidence, "Uncertainty": uncertainty,
            "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-29",
            "Reviewer_Declaration": DECLARATION,
        }
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 18
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested") == 19
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 10
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
