#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 191."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_191_191_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with dental "
    "marks, syllabic nasal, glottal stops, and continuations rechecked at "
    "800 dpi; OCR/PDF text neither supplied nor verified any reading"
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
    "BAI": ("sɑl gɑeme, sɑl gookidɑ", "3"),
    "CHA": ("dʒulime, dʒulijɑe", "1"),
    "DIG": ("", ""),
    "DUM": ("dʒullem, dʒulliɑ", "1"),
    "LAD": ("lot̪ʔn̩me, loleŋɑ", "2"),
    "MAD": ("dʒulkijɑe", "1"),
    "MOH": ("dʒuluwɑ, dʒuʔlenɑ", "1"),
    "MUN": ("dʒult̪enɑ, hɛt̪ɑrdʒɑnɑ", "1"),
    "POD": ("dʒulem, dʒullinɑ", "1"),
    "UDA": ("dʒulkidɑ", "1"),
    "MCH": ("lot̪ɑnɑ, lodʒɑnɑ", "2"),
    "MDI": ("lo | ɑt̪ɑɾ", "2 | 5"),
    "MDH": ("dʒulkidɑ", "1"),
    "MJH": ("dʒulem, dʒullenɑ", "1"),
    "HDI": ("dʒult̪inɑ, dʒulkid̪ɑ", "1"),
    "SDI": ("lo | dʒeɾet", "2 | 4"),
    "SNA": ("dʒulukɑnɑ, dʒulenɑ", "1"),
    "ORI": ("dʒolibɑ", "1"),
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
            "Item": "191", "Gloss": "it burns, it burned", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "72",
            "Printed_Page": "67", "Column": "left",
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
