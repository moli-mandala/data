#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 193."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_193_193_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with dental "
    "marks, glottal stops, and continuations rechecked at 800 dpi; OCR/PDF "
    "text neither supplied nor verified any reading"
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
    "BAI": ("goeʔme, goʔeliɑ", "1"),
    "CHA": ("godʒidʒime, goelidijɑe", "1"),
    "DIG": ("", ""),
    "DUM": ("godʒidʒme, goelidʒiɑ", "1"),
    "LAD": ("goime, goikijʌ", "1"),
    "MAD": ("goet̪kijɑe", "1"),
    "MOH": ("gudʒije, godʒt̪edʒije", "1"),
    "MUN": ("ɾujje, goʔdʒenɑ", "5"),
    "POD": ("godʒidʒme, goeletʔdʒie", "1"),
    "UDA": ("get̪kijɑ", "1"),
    "MCH": ("d̪ɑlie, d̪elkie", "3"),
    "MDI": ("goe", "1"),
    "MDH": ("get̪kijɑ", "1"),
    "MJH": ("gudʒije, gojt̪edʒe", "1"),
    "HDI": ("godʒijɑ, goʔjkɛd̪ejɑj", "1"),
    "SDI": ("gotʃʔ | mɑɾɑo", "1 | 2"),
    "SNA": ("godʒijɑ, goʔjkɛd̪ejɑj", "1"),
    "ORI": ("mɑɾibɑ", "2"),
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
            "Item": "193", "Gloss": "don't kill!, he killed", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "72",
            "Printed_Page": "67", "Column": "right",
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
