#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 192."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_192_192_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with glottal "
    "stops, nasalization, continuations, and the column break rechecked at "
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
    "BAI": ("godʒʌʔme, goejɑnɑ", "1"),
    "CHA": ("gudʒuʔme, godʒʔdʒɑnɑe", "1"),
    "DIG": ("", ""),
    "DUM": ("godʒoʔme, goeʔlinɑ", "1"),
    "LAD": ("godʒome, goedʒɑnʌ", "1"),
    "MAD": ("godʒuʔme, godʒʔdʒɑnɑ", "1"),
    "MOH": ("godʒuwɑ, godʒɑnɑ", "1"),
    "MUN": ("godʒoɑ, goʔdʒenɑ", "1"),
    "POD": ("godʒuʔme, godʒʔdʒɑnɑ", "1"),
    "UDA": ("goejɑnɑ", "1"),
    "MCH": ("goʔedʒenɑ, goʔedʒɑnɑ", "1"),
    "MDI": ("godʒo", "1"),
    "MDH": ("goejɑnɑ", "1"),
    "MJH": ("godʒijɑ, gojnenɑj", "1"),
    "HDI": ("godʒo, goʔjenɑ", "1"),
    "SDI": ("gudʒuk, gotʃʔ", "1"),
    "SNA": ("gudʒuwɑ, godʒinɑj", "1"),
    "ORI": ("moɾonõ", "2"),
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
        right = code not in {"BAI", "CHA"}
        row = {
            "Item": "192", "Gloss": "don't die!, he died", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "72",
            "Printed_Page": "67", "Column": "right" if right else "left",
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
