#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 199."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_199_199_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF pages with the page "
    "continuation, palatal nasal, dental marks, and separately numbered "
    "responses rechecked in 800-dpi source-image crops; OCR/PDF text neither "
    "supplied nor verified any reading"
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
    "BAI": ("dʒɑɾem, dʒɑgɑɾlɑ", "3"),
    "CHA": ("kɑdʒilijɑe | menkejʌe", "1 | 2"),
    "DIG": ("", ""),
    "DUM": ("kɑdʒidʒme, kɑdʒiluiɑ", "1"),
    "LAD": ("kɑdʒiʔme, kɑdʒilɛd̪ɑ", "1"),
    "MAD": ("kɛdʒidʒme, kɛdʒikijɑ", "1"),
    "MOH": ("kɛdʒiʔɲe, kɛdʒilɛʔjɑ", "1"),
    "MUN": ("kɛdʒidʒɲe, kɑdʒikedɑ", "1"),
    "POD": ("kɑdʒidʒme, kɑdʒikie", "1"),
    "UDA": ("menem, menkid̪ɑ", "2"),
    "MCH": ("dʒɑgɑɾ, dʒɑgɑɾ kid̪ɑ", "3"),
    "MDI": ("kɑdʒi | men", "1 | 2"),
    "MDH": ("menem, menkid̪ɑ", "2"),
    "MJH": ("kɛdʒime, kɛdʒikidɑ", "1"),
    "HDI": ("kɛdʒime, kɛdʒikidɑ", "1"),
    "SDI": ("men | ɾoɾ", "2 | 5"),
    "SNA": ("menme, menkijɑ", "2"),
    "ORI": ("kɔhilɑ, kuhɑ", "6"),
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
        first_page = index < 3
        row = {
            "Item": "199", "Gloss": "speak!, he spoke", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no",
            "PDF_Page": "73" if first_page else "74",
            "Printed_Page": "68" if first_page else "69",
            "Column": "right" if first_page else "left",
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
               if row["Review_Status"] == "attested") == 20
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows
               if row["Review_Status"] == "attested" and row["Target"] == "yes") == 10
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
