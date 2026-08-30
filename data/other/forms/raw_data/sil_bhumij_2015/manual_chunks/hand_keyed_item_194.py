#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 194."""

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "items_194_194_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF pages with the "
    "page continuation, dental marks, glottal stop, and vowel qualities "
    "rechecked in tight source-image crops; OCR/PDF text neither supplied "
    "nor verified any reading"
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
    "BAI": ("otɑŋme, otɑŋjɑnɑ", "1"),
    "CHA": ("ud̪ʌɾenme, ud̪ɑedʒɑnɑe", "1"),
    "DIG": ("", ""),
    "DUM": ("ud̪oʔlinɑe, ud̪oʔene", "1"),
    "LAD": ("ɑpiɾme, ɑpiɾdʒɛnɑ", "2"),
    "MAD": ("od̪ome, od̪odʒɑnɑe", "1"),
    "MOH": ("ud̪ow, ud̪owlɛnɑ", "1"),
    "MUN": ("ɑpiɾenme, ɛpiɾdʒɑnɑi", "2"),
    "POD": ("udodʒɑnɑe, udou", "1"),
    "UDA": ("ɑpiɾme, ɑpiɾjɑnɑ", "2"),
    "MCH": ("ɑpiɾ, ɑpiɾdʒɑnɑ", "2"),
    "MDI": ("ɑpiɾ", "2"),
    "MDH": ("ɑpiɾme, ɑpiɾjɑnɑ", "2"),
    "MJH": ("biɾit̪me, ɔtɑŋnenɑj", "1"),
    "HDI": ("ɛpəɾeme, ɑpəɾiɑnɑ", "2"),
    "SDI": ("udɑu | phɑɾkɑo", "1 | 3"),
    "SNA": ("ud̪oʔpe, ud̪ojenɑj", "1"),
    "ORI": ("udutʃi", "1"),
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
        page = "72" if index < 8 else "73"
        printed_page = "67" if index < 8 else "68"
        row = {
            "Item": "194", "Gloss": "fly!, it flew", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": page,
            "Printed_Page": printed_page,
            "Column": "right" if index < 8 else "left",
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
