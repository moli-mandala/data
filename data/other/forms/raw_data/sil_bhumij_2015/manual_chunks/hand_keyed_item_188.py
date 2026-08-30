#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for item 188."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_188_188_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of the 400-dpi rendered PDF page with repeated "
    "dental marks, glottal stops, and unlabeled continuations rechecked at "
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
    "BAI": ("", ""),
    "CHA": ("git̪ikene, git̪iʔne", "1"),
    "DIG": ("", ""),
    "DUM": ("git̪iʔt̪me, git̪iʔt̪linɑ", "1"),
    "LAD": ("git̪ime, git̪iɑkʌn t̪aikinʌ", "1"),
    "MAD": ("git̪it̪me, git̪itdʒɑnɑe", "1"),
    "MOH": ("dʒiɑwo, dʒiolenɑ", "3"),
    "MUN": ("git̪iʔme, git̪iʔdʒɛnɑ", "1"),
    "POD": ("git̪itme, git̪itdʒɑnɑe | git̪iʔmeʔ, git̪id̪linɑ", "1 | 1"),
    "UDA": ("gidime, gidijɑnɑ", "1"),
    "MCH": ("bɑt̪in, bɑt̪indʒene", "2"),
    "MDI": ("git̪i | bɑt̪in | buɾum", "1 | 2 | 5"),
    "MDH": ("gidime, gidijɑnɑ", "1"),
    "MJH": ("d̪ugme, git̪ignenɑj", "1"),
    "HDI": ("git̪ime, git̪ijene", "1"),
    "SDI": ("git̪itʃ", "1"),
    "SNA": ("gud̪t̪o hɛnt̪ɑd̪ope, gud̪t̪owenɑj", "4"),
    "ORI": ("poɾigolɑ", "6"),
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
        source_blank = code in {"BAI", "DIG"}
        row = {
            "Item": "188", "Gloss": "lie down!, he lay down", "Site_Code": code,
            "Language_Label": language, "Site_Name": site,
            "Target": "yes" if target else "no", "PDF_Page": "71",
            "Printed_Page": "66", "Column": "right",
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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
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
