#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 40--44."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_040_044_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with difficult "
    "glyphs rechecked at 800/1600 dpi; text scaffold not accepted without "
    "cell visual match"
)
SITES = {
    "BAI": ("Bhumij", "Baigodia", True),
    "CHA": ("Bhumij", "Champi", True),
    "DIG": ("Bhumij", "Dighinuasahi", True),
    "DUM": ("Bhumij", "Dumadie", True),
    "LAD": ("Bhumij", "Ladhiramsai", True),
    "MAD": ("Bhumij", "Madhupur", True),
    "MOH": ("Bhumij", "Mohuldiha", True),
    "MUN": ("Bhumij", "Munduy", True),
    "POD": ("Bhumij", "Podadiha", True),
    "UDA": ("Bhumij/Mundari", "Udala", True),
    "MCH": ("Mundari", "Chalagi", False),
    "MDI": ("Mundari", "Dictionary", False),
    "MDH": ("Mundari", "Dhungarisai", False),
    "MJH": ("Mundari", "Jharmunda", False),
    "HDI": ("Ho", "Dillisore", False),
    "SDI": ("Santali", "Dictionary", False),
    "SNA": ("Santali", "Nayarangamotia", False),
    "ORI": ("Oriya", "Cuttack", False),
}
ITEM40_P41 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI",
}
ITEM43_LEFT = {"BAI", "CHA", "DIG"}
BLANKS = {(41, "HDI"), (42, "HDI"), (43, "BAI")}

DATA = {
    40: ("ring", {
        "BAI": ("mud̪ɑm", "1"), "CHA": ("mud̪ʌm", "1"),
        "DIG": ("pɔlɑ", "3"), "DUM": ("mud̪em", "1"),
        "LAD": ("muld̪ʌm", "1"), "MAD": ("mud̪em", "1"),
        "MOH": ("mud̪ɑm", "1"), "MUN": ("mud̪ɑm", "1"),
        "POD": ("muɳd̪em", "1"), "UDA": ("polɑ", "3"),
        "MCH": ("mud̪ɑm", "1"), "MDI": ("polɑ", "3"),
        "MDH": ("polɑ", "3"), "MJH": ("mud̪ɑm", "1"),
        "HDI": ("pɔlɑ", "3"), "SDI": ("mundɑm", "1"),
        "SNA": ("mud̪em", "1"), "ORI": ("mud̪i", "2"),
    }),
    41: ("sun", {
        "BAI": ("siŋgi", "1"), "CHA": ("siŋgi", "1"),
        "DIG": ("siŋki", "1"), "DUM": ("siŋgi", "1"),
        "LAD": ("siŋgi", "1"), "MAD": ("siŋgi", "1"),
        "MOH": ("siŋki", "1"), "MUN": ("siŋki", "1"),
        "POD": ("siŋgi", "1"), "UDA": ("siŋgi", "1"),
        "MCH": ("siŋgi", "1"), "MDI": ("siŋgi", "1"),
        "MDH": ("siŋgi", "1"), "MJH": ("siŋki", "1"),
        "HDI": ("", ""),
        "SDI": ("sin tʃɑndo | belɑ", "2 | 4"),
        "SNA": ("tʃɑnt̪o", "2"), "ORI": ("sudʒo", "3"),
    }),
    42: ("moon", {
        "BAI": ("tʃɑndu", "1"), "CHA": ("tʃʌdup", "1"),
        "DIG": ("tʃɑnt̪u", "1"), "DUM": ("tʃɑdub", "1"),
        "LAD": ("tʃɑnd̪u", "1"), "MAD": ("tʃɑndu", "1"),
        "MOH": ("tʃɑnt̪u", "1"), "MUN": ("tʃɑnt̪u", "1"),
        "POD": ("tʃɑndupʔ", "1"), "UDA": ("tʃɑnd̪u", "1"),
        "MCH": ("tʃɑnt̪uuʔ", "1"), "MDI": ("tʃɑndu", "1"),
        "MDH": ("tʃɑnd̪u", "1"), "MJH": ("tʃɑnt̪u", "1"),
        "HDI": ("", ""), "SDI": ("nindɑ tʃɑndo", "1"),
        "SNA": ("tʃɑnt̪o", "1"), "ORI": ("dʒɑnhɑ", "2"),
    }),
    43: ("sky", {
        "BAI": ("", ""), "CHA": ("siɾmɑ", "1"),
        "DIG": ("siɾmɑ", "1"), "DUM": ("siɾmɑ", "1"),
        "LAD": ("siɾmɑ", "1"), "MAD": ("siɾmɑ", "1"),
        "MOH": ("siɾmɑ", "1"), "MUN": ("siɾmɑ", "1"),
        "POD": ("siɾmʌ", "1"), "UDA": ("siɾmɑ", "1"),
        "MCH": ("siɾmɑ", "1"), "MDI": ("siɾmɑ", "1"),
        "MDH": ("siɾmɑ", "1"), "MJH": ("siɾmɑ", "1"),
        "HDI": ("siɾme", "1"), "SDI": ("seɾmʌ", "1"),
        "SNA": ("seɾmɑ", "1"), "ORI": ("ɑkɑsɑu", "3"),
    }),
    44: ("star", {
        "BAI": ("ipil", "1"), "CHA": ("ipil", "1"),
        "DIG": ("ipil", "1"), "DUM": ("ipil", "1"),
        "LAD": ("ipil", "1"), "MAD": ("ipil", "1"),
        "MOH": ("ipil", "1"), "MUN": ("ipil", "1"),
        "POD": ("ipil", "1"), "UDA": ("ipil", "1"),
        "MCH": ("ipil", "1"), "MDI": ("ipil", "1"),
        "MDH": ("ipil", "1"), "MJH": ("ipil", "1"),
        "HDI": ("ipil", "1"), "SDI": ("ipil", "1"),
        "SNA": ("ipil", "1"), "ORI": ("t̪ɑɾɑ", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 40 and code in ITEM40_P41:
        return "41", "36", "right"
    if item in {40, 41, 42} or (item == 43 and code in ITEM43_LEFT):
        return "42", "37", "left"
    return "42", "37", "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            source_blank = (item, code) in BLANKS
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if source_blank else "attested",
                "Confidence": "high",
                "Uncertainty": "source explicitly prints '0 no entry'" if source_blank else "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 5 * 18 == 90
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 3
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 88
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
