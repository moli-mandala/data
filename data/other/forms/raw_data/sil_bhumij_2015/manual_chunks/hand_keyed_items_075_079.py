#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 75--79."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_075_079_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with rhotics, "
    "nasals, vowel quality, and page/column breaks rechecked at 800 dpi; "
    "text scaffold not accepted without cell visual match"
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
ITEM76_PAGE48 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH",
}
ITEM79_LEFT = {"BAI", "CHA", "DIG"}
BLANKS = {(79, "MDI"), (79, "SDI")}

DATA = {
    75: ("chili", {
        "BAI": ("muɾtʃi", "1"), "CHA": ("muɾtʃi", "1"),
        "DIG": ("muɾtʃi", "1"), "DUM": ("muɾtʃi", "1"),
        "LAD": ("muɾtʃi", "1"), "MAD": ("mɛɾtʃi", "1"),
        "MOH": ("muɾtʃi", "1"), "MUN": ("muɾtʃi", "1"),
        "POD": ("muɾtʃi", "1"), "UDA": ("moɾtʃi", "1"),
        "MCH": ("mɛɾtʃi", "1"), "MDI": ("mɑɾtʃi", "1"),
        "MDH": ("moɾtʃi", "1"), "MJH": ("mɛɾtʃi", "1"),
        "HDI": ("mɛɾtʃi", "1"), "SDI": ("mɑɾitʃ", "1"),
        "SNA": ("mɛɾitʃ", "1"), "ORI": ("məɾitʃə", "1"),
    }),
    76: ("turmeric", {
        "BAI": ("sɑsɑŋ", "1"), "CHA": ("sʌsɑŋ", "1"),
        "DIG": ("sɛsɑŋ", "1"), "DUM": ("sʌsʌŋ", "1"),
        "LAD": ("sʌsɑŋ", "1"), "MAD": ("sɛsɛŋ", "1"),
        "MOH": ("sɛsɑŋ", "1"), "MUN": ("sɛsɑn", "1"),
        "POD": ("sʌsɑŋ", "1"), "UDA": ("sɛsɑŋ", "1"),
        "MCH": ("sɛsɑŋ", "1"), "MDI": ("sɑsɑŋ", "1"),
        "MDH": ("sɛsɑŋ", "1"), "MJH": ("sɛsɑŋ", "1"),
        "HDI": ("sɛsɑŋ", "1"), "SDI": ("sɑsɑŋ", "1"),
        "SNA": ("sɛsɑn", "1"), "ORI": ("holidi", "2"),
    }),
    77: ("garlic", {
        "BAI": ("ɾɑsuɳi", "1"), "CHA": ("ɾɑsuɳi", "1"),
        "DIG": ("ɾɑsuɳi", "1"), "DUM": ("ɾʌsuɲĩ", "1"),
        "LAD": ("ɾɑsuɳi", "1"), "MAD": ("ɾɑsuɳi", "1"),
        "MOH": ("ɾɑsuɳi", "1"), "MUN": ("ɾɑsuɳi", "1"),
        "POD": ("ɾʌsuni", "1"), "UDA": ("ɾɑsuɳi", "1"),
        "MCH": ("ɾɑsuɳi", "1"), "MDI": ("ɾɑsuni", "1"),
        "MDH": ("ɾɑsuɳi", "1"), "MJH": ("ɾɑsuɳi", "1"),
        "HDI": ("ɾɛsuiŋ", "1"), "SDI": ("ɾɑsun", "1"),
        "SNA": ("ɾesunɭ", "1"), "ORI": ("ɾəsunə", "1"),
    }),
    78: ("onion", {
        "BAI": ("piɑdʒu", "1"), "CHA": ("piɑdʒi", "1"),
        "DIG": ("piɑdʒi", "1"), "DUM": ("piɑdʒ", "1"),
        "LAD": ("pjɑdʒi", "1"), "MAD": ("piɑdʒ", "1"),
        "MOH": ("piɑdʒi", "1"), "MUN": ("piɑdʒi", "1"),
        "POD": ("pijɑdʒi", "1"), "UDA": ("piɑdʒi", "1"),
        "MCH": ("piɑdʒu", "1"), "MDI": ("peɑdʒu", "1"),
        "MDH": ("piɑdʒi", "1"), "MJH": ("piɑdʒ", "1"),
        "HDI": ("piɑdʒi", "1"), "SDI": ("peɑdʒ", "1"),
        "SNA": ("piɑdʒ", "1"), "ORI": ("piɑdʒo", "1"),
    }),
    79: ("cauliflower", {
        "BAI": ("bɑ kobi", "1"), "CHA": ("bɑhɑ kobi", "1"),
        "DIG": ("bɑkobi", "1"), "DUM": ("bɑhɑ kobi", "1"),
        "LAD": ("bɑ kobi", "1"), "MAD": ("bɑhu kobi", "1"),
        "MOH": ("bɑhɑkobi", "1"), "MUN": ("bɑkobi", "1"),
        "POD": ("bo kobi", "1"), "UDA": ("bɑ kobi", "1"),
        "MCH": ("bɑhɑkobi", "1"), "MDI": ("", ""),
        "MDH": ("bɑ kobi", "1"), "MJH": ("bɑhɑkobi", "1"),
        "HDI": ("bɑkobi", "1"), "SDI": ("", ""),
        "SNA": ("bɑhɑkobi", "1"), "ORI": ("phul kobi", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 75 or (item == 76 and code in ITEM76_PAGE48):
        return "48", "43", "right"
    if item in {76, 77, 78} or (item == 79 and code in ITEM79_LEFT):
        return "49", "44", "left"
    return "49", "44", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
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
