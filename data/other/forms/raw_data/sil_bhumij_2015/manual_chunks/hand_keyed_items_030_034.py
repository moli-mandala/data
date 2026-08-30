#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 30--34."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_030_034_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages; "
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
ITEM30_P39 = set(SITES) - {"ORI"}
ITEM33_LEFT = {"BAI", "CHA", "DIG", "DUM"}

DATA = {
    30: ("broom", {
        "BAI": ("dʒonoʔo", "1"), "CHA": ("dʒonoʔ", "1"),
        "DIG": ("dʒɔno", "1"), "DUM": ("dʒonoʔ", "1"),
        "LAD": ("dʒonoʔ", "1"), "MAD": ("dʒɔno", "1"),
        "MOH": ("dʒɔno", "1"), "MUN": ("dʒɔno", "1"),
        "POD": ("dʒonoʔo", "1"), "UDA": ("dʒonoʔ", "1"),
        "MCH": ("dʒɔnoʔ", "1"), "MDI": ("", "0"),
        "MDH": ("dʒonoʔ", "1"), "MJH": ("dʒɔnooʔ", "1"),
        "HDI": ("dʒɔno", "1"), "SDI": ("dʒonok", "1"),
        "SNA": ("dʒəno", "1"), "ORI": ("tʃɑntʃoni", "3"),
    }),
    31: ("mortar", {
        "BAI": ("", "0"), "CHA": ("sʌsɑŋɾɪɖd̪iɾi", "1"),
        "DIG": ("tʃɐki", "2"), "DUM": ("sʌsɑŋɾɪɖd̪iɾi", "1"),
        "LAD": ("sʌsɑŋɾɪɖd̪iɾi", "1"), "MAD": ("ɾiɾit d̪hiɾi", "1"),
        "MOH": ("ɾiʔdiɾi", "1"), "MUN": ("sɐsɑŋɾiʔd̪iɾi", "1"),
        "POD": ("ɾɪɖd̪iɾi", "1"),
        "UDA": ("sʌsɑŋ | sil | sɑsɑŋɾeʔd dhiɾi", "1 | 3 | 1"),
        "MCH": ("sɐsɑŋɾiʔd̪iɾi", "1"), "MDI": ("", "0"),
        "MDH": ("sɑsɑŋɾeʔd dhiɾi", "1"), "MJH": ("tʃɐki", "2"),
        "HDI": ("guɖgu", "4"), "SDI": ("kɑndi | ukhuɾ", "5 | 6"),
        "SNA": ("tʃɐki", "2"), "ORI": ("siɭɔ | kot̪t̪uni", "3 | 7"),
    }),
    32: ("pestle", {
        "BAI": ("gudgu", "1"), "CHA": ("hone d̪iɾi", "3"),
        "DIG": ("tʃɐki", "2"), "DUM": ("hone d̪ɦiɾi", "3"),
        "LAD": ("goɾgi d̪iɾi", "1"), "MAD": ("guɖgu d̪iɾi", "1"),
        "MOH": ("guɖgud̪iɾi", "1"), "MUN": ("ɔne d̪iɾi", "3"),
        "POD": ("guɖugu d̪iɾi", "1"), "UDA": ("gudgi dhiɾi", "1"),
        "MCH": ("guɖgud̪iɾi", "1"), "MDI": ("", "0"),
        "MDH": ("gudgi dhiɾi", "1"), "MJH": ("tʃɐki", "2"),
        "HDI": ("guɖgud̪iɾi", "1"), "SDI": ("tok | dɦusɾɑ", "4 | 5"),
        "SNA": ("tʃɐki", "2"), "ORI": ("pothoɽo", "6"),
    }),
    33: ("hammer", {
        "BAI": ("mɑɾt̪ul", "1"), "CHA": ("mɑɾt̪ul", "1"),
        "DIG": ("koʈɑsi", "2"), "DUM": ("mɑɾt̪ul", "1"),
        "LAD": ("mɑɾt̪uɾ", "1"), "MAD": ("mɐɾt̪ul", "1"),
        "MOH": ("mɑɾt̪ul", "1"), "MUN": ("mɐɾt̪ul", "1"),
        "POD": ("mɑɾt̪ud", "1"), "UDA": ("kutuɾi kotɑs", "2"),
        "MCH": ("koʈɑsi", "2"), "MDI": ("kuʈɑsi | hɑtɑoɽi", "2 | 3"),
        "MDH": ("kutuɾi kotɑs", "2"), "MJH": ("mɑɾt̪ul", "1"),
        "HDI": ("mɐɾt̪ul", "1"), "SDI": ("mɑɾt̪ul | kutɑsi", "1 | 2"),
        "SNA": ("kuʈɛsi", "2"), "ORI": ("hɑt̪uɖi", "3"),
    }),
    34: ("knife", {
        "BAI": ("tʃhuɾi", "1"), "CHA": ("tʃhuɾi", "1"),
        "DIG": ("tʃuɾi", "1"), "DUM": ("tʃhuɾi", "1"),
        "LAD": ("tʃuɾi", "1"), "MAD": ("tʃhuɾi", "1"),
        "MOH": ("tʃuɾi", "1"), "MUN": ("tʃuɾi", "1"),
        "POD": ("tʃhuɾi | puŋki", "1 | 3"), "UDA": ("tʃhuɾi", "1"),
        "MCH": ("kɐt̪u", "2"), "MDI": ("kʌt̪u", "2"),
        "MDH": ("tʃhuɾi", "1"), "MJH": ("kɐt̪u", "2"),
        "HDI": ("kɐt̪u", "2"), "SDI": ("tʃhuɾi", "1"),
        "SNA": ("tʃuɾi", "1"), "ORI": ("tʃhuɾi", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 30 and code in ITEM30_P39:
        return "39", "34", "right"
    if item in {30, 31, 32} or (item == 33 and code in ITEM33_LEFT):
        return "40", "35", "left"
    return "40", "35", "right"


def build_rows():
    rows = []
    blanks = {(30, "MDI"), (31, "BAI"), (31, "MDI"), (32, "MDI")}
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            source_blank = (item, code) in blanks
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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 4
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
