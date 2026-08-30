#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 50--54."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_050_054_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with page-break "
    "continuations enlarged separately; text scaffold not accepted without "
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
ITEM50_P43 = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH",
}
ITEM53_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD"}
BLANKS = {(50, "BAI")}

DATA = {
    50: ("rainbow", {
        "BAI": ("", ""), "CHA": ("ɾɑm dɦʌnus", "5"),
        "DIG": ("kɑlɑnd̪uki", "6"), "DUM": ("ɾohod̪biŋ", "1"),
        "LAD": ("bɑnd̪ɛlɛlɛʔ", "2"), "MAD": ("lite", "3"),
        "MOH": ("ɾod̪biŋ", "1"), "MUN": ("nud̪ubiŋ", "4"),
        "POD": ("luɳd̪ubiŋ", "4"), "UDA": ("ɾɑm diluɑn", "5"),
        "MCH": ("bɑnd̪ɑsike | lɔhɔɾbiŋ", "2 | 4"),
        "MDI": ("bɑnd̪ɑlele", "2"), "MDH": ("tʃɑndil", "7"),
        "MJH": ("lɔhɔɾbiŋ", "4"), "HDI": ("ɾulbiŋoŋ", "1"),
        "SDI": ("litɑ ɑʔk", "3"), "SNA": ("lite", "3"),
        "ORI": ("ind̪ɾod̪ɑnɑsə", "8"),
    }),
    51: ("wind", {
        "BAI": ("hʌjʌ", "1"), "CHA": ("hʌjo", "1"),
        "DIG": ("hojo", "1"), "DUM": ("hʌjʌ", "1"),
        "LAD": ("hojo", "1"), "MAD": ("hojo", "1"),
        "MOH": ("hojo", "1"), "MUN": ("hojo", "1"),
        "POD": ("hojo", "1"), "UDA": ("hojɑ", "1"),
        "MCH": ("hojo", "1"), "MDI": ("hojo", "1"),
        "MDH": ("hojɑ", "1"), "MJH": ("hojo", "1"),
        "HDI": ("ɔjo", "1"), "SDI": ("hoe", "1"),
        "SNA": ("hoi", "1"), "ORI": ("dʒhoɾɑkɑ", "2"),
    }),
    52: ("stone", {
        "BAI": ("d̪iɾi", "1"), "CHA": ("d̪iɾi", "1"),
        "DIG": ("d̪iɾi", "1"), "DUM": ("d̪hiɾi", "1"),
        "LAD": ("d̪iɾi", "1"), "MAD": ("d̪hiɾi", "1"),
        "MOH": ("d̪iɾi", "1"), "MUN": ("d̪iɾi", "1"),
        "POD": ("d̪iɾi", "1"), "UDA": ("d̪hiɾi", "1"),
        "MCH": ("d̪iɾi", "1"), "MDI": ("d̪iɾi", "1"),
        "MDH": ("d̪hiɾi", "1"), "MJH": ("d̪iɾi", "1"),
        "HDI": ("d̪iɾi", "1"), "SDI": ("d̪hiɾi", "1"),
        "SNA": ("d̪iɾi", "1"), "ORI": ("pət̪həɾə", "2"),
    }),
    53: ("path", {
        "BAI": ("hoɾɑ", "1"), "CHA": ("hoɾeŋ", "1"),
        "DIG": ("hoɾɑ", "1"), "DUM": ("hoɾeŋ", "1"),
        "LAD": ("hoɾɑ", "1"), "MAD": ("hoɾeŋ", "1"),
        "MOH": ("hoɾeŋ", "1"), "MUN": ("hoɾeŋ", "1"),
        "POD": ("hoɾen", "1"), "UDA": ("hoɾɑ", "1"),
        "MCH": ("hɔɾɑ", "1"), "MDI": ("hoɾɑ", "1"),
        "MDH": ("hoɾɑ", "1"), "MJH": ("hɔɾɑ", "1"),
        "HDI": ("hɔɾɑ", "1"), "SDI": ("hoɾ | sesɑ", "1 | 2"),
        "SNA": ("hoɾ | sesɑ", "1 | 2"),
        "ORI": ("ɾɑst̪ɾɑ | bɑto", "3 | 4"),
    }),
    54: ("sand", {
        "BAI": ("git̪il", "1"), "CHA": ("git̪il", "1"),
        "DIG": ("git̪il", "1"), "DUM": ("git̪il", "1"),
        "LAD": ("git̪il", "1"), "MAD": ("git̪il", "1"),
        "MOH": ("git̪il", "1"), "MUN": ("git̪il", "1"),
        "POD": ("git̪il", "1"), "UDA": ("git̪il", "1"),
        "MCH": ("git̪il", "1"), "MDI": ("git̪il", "1"),
        "MDH": ("git̪il", "1"), "MJH": ("git̪il", "1"),
        "HDI": ("git̪il", "1"), "SDI": ("git̪il", "1"),
        "SNA": ("bɑli", "2"), "ORI": ("bɑɭi", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 50 and code in ITEM50_P43:
        return "43", "38", "right"
    if item in {50, 51, 52} or (item == 53 and code in ITEM53_LEFT):
        return "44", "39", "left"
    return "44", "39", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 93
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
