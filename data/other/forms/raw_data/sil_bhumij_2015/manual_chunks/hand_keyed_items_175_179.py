#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 175--179."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_175_179_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with continuation "
    "lines, dental marks, vowel quality, glottals, and the page split rechecked "
    "at 800 dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM176_PAGE68 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH"}
ITEM178_PAGE69_LEFT = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH", "MJH", "HDI", "SDI",
}
BLANKS = {(178, "MDI")}

DATA = {
    175: ("same", {
        "BAI": ("sʌmɑn", "1"), "CHA": ("sʌmɑn", "1"),
        "DIG": ("sɛmɑn | sɔmɑn", "1 | 1"), "DUM": ("modʔgeɑ", "2"),
        "LAD": ("bɑɾɑ bʌɾi", "6"), "MAD": ("sɛmɑn", "1"),
        "MOH": ("sɔmɑn", "1"), "MUN": ("motgejɑ", "2"),
        "POD": ("sʌmɑn | motgiɑ", "1 | 2"), "UDA": ("sɛmɑn", "1"),
        "MCH": ("enlikɑ", "5"), "MDI": ("midge", "2"),
        "MDH": ("sɛmɑn", "1"), "MJH": ("mit̪gi", "2"),
        "HDI": ("sɔmɑn | ɔnkɑ", "1 | 3"),
        "SDI": ("somɑn | inɑ", "1 | 4"), "SNA": ("ɔnkɑ", "3"),
        "ORI": ("səmɑn", "1"),
    }),
    176: ("different", {
        "BAI": ("biŋgɑ biŋgɑ", "1"), "CHA": ("benɑ benɑ", "1"),
        "DIG": ("binkɑ", "1"), "DUM": ("vegʌɾ vegʌɾ", "2"),
        "LAD": ("biŋgʌ", "1"), "MAD": ("bɦinɑ", "1"),
        "MOH": ("bɛnɑbini", "1"), "MUN": ("binɑ benɑ", "1"),
        "POD": ("bɦenɑ bɦenɑ | begɑɾ begɑɾ", "1 | 2"),
        "UDA": ("bɦiŋgɑ bɦiŋgɑ", "1"),
        "MCH": ("et̪ɑ et̪ɑʔɑ", "3"),
        "MDI": ("et̪ɑ | kilimili", "3 | 5"),
        "MDH": ("bɦiŋgɑ bɦiŋgɑ", "1"),
        "MJH": ("ɛt̪eɑʔ ɛt̪eɑʔ", "3"),
        "HDI": ("ɔlgɑ ɔlgɑ", "4"), "SDI": ("dʒudɑ", "6"),
        "SNA": ("begɑɾ", "2"),
        "ORI": ("bɦino bɦino | əlegɑ", "1 | 4"),
    }),
    177: ("whole", {
        "BAI": ("gotɑ", "1"), "CHA": ("gotɑ", "1"),
        "DIG": ("dʒɔt̪o", "2"), "DUM": ("best̪iɑ", "3"),
        "LAD": ("gotɑ", "1"), "MAD": ("bes", "3"),
        "MOH": ("dʒɔt̪o", "2"), "MUN": ("dʒɔt̪o", "2"),
        "POD": ("gotɑ", "1"), "UDA": ("gotɑ", "1"),
        "MCH": ("gɔt̪ɑ | soben", "1 | 4"), "MDI": ("gotɑ", "1"),
        "MDH": ("buginɑ", "5"), "MJH": ("gɔtɑ", "1"),
        "HDI": ("gɔtɑ", "1"), "SDI": ("gotɑɾ", "1"),
        "SNA": ("dʒɔt̪o", "2"), "ORI": ("puɾɑ", "6"),
    }),
    178: ("broken", {
        "BAI": ("ɾɑpudʔ", "1"), "CHA": ("ɾɑpud", "1"),
        "DIG": ("ɾɑpuʔ", "1"), "DUM": ("ɾɑpud", "1"),
        "LAD": ("ɾɑpɑ̃tʔn̩", "1"), "MAD": ("ɾɑpud", "1"),
        "MOH": ("ɾɑpuʔ", "1"), "MUN": ("ɾɑpuʔ", "1"),
        "POD": ("ɾʌpud", "1"), "UDA": ("ɾɑpud", "1"),
        "MCH": ("ɾɑpud̪", "1"), "MDI": ("", ""),
        "MDH": ("ɾɑpud", "1"), "MJH": ("ɾɑpuʔ", "1"),
        "HDI": ("ɾɑpuʔd", "1"),
        "SDI": ("bɦɑŋgɑ | kɑtʃɑ | t̪ut̪ɑ", "2 | 3 | 4"),
        "SNA": ("ɾɑpu", "1"), "ORI": ("bɑŋgilɑ", "2"),
    }),
    179: ("few", {
        "BAI": ("hudiʔ", "1"), "CHA": ("ɑŋgɑ", "3"),
        "DIG": ("kud̪i", "4"), "DUM": ("hudiŋ", "1"),
        "LAD": ("uɾiŋ", "1"), "MAD": ("ɑŋgɑno", "3"),
        "MOH": ("kom", "5"), "MUN": ("heŋkɑ", "3"),
        "POD": ("hʌŋgɑ", "3"), "UDA": ("hud̪i", "1"),
        "MCH": ("ket̪i", "4"), "MDI": ("hud̪uɾiŋ", "1"),
        "MDH": ("hud̪i", "1"), "MJH": ("ket̪i", "4"),
        "HDI": ("dʒɔkɑ", "6"),
        "SDI": ("thoɾɑ gɑn | ekɑ | dukɑ", "7 | 8 | 9"),
        "SNA": ("ket̪i", "4"), "ORI": ("kom", "5"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 175 or (item == 176 and code in ITEM176_PAGE68):
        return "68", "63", "right"
    if item in {176, 177} or (item == 178 and code in ITEM178_PAGE69_LEFT):
        return "69", "64", "left"
    return "69", "64", "right"


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
    ) == 101
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows
        if row["Review_Status"] == "attested" and row["Target"] == "yes"
    ) == 53
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
