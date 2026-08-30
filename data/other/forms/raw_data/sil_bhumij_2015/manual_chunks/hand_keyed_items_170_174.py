#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 170--174."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_170_174_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with continuations, "
    "dental marks, vowel quality, nasalization, and page split rechecked at 800 "
    "dpi; OCR/PDF text neither supplied nor verified any reading"
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
ITEM171_PAGE67 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA"}
BLANKS = {(170, "BAI"), (170, "MDI")}

DATA = {
    170: ("what kind?", {
        "BAI": ("", ""), "CHA": ("tʃilikɑnɑ", "1"),
        "DIG": ("tʃilkɑ", "1"), "DUM": ("tʃilekʌn", "1"),
        "LAD": ("tʃilkɑ", "1"), "MAD": ("tʃimin prɑkɑɾ", "2"),
        "MOH": ("tʃiləkɑ", "1"), "MUN": ("tʃilkɑ", "1"),
        "POD": ("tʃilikɑnɑ | tʃiminprɑkɑɾ", "1 | 2"),
        "UDA": ("kɑnɑlekɑnɑ", "3"), "MCH": ("tʃilkɑ", "1"),
        "MDI": ("", ""), "MDH": ("tʃilikɑ prɑkɑɾ", "1"),
        "MJH": ("tʃilkɑ", "1"), "HDI": ("tʃilke", "1"),
        "SDI": ("tʃekɑn lekɑn", "1"), "SNA": ("tʃɛʔlɛkɑ", "1"),
        "ORI": ("kemit̪i", "4"),
    }),
    171: ("this", {
        "BAI": ("niɑ", "1"), "CHA": ("niɑ", "1"),
        "DIG": ("nijɑ", "1"), "DUM": ("neɑ̃", "1"),
        "LAD": ("nijɑ", "1"), "MAD": ("ine", "1"),
        "MOH": ("nijɑ", "1"), "MUN": ("nijɑ", "1"),
        "POD": ("nijɑ t̪ed", "1"), "UDA": ("niɑtɑ", "1"),
        "MCH": ("nejɑ", "1"), "MDI": ("neɑ", "1"),
        "MDH": ("niɑtɑ", "1"), "MJH": ("nijɑ", "1"),
        "HDI": ("nijɑ", "1"), "SDI": ("niɑ | noɑ", "1 | 2"),
        "SNA": ("nuʋɑ̃", "2"), "ORI": ("eitɑ", "3"),
    }),
    172: ("that", {
        "BAI": ("hɑnɑ", "1"), "CHA": ("hɑnɑ | inɑ", "1 | 1"),
        "DIG": ("hɛnɑ", "1"), "DUM": ("hʌːe", "1"),
        "LAD": ("hinʌ", "1"), "MAD": ("hɑne", "1"),
        "MOH": ("hɑnɑ", "1"), "MUN": ("inɑ", "1"),
        "POD": ("hɑnɑ", "1"), "UDA": ("hɑnɑtɑ", "1"),
        "MCH": ("hɑnɑ", "1"), "MDI": ("enɑ", "1"),
        "MDH": ("hɑnɑtɑ", "1"), "MJH": ("hɑnɑ", "1"),
        "HDI": ("inenɑ", "1"), "SDI": ("onɑ | one", "1 | 1"),
        "SNA": ("ɔnɑ", "1"), "ORI": ("seitɑ", "2"),
    }),
    173: ("these", {
        "BAI": ("niɑko", "1"), "CHA": ("niko", "1"),
        "DIG": ("nɛʔnijɑko", "1"), "DUM": ("nẽʔeko", "1"),
        "LAD": ("nijɑko", "1"), "MAD": ("inekoː", "1"),
        "MOH": ("nijɑ", "1"), "MUN": ("nint̪ɑi", "2"),
        "POD": ("nijɑko", "1"), "UDA": ("niɑko", "1"),
        "MCH": ("nejɑko", "1"), "MDI": ("neɑko | niku", "1 | 1"),
        "MDH": ("niɑko", "1"), "MJH": ("nijɑ", "1"),
        "HDI": ("nijɑ", "1"), "SDI": ("noɑko | noko", "1 | 1"),
        "SNA": ("nuɑko", "1"), "ORI": ("eisɑbu", "3"),
    }),
    174: ("those", {
        "BAI": ("hɑnɑko", "1"), "CHA": ("hɑnɑko", "1"),
        "DIG": ("hɛnɑko", "1"), "DUM": ("hʌːeko", "1"),
        "LAD": ("hɑnʌko", "1"), "MAD": ("hɑneko", "1"),
        "MOH": ("hɛnɑ", "1"), "MUN": ("hɑnt̪ɑi", "2"),
        "POD": ("hʌnɑko", "1"), "UDA": ("hɑnɑko", "1"),
        "MCH": ("hɑnɑko", "1"), "MDI": ("einko", "3"),
        "MDH": ("hɑnɑko", "1"), "MJH": ("hen", "4"),
        "HDI": ("hɛnɑ", "1"), "SDI": ("onko", "1"),
        "SNA": ("hɛnɑko", "1"), "ORI": ("seisɑbu", "5"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 170 or (item == 171 and code in ITEM171_PAGE67):
        return "67", "62", "right"
    if item in {171, 172, 173}:
        return "68", "63", "left"
    return "68", "63", "right"


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
    ) == 94
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows
        if row["Review_Status"] == "attested" and row["Target"] == "yes"
    ) == 51
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
