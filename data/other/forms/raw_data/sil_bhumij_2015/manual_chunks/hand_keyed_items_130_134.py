#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 130--134."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_130_134_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental and "
    "retroflex marks, aspiration, vowel quality, continuations, and the "
    "page/column breaks rechecked at 800 dpi; OCR/PDF text neither supplied "
    "nor verified any reading"
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
ITEM131_PAGE59 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD"}

DATA = {
    130: ("good", {
        "BAI": ("bes", "1"), "CHA": ("bes", "1"),
        "DIG": ("bɔgi", "2"), "DUM": ("bes", "1"),
        "LAD": ("bogin", "2"), "MAD": ("bes", "1"),
        "MOH": ("bes", "1"), "MUN": ("bes", "1"),
        "POD": ("bes", "1"), "UDA": ("bes | bigi", "1 | 2"),
        "MCH": ("bes | bugin", "1 | 2"), "MDI": ("bugin", "2"),
        "MDH": ("bes | bugin", "1 | 2"), "MJH": ("bes", "1"),
        "HDI": ("bugin", "2"), "SDI": ("bes | boge", "1 | 2"),
        "SNA": ("bes", "1"), "ORI": ("bɦolo", "3"),
    }),
    131: ("bad", {
        "BAI": ("khɑɾɑp", "1"), "CHA": ("khʌɾɑp", "1"),
        "DIG": ("dʒuɖɑ", "4"), "DUM": ("khʌɾʌp", "1"),
        "LAD": ("khʌɾɑb", "1"), "MAD": ("khɛɾɑp", "1"),
        "MOH": ("khɛɾɑb", "1"), "MUN": ("khɛɾɑb", "1"),
        "POD": ("koɾɑb", "1"), "UDA": ("khɑɾɑp", "1"),
        "MCH": ("eʔʈkɑ", "2"), "MDI": ("eʈkɑn", "2"),
        "MDH": ("khɑɾɑp", "1"), "MJH": ("siʈɾu", "5"),
        "HDI": ("kɛɾɑb", "1"), "SDI": ("kɑɾɑp | bɑɾitʃʔ", "1 | 3"),
        "SNA": ("khɛɾɑb", "1"), "ORI": ("kɑɾɑpo", "1"),
    }),
    132: ("wet", {
        "BAI": ("ɑɖɑʔt", "2"), "CHA": ("lum", "1"),
        "DIG": ("lum", "1"), "DUM": ("lum", "1"),
        "LAD": ("lum", "1"), "MAD": ("oɖɑɖ", "2"),
        "MOH": ("ɔɖɑʔ", "2"), "MUN": ("lum", "1"),
        "POD": ("lumdʒ", "1"), "UDA": ("oɖɑɖ | lejeɾ", "2 | 4"),
        "MCH": ("lum", "1"), "MDI": ("lum", "1"),
        "MDH": ("ɑɖɑt", "2"), "MJH": ("lum", "1"),
        "HDI": ("lum", "1"), "SDI": ("odɑ | lohot", "2 | 3"),
        "SNA": ("ɔɖɑ", "2"), "ORI": ("oɖɑ", "2"),
    }),
    133: ("dry", {
        "BAI": ("ɾoʔlo", "1"), "CHA": ("ɾoɖo", "1"),
        "DIG": ("ɾoɭo", "1"), "DUM": ("ɾʌhoɖ", "2"),
        "LAD": ("ɾoɾo", "1"), "MAD": ("ɾɛheɖ", "2"),
        "MOH": ("ɾoɖ", "1"), "MUN": ("ɾoɖ", "1"),
        "POD": ("ɾoɖu", "1"), "UDA": ("ɾoɖ", "1"),
        "MCH": ("ɾoɖɔ", "1"), "MDI": ("ɾoɾ", "1"),
        "MDH": ("ɾoɖ", "1"), "MJH": ("ɾoboɾ", "3"),
        "HDI": ("ɾo", "5"), "SDI": ("hindʒit | tʃuttʃɑt", "6 | 7"),
        "SNA": ("ɾohol", "2"), "ORI": ("sukhilɑ", "8"),
    }),
    134: ("long", {
        "BAI": ("dʒiliŋ", "1"), "CHA": ("dʒiliŋ", "1"),
        "DIG": ("dʒiliŋ", "1"), "DUM": ("dʒiliŋ", "1"),
        "LAD": ("dʒiliŋ", "1"), "MAD": ("dʒiliŋ", "1"),
        "MOH": ("dʒiliŋ", "1"), "MUN": ("dʒiliŋ", "1"),
        "POD": ("dʒiliŋ", "1"), "UDA": ("dʒiliŋ", "1"),
        "MCH": ("dʒiliŋ", "1"), "MDI": ("dʒiliŋ", "1"),
        "MDH": ("dʒiliŋ", "1"), "MJH": ("dʒiliŋ", "1"),
        "HDI": ("dʒiliŋ", "1"), "SDI": ("dʒelen | dʒɦɑɭ", "1 | 2"),
        "SNA": ("dʒibel", "1"), "ORI": ("lombɑ", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 130 or (item == 131 and code in ITEM131_PAGE59):
        return "59", "54", "right"
    if item in {131, 132}:
        return "60", "55", "left"
    if item == 133:
        return "60", "55", "left/right" if code == "SDI" else (
            "right" if code in {"SNA", "ORI"} else "left"
        )
    return "60", "55", "right"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = source_coordinates(item, code)
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": "attested", "Confidence": "high",
                "Uncertainty": "", "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 5 * 18 == 90
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 99
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
