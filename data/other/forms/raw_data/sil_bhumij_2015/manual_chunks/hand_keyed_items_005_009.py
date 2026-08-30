#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 5--9."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_005_009_hand_keyed.tsv"
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

DATA = {
    5: ("eye", {
        "BAI": ("meʔn", "1"), "CHA": ("metʔ", "1"),
        "DIG": ("meʔ", "1"), "DUM": ("metʔ", "1"),
        "LAD": ("metʔn̩", "1"), "MAD": ("met", "1"),
        "MOH": ("meʔn", "1"), "MUN": ("meʔn", "1"),
        "POD": ("metʔ", "1"), "UDA": ("meʔt̪", "1"),
        "MCH": ("meʔ", "1"), "MDI": ("med", "1"),
        "MDH": ("meʔt̪", "1"), "MJH": ("meʔt̪", "1"),
        "HDI": ("meʔn", "1"), "SDI": ("mẽt", "1"),
        "SNA": ("meʔt̪", "1"), "ORI": ("ɑkhi", "2"),
    }),
    6: ("ear", {
        "BAI": ("lut̪uɾ", "1"), "CHA": ("lut̪uɾ", "1"),
        "DIG": ("lut̪uɾ", "1"), "DUM": ("lut̪uɾ", "1"),
        "LAD": ("lut̪uɾ", "1"), "MAD": ("lut̪uɾ", "1"),
        "MOH": ("lut̪uɾ", "1"), "MUN": ("lut̪uɾ", "1"),
        "POD": ("lut̪uɾ", "1"), "UDA": ("lut̪uɾ", "1"),
        "MCH": ("lut̪uɾ", "1"), "MDI": ("lut̪uɾ", "1"),
        "MDH": ("lut̪uɾ", "1"), "MJH": ("lut̪uɾ", "1"),
        "HDI": ("lut̪uɾ", "1"), "SDI": ("lut̪uɾ", "1"),
        "SNA": ("lut̪uɾ", "1"), "ORI": ("kɑɳo", "2"),
    }),
    7: ("nose", {
        "BAI": ("muː", "1"), "CHA": ("muː", "1"),
        "DIG": ("mu", "1"), "DUM": ("mũ", "1"),
        "LAD": ("mũ", "1"), "MAD": ("mu", "1"),
        "MOH": ("mu", "1"), "MUN": ("mu", "1"),
        "POD": ("mu", "1"), "UDA": ("mũ", "1"),
        "MCH": ("mu", "1"), "MDI": ("muhu", "1"),
        "MDH": ("mũ", "1"), "MJH": ("muhũ", "1"),
        "HDI": ("muʈe", "1"), "SDI": ("mũ", "1"),
        "SNA": ("mu", "1"), "ORI": ("nɑkho", "2"),
    }),
    8: ("mouth", {
        "BAI": ("motʃɑ", "1"), "CHA": ("motʃʌŋ", "1"),
        "DIG": ("mɔtʃɑ", "1"), "DUM": ("motʃoŋ", "1"),
        "LAD": ("motʃɑ", "1"), "MAD": ("motʃoŋ", "1"),
        "MOH": ("luʈi", "3"), "MUN": ("mɔtʃɑ", "1"),
        "POD": ("motʃoŋ", "1"), "UDA": ("motʃɑ", "1"),
        "MCH": ("mɔtʃɑ", "1"), "MDI": ("motʃɑ | thotnɑ", "1 | 4"),
        "MDH": ("motʃɑ", "1"), "MJH": ("mɔtʃɑ", "1"),
        "HDI": ("ɑʔ", "2"), "SDI": ("ɑ", "2"),
        "SNA": ("mɔtʃɑ", "1"), "ORI": ("pɑʈːi", "5"),
    }),
    9: ("tooth", {
        "BAI": ("dɑtɑ", "1"), "CHA": ("dɑtɑ", "1"),
        "DIG": ("ɖɑʈɑ", "1"), "DUM": ("dɑtɑ", "1"),
        "LAD": ("d̪ɑʔtɑ", "1"), "MAD": ("dɑtɑ", "1"),
        "MOH": ("ɖɑʈɑ", "1"), "MUN": ("ɖɑʈɑ", "1"),
        "POD": ("dɑtɑ", "1"), "UDA": ("dɑtɑ", "1"),
        "MCH": ("ɖɐʈɑ", "1"), "MDI": ("ɖɑʈɑ", "1"),
        "MDH": ("dɑtɑ", "1"), "MJH": ("ɖɑʈɑ", "1"),
        "HDI": ("ɖɐʈɑ", "1"), "SDI": ("ɖɑʈɑ", "1"),
        "SNA": ("ɖɑʈɑ", "1"), "ORI": ("d̪ɑnt̪o", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 5 and code in {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH"}:
        return "34", "29", "right"
    if item in {5, 6, 7}:
        return "35", "30", "left"
    return "35", "30", "right"


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
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
