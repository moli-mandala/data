#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 1--4."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_001_004_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
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
    1: ("body", {
        "BAI": ("hoɖomo", "1"), "CHA": ("hoɖomo", "1"),
        "DIG": ("hɔɭmo", "1"), "DUM": ("hoɾəmo", "1"),
        "LAD": ("hoɾəmo", "1"), "MAD": ("hoɖomo", "1"),
        "MOH": ("hɔɖmo", "1"), "MUN": ("hɔɖmo", "1"),
        "POD": ("hoɖomo", "1"), "UDA": ("hoɖomo", "1"),
        "MCH": ("hɔɖmo", "1"), "MDI": ("hɔɖmo | hoɽmo", "1 | 1"),
        "MDH": ("hoɖomo", "1"), "MJH": ("hɔɖmo", "1"),
        "HDI": ("hɔmo", "1"), "SDI": ("hoɾmo", "1"),
        "SNA": ("hɔɾmo", "1"), "ORI": ("soɾiɾo", "2"),
    }),
    2: ("head", {
        "BAI": ("boʔo", "1"), "CHA": ("boʔ", "1"),
        "DIG": ("bɔʔo", "1"), "DUM": ("boʔo", "1"),
        "LAD": ("boʔo", "1"), "MAD": ("bohoʔ", "1"),
        "MOH": ("bɔho", "1"), "MUN": ("bo", "1"),
        "POD": ("boho", "1"), "UDA": ("boʔo", "1"),
        "MCH": ("bɔʔo", "1"), "MDI": ("bo | mund", "1 | 2"),
        "MDH": ("boʔo", "1"), "MJH": ("bɔho", "1"),
        "HDI": ("bo", "1"), "SDI": ("bohok", "1"),
        "SNA": ("bɔho", "1"), "ORI": ("mũnɖɔ", "2"),
    }),
    3: ("hair", {
        "BAI": ("boʔo up", "1"), "CHA": ("uʔp", "1"),
        "DIG": ("up", "1"), "DUM": ("boʔo uʔb", "1"),
        "LAD": ("ubʔ", "1"), "MAD": ("uːb", "1"),
        "MOH": ("uʔp", "1"), "MUN": ("uʔp", "1"),
        "POD": ("ub", "1"), "UDA": ("uʔp", "1"),
        "MCH": ("uʔmɪm", "3"), "MDI": ("ub", "1"),
        "MDH": ("uʔp", "1"), "MJH": ("uʔp", "1"),
        "HDI": ("bɐle", "2"), "SDI": ("uʔp", "1"),
        "SNA": ("uʔp", "1"), "ORI": ("bɑlə", "2"),
    }),
    4: ("face", {
        "BAI": ("meʔt moɑŋ", "1"), "CHA": ("met mutɑ", "1"),
        "DIG": ("menmuhɑ̃ʔ", "1"), "DUM": ("meʔd moʈe", "1"),
        "LAD": ("metʔn̩mũɑɳɑ", "1"), "MAD": ("met mute", "1"),
        "MOH": ("menmuhɑɖ", "1"), "MUN": ("menmuʈɑ", "1"),
        "POD": ("met mute", "1"), "UDA": ("meʔt mũɑ", "1"),
        "MCH": ("menmuɑ̃ɳ", "1"), "MDI": ("med muɑnɾɑ", "1"),
        "MDH": ("meʔt motʃɑ", "1"), "MJH": ("tʃɐnkɑ", "2"),
        "HDI": ("menmutʃɑ", "1"), "SDI": ("mẽtʔ ɑ̃hɑ̃", "1"),
        "SNA": ("mənhɑ̃ʔ", "1"), "ORI": ("mũhə", "1"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, (language, site, target) in SITES.items():
            form, labels = cells[code]
            column = "right" if item == 4 or (item == 3 and code not in {"BAI", "CHA"}) else "left"
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Language_Label": language, "Site_Name": site,
                "Target": "yes" if target else "no", "PDF_Page": "34",
                "Printed_Page": "29", "Column": column,
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
    assert len(rows) == 4 * 18 == 72
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
