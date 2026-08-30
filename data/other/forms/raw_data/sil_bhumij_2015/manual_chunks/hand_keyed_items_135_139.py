#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 135--139."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_135_139_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with dental and "
    "retroflex marks, nasalization, length, continuations, and the page/column "
    "breaks rechecked at 800 dpi; OCR/PDF text neither supplied nor verified "
    "any reading"
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
ITEM136_PAGE60 = {"BAI", "CHA", "DIG", "DUM"}
ITEM138_LEFT = {
    "BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD",
    "UDA", "MCH", "MDI", "MDH", "MJH", "HDI",
}

DATA = {
    135: ("short", {
        "BAI": ("huɖiŋ", "3"), "CHA": ("khɑtʌ", "1"),
        "DIG": ("huɖiŋ", "3"), "DUM": ("khʌto", "1"),
        "LAD": ("kʌto", "1"), "MAD": ("khɑto", "1"),
        "MOH": ("huɖiŋ", "3"), "MUN": ("huɖiŋ", "3"),
        "POD": ("kʌto", "1"), "UDA": ("khɑ̃diɑ", "2"),
        "MCH": ("diŋgɑʔɑ", "5"), "MDI": ("huɾiŋ | t̪um", "3 | 6"),
        "MDH": ("khɑndiɑ", "2"), "MJH": ("huɖiŋ", "3"),
        "HDI": ("ɖuŋkui", "4"), "SDI": ("khɑto | geɖɑ", "1 | 7"),
        "SNA": ("buɖiŋ", "3"), "ORI": ("tsoʈiɑ", "8"),
    }),
    136: ("hot", {
        "BAI": ("lolo", "1"), "CHA": ("lʌlʌ", "1"),
        "DIG": ("lɔlo", "1"), "DUM": ("lʌlʌ", "1"),
        "LAD": ("lolo", "1"), "MAD": ("lɛlɛ", "1"),
        "MOH": ("lɔlo", "1"), "MUN": ("lɔlo", "1"),
        "POD": ("lolo", "1"), "UDA": ("lɛlɛ", "1"),
        "MCH": ("lɔlo", "1"), "MDI": ("lolo | dʒete", "1 | 2"),
        "MDH": ("lɛlɛ", "1"), "MJH": ("lɔlo", "1"),
        "HDI": ("lɔlo", "1"), "SDI": ("lolo", "1"),
        "SNA": ("lɔlo", "1"), "ORI": ("goɾɑmo", "3"),
    }),
    137: ("cold", {
        "BAI": ("ɾejɑɾɑ", "1"), "CHA": ("ɾijʌd", "1"),
        "DIG": ("ɾeɭɑ", "1"), "DUM": ("ɾejɑɖɑ", "1"),
        "LAD": ("ɾeɭɑ", "1"), "MAD": ("ɾijed", "1"),
        "MOH": ("ɾejɑɾ", "1"), "MUN": ("ɾɛjɑɾ", "1"),
        "POD": ("ɾijʌd | ɾʌbʌn", "1 | 2"), "UDA": ("ɾijɑd", "1"),
        "MCH": ("t̪ut̪ukun", "4"), "MDI": ("ɾɑbɑŋ", "2"),
        "MDH": ("ɾijɑd", "1"), "MJH": ("ɾɑbɑŋ", "2"),
        "HDI": ("sɑsɑ", "3"), "SDI": ("ɾɑbɑn", "2"),
        "SNA": ("ɾijɑɾ", "1"), "ORI": ("t̪hɑndɑ", "5"),
    }),
    138: ("right", {
        "BAI": ("dʒʌm t̪i", "1"), "CHA": ("mɑndi kuʈi", "2"),
        "DIG": ("dʒɔm sɑi", "1"), "DUM": ("dʒʌdʒoŋ", "1"),
        "LAD": ("dʒom", "1"), "MAD": ("dʒodʒemkuti", "1"),
        "MOH": ("dʒɛ dʒɔm", "1"), "MUN": ("mɑndit̪i", "2"),
        "POD": ("dʒo dʒom kuti", "1"), "UDA": ("dʒɔm sɑi", "1"),
        "MCH": ("dʒom", "1"), "MDI": ("dʒom t̪i", "1"),
        "MDH": ("dʒom sɑi", "1"), "MJH": ("dʒom t̪iʔ", "1"),
        "HDI": ("dʒom t̪i", "1"), "SDI": ("dʒo dʒom", "1"),
        "SNA": ("dʒɑ dʒɔm", "1"), "ORI": ("ɖɑhɑno", "3"),
    }),
    139: ("left", {
        "BAI": ("leŋgɑ t̪i", "1"), "CHA": ("liŋgɑ kuʈi", "1"),
        "DIG": ("lɛkɑ sɑi", "1"), "DUM": ("liŋgɑ", "1"),
        "LAD": ("liŋgɑ", "1"), "MAD": ("leŋgɑt̪i", "1"),
        "MOH": ("lɛŋkɑ", "1"), "MUN": ("lɛŋkɑt̪i", "1"),
        "POD": ("liŋgɑ kuti", "1"), "UDA": ("leŋgɑ sɑi", "1"),
        "MCH": ("lɛŋgɑ", "1"), "MDI": ("leŋgɑ", "1"),
        "MDH": ("leŋgɑ sɑi", "1"), "MJH": ("lɛŋkɑ t̪i", "1"),
        "HDI": ("liŋpt̪i", "1"), "SDI": ("leŋgɑ", "1"),
        "SNA": ("lɛŋkɑ", "1"), "ORI": ("bɑːmo", "2"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 135 or (item == 136 and code in ITEM136_PAGE60):
        return "60", "55", "right"
    if item in {136, 137} or (item == 138 and code in ITEM138_LEFT):
        return "61", "56", "left"
    return "61", "56", "right"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in rows) == 94
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
