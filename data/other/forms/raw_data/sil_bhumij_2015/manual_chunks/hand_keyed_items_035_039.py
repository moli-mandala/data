#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 35--39."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_035_039_hand_keyed.tsv"
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
ITEM35_P40 = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD", "UDA", "MCH", "MDI", "MDH"}
ITEM38_LEFT = {"BAI", "CHA"}

DATA = {
    35: ("axe", {
        "BAI": ("hɑke", "2"), "CHA": ("boɖiɑ", "3"),
        "DIG": ("hɐke", "2"), "DUM": ("boɖiɑ", "3"),
        "LAD": ("hɑke", "2"), "MAD": ("boɖiɑ", "3"),
        "MOH": ("bɔɖɛjɑ", "3"), "MUN": ("hɐke", "2"),
        "POD": ("boɖejɑ", "3"), "UDA": ("hɑke", "2"),
        "MCH": ("hɐke | hɔɽɑmhɑke", "2 | 2"), "MDI": ("kɑpi", "4"),
        "MDH": ("hɑke", "2"), "MJH": ("hɐke", "2"),
        "HDI": ("hɐke", "2"), "SDI": ("tɑŋgu | potɑm", "1 | 5"),
        "SNA": ("ʈɐŋkɑ", "1"), "ORI": ("tɑŋgiːɑ", "1"),
    }),
    36: ("rope", {
        "BAI": ("bɑjɑɾ", "1"), "CHA": ("bɑjɑɾ", "1"),
        "DIG": ("bɐɾɑi", "2"), "DUM": ("bɑbeɾ", "1"),
        "LAD": ("bɑijɑɾ", "1"), "MAD": ("bɑbeɾ", "1"),
        "MOH": ("bɐbeɾ", "1"), "MUN": ("bɐjɛɾ", "1"),
        "POD": ("bɑbeɾ", "1"), "UDA": ("bɑjɑɾ", "1"),
        "MCH": ("bɑjɑɾ", "1"), "MDI": ("boɽ", "3"),
        "MDH": ("bɑjɑɾ", "1"), "MJH": ("bɑibɑɾ", "1"),
        "HDI": ("bɐjɛɾ", "1"),
        "SDI": ("bɑhɑɾi | bɑhɑɾi | boɽ", "1 | 2 | 3"),
        "SNA": ("bɐbɛɾ", "1"), "ORI": ("dowudi", "4"),
    }),
    37: ("thread", {
        "BAI": ("sut̪ɑm", "1"), "CHA": ("sut̪em", "1"),
        "DIG": ("sut̪ɑm", "1"), "DUM": ("sut̪em", "1"),
        "LAD": ("sut̪ʌm", "1"), "MAD": ("sut̪em", "1"),
        "MOH": ("sut̪ɑm", "1"), "MUN": ("sut̪ɑm", "1"),
        "POD": ("sut̪em", "1"), "UDA": ("sut̪ɑm", "1"),
        "MCH": ("sut̪ɑm", "1"), "MDI": ("sut̪ɑm", "1"),
        "MDH": ("sut̪ɑm", "1"), "MJH": ("sut̪ɑm", "1"),
        "HDI": ("sut̪ɛm", "1"), "SDI": ("sut̪ɑm", "1"),
        "SNA": ("sut̪ɑm", "1"), "ORI": ("suːt̪ɑ", "1"),
    }),
    38: ("needle", {
        "BAI": ("sui", "1"), "CHA": ("susi", "1"),
        "DIG": ("sui", "1"), "DUM": ("sui", "1"),
        "LAD": ("sui", "1"), "MAD": ("sui", "1"),
        "MOH": ("sui", "1"), "MUN": ("susi", "1"),
        "POD": ("susi", "1"), "UDA": ("sui", "1"),
        "MCH": ("sui", "1"), "MDI": ("sui", "1"),
        "MDH": ("sui", "1"), "MJH": ("sui", "1"),
        "HDI": ("sudʒe", "2"), "SDI": ("sui", "1"),
        "SNA": ("sui", "1"), "ORI": ("sũnːtʃi", "2"),
    }),
    39: ("cloth", {
        "BAI": ("kitʃiɾi", "1"), "CHA": ("ulu", "5"),
        "DIG": ("kitʃiɾ", "1"), "DUM": ("tieŋ", "4"),
        "LAD": ("kitʃɪɾ", "1"), "MAD": ("tieŋ", "4"),
        "MOH": ("hulu", "5"), "MUN": ("hulu", "5"),
        "POD": ("tijʌŋ", "4"), "UDA": ("kitʃiɾi", "1"),
        "MCH": ("lidʒɑʔ", "2"),
        "MDI": ("kitʃɾi | lidʒɑ | lugɑ", "1 | 2 | 3"),
        "MDH": ("kitʃiɾi", "1"), "MJH": ("lidʒɑ", "2"),
        "HDI": ("lidʒe", "2"), "SDI": ("kitʃɾitʃ | lugɾi", "1 | 3"),
        "SNA": ("luguɖi", "3"), "ORI": ("luːgɑ", "3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item == 35 and code in ITEM35_P40:
        return "40", "35", "right"
    if item in {35, 36, 37} or (item == 38 and code in ITEM38_LEFT):
        return "41", "36", "left"
    return "41", "36", "right"


def qualifier_note(item, code):
    if item != 35:
        return ""
    if code == "MCH":
        return "source parenthetical qualifiers align to variants: small | big"
    if code in {"DIG", "MUN", "MJH", "HDI"}:
        return "source parenthetical qualifier: small"
    return ""


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
                "Uncertainty": qualifier_note(item, code), "Reviewer_Method": METHOD,
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
