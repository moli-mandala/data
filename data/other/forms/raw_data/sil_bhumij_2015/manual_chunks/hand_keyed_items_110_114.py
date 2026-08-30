#!/usr/bin/env python3
"""Write visually checked Bhumij Appendix B3 cells for items 110--114."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_110_114_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF pages with retroflexes, "
    "nasalization, dental marks, vowel length, continuations, and page/column "
    "breaks rechecked at 800 dpi; text scaffold neither supplied nor verified "
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
ITEM114_LEFT = {"BAI", "CHA", "DIG", "DUM", "LAD", "MAD", "MOH", "MUN", "POD"}
BLANKS = {(113, "BAI"), (113, "HDI"), (114, "MJH"), (114, "HDI")}

DATA = {
    110: ("younger sister", {
        "BAI": ("buɖi", "2"), "CHA": ("hon misi", "1"),
        "DIG": ("misi", "1"), "DUM": ("misi iɾɑ", "1"),
        "LAD": ("uɾiŋmisi", "1"), "MAD": ("huɾiŋni", "1"),
        "MOH": ("misi", "1"), "MUN": ("hon misi", "1"),
        "POD": ("misi iɾɑ", "1"), "UDA": ("huɾin buɖi", "2"),
        "MCH": ("misi", "1"), "MDI": ("misi", "1"),
        "MDH": ("huɾin buɖi", "2"), "MJH": ("misi", "1"),
        "HDI": ("edʒin", "4"), "SDI": ("bokot kuɾi", "5"),
        "SNA": ("hepɔn mɑi", "3"), "ORI": ("sɑnəbɦouni", "6"),
    }),
    111: ("son", {
        "BAI": ("koɾɑ hon", "1"), "CHA": ("hon heɾel", "1"),
        "DIG": ("hɔn koɖɑ", "1"), "DUM": ("hone", "1"),
        "LAD": ("koɾɑ hɔn", "1"), "MAD": ("koɾɑ hon heɾel", "1"),
        "MOH": ("kɔɖɑ hɔne", "1"), "MUN": ("hone", "1"),
        "POD": ("kuɖɑ hone", "1"), "UDA": ("koɖɑ hõn", "1"),
        "MCH": ("koɖɑ hõn", "1"), "MDI": ("hon", "1"),
        "MDH": ("koɖɑ hõn", "1"), "MJH": ("kũɑ", "1"),
        "HDI": ("hon", "1"), "SDI": ("hon | koɾɑ hopon", "1 | 1"),
        "SNA": ("koɖɑ", "1"), "ORI": ("puːo", "2"),
    }),
    112: ("daughter", {
        "BAI": ("kuɾi hon", "1"), "CHA": ("hon eɾɑ", "1"),
        "DIG": ("hon kuɖi", "1"), "DUM": ("kuɖi hone", "1"),
        "LAD": ("kuɾi hɔn", "1"), "MAD": ("kuɖihoniɾɑ", "1"),
        "MOH": ("kuɖi hɔne", "1"), "MUN": ("honeɾɑ", "1"),
        "POD": ("kuɖi hone", "1"), "UDA": ("kuɖi hon", "1"),
        "MCH": ("kuɖi hõn", "1"), "MDI": ("kuɾi hon", "1"),
        "MDH": ("kuɖi hon", "1"), "MJH": ("kuɖi", "1"),
        "HDI": ("mɑi", "2"), "SDI": ("hopon eɾɑ", "1"),
        "SNA": ("kuɖi", "1"), "ORI": ("dʒiːo", "3"),
    }),
    113: ("husband", {
        "BAI": ("", ""), "CHA": ("koɖɑ", "1"),
        "DIG": ("kisɑn", "2"), "DUM": ("koɖɑ", "1"),
        "LAD": ("kisɑŋ", "2"), "MAD": ("kodɑ", "1"),
        "MOH": ("ejɑkoɖɑ", "1"), "MUN": ("koɖɑ", "1"),
        "POD": ("kuɖɑ", "1"), "UDA": ("koɖɑ", "1"),
        "MCH": ("koɖɑ", "1"), "MDI": ("koɾɑ | puɾus", "1 | 5"),
        "MDH": ("koɖɑ", "1"), "MJH": ("koɖɑ", "1"),
        "HDI": ("", ""), "SDI": ("dʒɑ̃wɑ̃e | heɾel", "3 | 4"),
        "SNA": ("dʒɛwɑi", "3"), "ORI": ("suɑmi", "6"),
    }),
    114: ("wife", {
        "BAI": ("hɑdɑm buɖi", "1"), "CHA": ("iɾɑ", "3"),
        "DIG": ("t̪iɾi", "5"), "DUM": ("eɾɑ", "3"),
        "LAD": ("kuɾi", "1"), "MAD": ("kuɖi", "1"),
        "MOH": ("kuɖi", "1"), "MUN": ("bɑhu", "2"),
        "POD": ("buɖi", "1"), "UDA": ("iɾɑ | buɖi", "3 | 1"),
        "MCH": ("kuɖi", "1"), "MDI": ("kuɾi | oɾɑ hoɾo", "1 | 4"),
        "MDH": ("buɖi | kuɖi", "1 | 1"), "MJH": ("", ""),
        "HDI": ("", ""), "SDI": ("bɑhu | oɾɑk hoɾ", "2 | 4"),
        "SNA": ("bɛhu", "2"), "ORI": ("st̪ɾi", "6"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Language_Label", "Site_Name", "Target",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription",
    "Source_Cognate_Labels", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def source_coordinates(item, code):
    if item in {110, 111}:
        return "55", "50", "right"
    if item in {112, 113} or (item == 114 and code in ITEM114_LEFT):
        return "56", "51", "left"
    return "56", "51", "right"


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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 4
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
