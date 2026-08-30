#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 115--128."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_115_128_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
    "text scaffold not accepted without cell visual match"
)
SITES = {
    "MN": "Minyong", "BR": "Bori", "RM": "Ramo", "ML": "Milang",
    "PL": "Pailibo", "AS": "Ashing (Bogum Bokang)", "PD": "Padam",
    "SM": "Shimong", "BK": "Bokar",
}

DATA = {
    115: ("fingernail", "left", {
        "MN": ("lagjin", "1"), "BR": ("lajin", "1"),
        "RM": ("lokjin", "1"), "ML": ("lakhan", "2"),
        "PL": ("lagjin", "1"), "AS": ("lagjin", "1"),
        "PD": ("lagjin", "1"), "SM": ("lagjin", "1"), "BK": ("lukjin", "1"),
    }),
    116: ("knee", "left", {
        "MN": ("ləbɯŋpoːe", "1"), "BR": ("ləbɯŋ", "1"),
        "RM": ("ləbɯŋ", "1"), "ML": ("bjabaŋ", "2"),
        "PL": ("ləbɯ", "1"), "AS": ("ləbɯŋ", "1"),
        "PD": ("ləbɯŋ", "1"), "SM": ("ləbɯŋ", "1"), "BK": ("lɯbɯŋ", "1"),
    }),
    117: ("foot", "left", {
        "MN": ("lət̪ɯŋ", "4"), "BR": ("alə", "1"), "RM": ("alə", "1"),
        "ML": ("bjapiu", "5"), "PL": ("ləʃo", "3"),
        "AS": ("ləpio", "2"), "PD": ("ləpio", "2"),
        "SM": ("ləpio", "2"), "BK": ("ləpio", "2"),
    }),
    118: ("bone", "left", {
        "MN": ("aloŋ", "1"), "BR": ("aloŋ", "1"), "RM": ("lõpoŋ", "2"),
        "ML": ("alo", "1"), "PL": ("lopo", "2"), "AS": ("aloŋ", "1"),
        "PD": ("aloŋ", "1"), "SM": ("aloŋ", "1"), "BK": ("lõpoŋ", "2"),
    }),
    119: ("fat", "left", {
        "MN": ("una", "1"), "BR": ("hunə | hunə", "1 | 2"),
        "RM": ("hunə | hunə", "1 | 2"), "ML": ("ahuɲi", "2"),
        "PL": ("unə", "1"), "AS": ("unə", "1"), "PD": ("oph", "3"),
        "SM": ("unə", "1"), "BK": ("hunə̃ | hunə̃", "1 | 2"),
    }),
    120: ("skin", "middle", {
        "MN": ("ajo", "1"), "BR": ("aʃik", "2"), "RM": ("epin", "3"),
        "ML": ("apan", "3"), "PL": ("apin", "3"), "AS": ("aʃɯk", "2"),
        "PD": ("ajo", "1"), "SM": ("ahɯk", "2"), "BK": ("apin", "3"),
    }),
    121: ("blood", "middle", {
        "MN": ("ijji", "1"), "BR": ("iji", "1"), "RM": ("uji", "1"),
        "ML": ("ajji", "1"), "PL": ("uji", "1"), "AS": ("ɯjjɯ", "1"),
        "PD": ("ijji", "1"), "SM": ("ijji", "1"), "BK": ("uji", "1"),
    }),
    122: ("sweat", "middle", {
        "MN": ("ɯɾnam", "1"), "BR": ("hanʃɯɾ | hɯɾbut̪", "2 | 7"),
        "RM": ("hõãɾ", "3"), "ML": ("kalʃi", "4"),
        "PL": ("aːɾuk", "5"), "AS": ("ɯɾnam", "1"),
        "PD": ("ɯlnam", "1"), "SM": ("ɯɾnam", "1"), "BK": ("hõŋhaɾ", "6"),
    }),
    123: ("belly", "middle", {
        "MN": ("kiːoŋ", "1"), "BR": ("aki", "1"), "RM": ("kipoŋ", "1"),
        "ML": ("t̪ha", "2"), "PL": ("kipo", "1"), "AS": ("kipoŋ", "1"),
        "PD": ("aki", "1"), "SM": ("aki", "1"), "BK": ("kipo", "1"),
    }),
    124: ("heart (organ)", "middle", {
        "MN": ("aʔɯ", "1"), "BR": ("aːŋ", "3"), "RM": ("hinoŋ", "4"),
        "ML": ("hapɯ", "2"), "PL": ("aːpuk", "2"), "AS": ("apuk", "2"),
        "PD": ("apɯ | apɯ", "1 | 2"), "SM": ("apɯ | apɯ", "1 | 2"),
        "BK": ("hinoŋ", "4"),
    }),
    125: ("back", "right", {
        "MN": ("lamku", "1"), "BR": ("laŋko", "1"), "RM": ("lamko", "1"),
        "ML": ("ɾamə", "2"), "PL": ("lamko", "1"), "AS": ("laŋku", "1"),
        "PD": ("lamku", "1"), "SM": ("lamku", "1"), "BK": ("lamko", "1"),
    }),
    126: ("body", "right", {
        "MN": ("amɯɾ", "2"), "BR": ("amɯɾ", "2"), "RM": ("eɯ", "1"),
        "ML": ("amɯl", "2"), "PL": ("aɯ", "1"), "AS": ("amɯɾ", "2"),
        "PD": ("amɯl", "2"), "SM": ("amɯɾ", "2"), "BK": ("aɯ", "1"),
    }),
    127: ("person", "right", {
        "MN": ("ami", "1"), "BR": ("ami", "1"), "RM": ("mi", "1"),
        "ML": ("mi", "1"), "PL": ("ami", "1"), "AS": ("ami", "1"),
        "PD": ("ami", "1"), "SM": ("ami", "1"), "BK": ("mi", "1"),
    }),
    128: ("man", "right", {
        "MN": ("miloko", "1"), "BR": ("ami", "1"),
        "RM": ("mit̪uɾ | mit̪uɾ", "1 | 3"), "ML": ("malu", "2"),
        "PL": ("ɲit̪uɾ", "3"), "AS": ("miloko", "1"),
        "PD": ("milokoŋ", "1"), "SM": ("miloko", "1"),
        "BK": ("mit̪uɾ | mit̪uɾ", "1 | 3"),
    }),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def build_rows():
    rows = []
    for item, (gloss, column, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, name in SITES.items():
            form, labels = cells[code]
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": "25", "Printed_Page": "21",
                "Column": column, "Manual_Transcription": form,
                "Source_Cognate_Labels": labels, "Review_Status": "attested",
                "Confidence": "high", "Uncertainty": "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 14 * 9 == 126
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
