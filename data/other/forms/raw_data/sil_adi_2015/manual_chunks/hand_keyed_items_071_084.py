#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 71--84."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_071_084_hand_keyed.tsv"
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
    71: ("monkey", "left", {
        "MN": ("hɯbeŋ", "2"), "BR": ("ʃibe | ʃibe", "1 | 2"),
        "RM": ("ʃebe | ʃebe", "1 | 2"), "ML": ("abe", "1"),
        "PL": ("ʃobe | ʃobe", "1 | 2"), "AS": ("ʃɯbej", "1"),
        "PD": ("ibe | ʃibe", "1 | 2"), "SM": ("hibeŋ", "2"),
        "BK": ("ʃəbe | ʃəbe", "1 | 2"),
    }),
    72: ("rabbit", "left", {
        "MN": ("ahuːpimuŋ", "1"), "BR": ("", "0"), "RM": ("", "0"),
        "ML": ("", "0"), "PL": ("kopu", "2"), "AS": ("", "0"),
        "PD": ("ʃit̪oɾud̪oŋ", "3"), "SM": ("ahipumuŋ", "1"),
        "BK": ("", "0"),
    }),
    73: ("snake", "left", {
        "MN": ("t̪abɯ", "1"), "BR": ("t̪abɯ", "1"),
        "RM": ("t̪əbɯ", "1"), "ML": ("d̪abɯ", "1"),
        "PL": ("t̪abɯ", "1"), "AS": ("t̪abɯ", "1"),
        "PD": ("t̪abɯ", "1"), "SM": ("t̪abɯ", "1"), "BK": ("t̪abɯ", "1"),
    }),
    74: ("crocodile", "left", {
        "MN": ("hoɾmon", "1"), "BR": ("", "0"), "RM": ("iʃbɾu", "2"),
        "ML": ("", "0"), "PL": ("buɾu", "2"), "AS": ("ʃoɾmon", "1"),
        "PD": ("ʃoɾmon", "1"), "SM": ("", "0"), "BK": ("buɾu", "2"),
    }),
    75: ("house lizard (gecko)", "left", {
        "MN": ("əkumuɲiŋ", "1"), "BR": ("", "0"),
        "RM": ("ʃõŋkɯŋ", "2"), "ML": ("tʃobɯlɯ", "3"),
        "PL": ("ʃodʒɯɾ", "4"), "AS": ("ʃomen", "5"),
        "PD": ("ʃipadʒondʒo", "6"), "SM": ("homen", "5"), "BK": ("", "0"),
    }),
    76: ("turtle", "middle", {
        "MN": ("raŋkop", "1"), "BR": ("mot̪oɾaŋkot̪", "1"),
        "RM": ("t̪atʃɯpad̪ɯɾ", "2"), "ML": ("koʃuɾaŋkop", "1"),
        "PL": ("ɾakop", "1"), "AS": ("raŋkop", "1"),
        "PD": ("raŋkop", "1"), "SM": ("raŋkop", "1"), "BK": ("raŋkop", "1"),
    }),
    77: ("frog", "middle", {
        "MN": ("t̪at̪ɯk", "1"), "BR": ("t̪at̪ɯk", "1"),
        "RM": ("t̪ət̪ɯk", "1"), "ML": ("pud̪uk", "2"),
        "PL": ("t̪at̪ɯk", "1"), "AS": ("t̪at̪ɯk", "1"),
        "PD": ("t̪at̪ɯk", "1"), "SM": ("t̪at̪ɯk", "1"),
        "BK": ("t̪at̪ɯk", "1"),
    }),
    78: ("dog", "middle", {
        "MN": ("əki", "1"), "BR": ("əki", "1"), "RM": ("ikki", "1"),
        "ML": ("akhe", "1"), "PL": ("ikki", "1"), "AS": ("əki", "1"),
        "PD": ("əki", "1"), "SM": ("əki", "1"), "BK": ("iki", "1"),
    }),
    79: ("cat", "middle", {
        "MN": ("mimikuɾi", "1"), "BR": ("mekuɾi", "1"),
        "RM": ("əli", "3"), "ML": ("kand̪aɾi", "2"),
        "PL": ("amikuɾi", "1"), "AS": ("billi", "4"),
        "PD": ("kand̪aɾi", "2"), "SM": ("kad̪aɾi", "2"), "BK": ("ali", "3"),
    }),
    80: ("cow", "right", {
        "MN": ("hoɯ", "1"), "BR": ("goru", "2"), "RM": ("god̪a", "2"),
        "ML": ("goru", "2"), "PL": ("ʃoə", "1"), "AS": ("ʃoɯ", "1"),
        "PD": ("goru", "2"), "SM": ("hoɯ", "1"), "BK": ("balaŋ", "3"),
    }),
    81: ("buffalo", "right", {
        "MN": ("bəndʒak", "1"), "BR": ("bendʒak", "1"),
        "RM": ("bəndʒak", "1"), "ML": ("bendʒak", "1"),
        "PL": ("mendʒik", "1"), "AS": ("bəndʒak", "1"),
        "PD": ("bəndʒak", "1"), "SM": ("bendʒak", "1"),
        "BK": ("bəndʒak", "1"),
    }),
    82: ("horn (of buffalo)", "right", {
        "MN": ("ɾəbuŋ", "1"), "BR": ("ɾəbuŋ", "1"), "RM": ("aɾəŋ", "2"),
        "ML": ("ɾəbuŋ", "1"), "PL": ("ɾəbu", "1"), "AS": ("ɾəbuŋ", "1"),
        "PD": ("aɾəŋ", "2"), "SM": ("ɾəbuŋ", "1"), "BK": ("aɾəŋ", "2"),
    }),
    83: ("tail", "right", {
        "MN": ("ammjo", "2"), "BR": ("ɲobuŋ", "3"), "RM": ("ãmjo", "2"),
        "ML": ("t̪ami", "1"), "PL": ("ɲobu", "3"), "AS": ("mebuŋ", "3"),
        "PD": ("t̪ame", "1"), "SM": ("ame", "1"), "BK": ("amɲio", "2"),
    }),
    84: ("goat", "right", {
        "MN": ("hoben", "1"), "BR": ("ʃoben", "1"), "RM": ("sobin", "1"),
        "ML": ("ʃoben", "1"), "PL": ("sobin", "1"), "AS": ("ʃoben", "1"),
        "PD": ("ʃoben", "1"), "SM": ("hoben", "1"), "BK": ("sobin", "1"),
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
            blank = not form
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": "22", "Printed_Page": "18",
                "Column": column, "Manual_Transcription": form,
                "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if blank else "attested",
                "Confidence": "" if blank else "high",
                "Uncertainty": "source prints no entry" if blank else "",
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
