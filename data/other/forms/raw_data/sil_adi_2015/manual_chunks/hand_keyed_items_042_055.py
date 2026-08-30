#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 42--55."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_042_055_hand_keyed.tsv"
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
    42: ("peanut", "left", {
        "MN": ("", "0"), "BR": ("", "0"), "RM": ("bad̪am", "1"),
        "ML": ("bad̪am", "1"), "PL": ("", "0"), "AS": ("", "0"),
        "PD": ("", "0"), "SM": ("bad̪am", "1"), "BK": ("bad̪am", "1"),
    }),
    43: ("tree", "left", {
        "MN": ("əsɯŋanə", "1"), "BR": ("əʃɯŋ", "1"),
        "RM": ("ʃɯ̃n", "1"), "ML": ("haŋɲi", "2"),
        "PL": ("ʃɯnə", "1"), "AS": ("əsɯŋanə", "1"),
        "PD": ("əʃɯŋ", "1"), "SM": ("əʃɯŋ", "1"), "BK": ("ʃɯŋne", "1"),
    }),
    44: ("branch", "left", {
        "MN": ("aːk", "1"), "BR": ("aːk", "1"),
        "RM": ("hokkekh", "2"), "ML": ("akh", "1"),
        "PL": ("akkək", "1"), "AS": ("agbe", "3"),
        "PD": ("aːk", "1"), "SM": ("aːk", "1"), "BK": ("aok", "1"),
    }),
    45: ("leaf", "left", {
        "MN": ("annə", "1"), "BR": ("aboɾ", "2"),
        "RM": ("ənɯ", "1"), "ML": ("t̪annom", "4"),
        "PL": ("anɯ", "1"), "AS": ("annə", "1"),
        "PD": ("annə", "1"), "SM": ("annə", "1"), "BK": ("anə", "1"),
    }),
    46: ("thorn", "left", {
        "MN": ("t̪aːŋ", "1"), "BR": ("pabu", "2"),
        "RM": ("pəbu", "2"), "ML": ("t̪anu | t̪anu", "1 | 3"),
        "PL": ("pabu", "2"), "AS": ("t̪ad̪ɯ", "3"),
        "PD": ("t̪aːŋ", "1"), "SM": ("t̪aːŋ", "1"), "BK": ("pabu", "2"),
    }),
    47: ("root", "middle", {
        "MN": ("ɑʔɯɾ", "1"), "BR": ("apɯɾ", "1"),
        "RM": ("pəpɯɾ", "1"), "ML": ("t̪apɯɾ", "1"),
        "PL": ("pəpɯɾ", "1"), "AS": ("appɯɾ", "1"),
        "PD": ("appɯɾ", "1"), "SM": ("appɯɾ", "1"), "BK": ("papɯɾ", "1"),
    }),
    48: ("bamboo", "middle", {
        "MN": ("d̪ibaŋ", "1"), "BR": ("ee", "2"), "RM": ("ja", "5"),
        "ML": ("ahu", "6"), "PL": ("ee", "2"), "AS": ("eŋ", "4"),
        "PD": ("ej", "3"), "SM": ("eŋ", "4"), "BK": ("ja", "5"),
    }),
    49: ("fruit", "middle", {
        "MN": ("apɯaɾɯ", "1"), "BR": ("apɯaje | apɯaje", "1 | 3"),
        "RM": ("əpɯ", "1"), "ML": ("baŋɲiaʃi", "2"),
        "PL": ("aje", "3"), "AS": ("apɯaje | apɯaje", "1 | 3"),
        "PD": ("əʃɯŋaje | əʃɯŋaje", "1 | 3"),
        "SM": ("apɯaje | aɯaje", "1 | 3"),
        "BK": ("əʃɯŋaje | əʃɯŋaje", "1 | 3"),
    }),
    50: ("jack fruit", "middle", {
        "MN": ("bəlaŋ", "1"), "BR": ("balaŋ", "1"), "RM": ("", "0"),
        "ML": ("bala", "1"), "PL": ("bəla", "1"),
        "AS": ("bəlaŋ", "1"), "PD": ("bəlaŋ", "1"),
        "SM": ("bəlaŋ", "1"), "BK": ("balaŋ", "1"),
    }),
    51: ("coconut (ripe)", "right", {
        "MN": ("", "0"), "BR": ("", "0"), "RM": ("naɾijal", "1"),
        "ML": ("", "0"), "PL": ("", "0"), "AS": ("", "0"),
        "PD": ("", "0"), "SM": ("", "0"), "BK": ("", "0"),
    }),
    52: ("banana", "right", {
        "MN": ("koʔak", "1"), "BR": ("kopak", "1"),
        "RM": ("kopakh", "1"), "ML": ("pagbe", "2"),
        "PL": ("kopak", "1"), "AS": ("kopak", "1"),
        "PD": ("kopak", "1"), "SM": ("kopak", "1"), "BK": ("kopak", "1"),
    }),
    53: ("mango", "right", {
        "MN": ("t̪aguŋ", "1"), "BR": ("t̪aguŋ", "1"), "RM": ("", "0"),
        "ML": ("t̪ahuŋ", "1"), "PL": ("t̪agu", "1"),
        "AS": ("t̪aguŋ", "1"), "PD": ("t̪aguŋ", "1"),
        "SM": ("t̪aguŋ", "1"), "BK": ("am", "2"),
    }),
    54: ("flower", "right", {
        "MN": ("aʔun", "1"), "BR": ("appun | appun", "1 | 3"),
        "RM": ("põpin", "2"), "ML": ("appun | appun", "1 | 3"),
        "PL": ("apu", "3"), "AS": ("appun | appun", "1 | 3"),
        "PD": ("appun | appun", "1 | 3"),
        "SM": ("appun | appun", "1 | 3"), "BK": ("põpin", "2"),
    }),
    55: ("seed", "right", {
        "MN": ("amɯ", "1"), "BR": ("amɯ", "1"), "RM": ("əlɯ", "2"),
        "ML": ("ɾamɯ", "1"), "PL": ("aje", "3"),
        "AS": ("amɯ", "1"), "PD": ("amɯ", "1"),
        "SM": ("amɯ", "1"), "BK": ("aje", "3"),
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
                "Site_Name": name, "PDF_Page": "20", "Printed_Page": "16",
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
