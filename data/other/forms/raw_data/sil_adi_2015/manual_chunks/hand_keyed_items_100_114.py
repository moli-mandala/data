#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 100--114."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_100_114_hand_keyed.tsv"
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
    100: ("face", "left", {
        "MN": ("miŋmo", "1"), "BR": ("mimo", "1"), "RM": ("mimo", "1"),
        "ML": ("miŋmu", "1"), "PL": ("ɲikmo", "1"), "AS": ("mimo", "1"),
        "PD": ("miŋmo", "1"), "SM": ("miŋmo", "1"), "BK": ("mimo", "1"),
    }),
    101: ("neck", "left", {
        "MN": ("lɯŋguŋ", "1"), "BR": ("lɯŋguŋ", "1"),
        "RM": ("lɯŋguŋ", "1"), "ML": ("alaŋ", "2"),
        "PL": ("lɯpo", "3"), "AS": ("lɯŋguŋ", "1"),
        "PD": ("alɯŋ", "1"), "SM": ("alɯŋ", "1"),
        "BK": ("lɯŋguŋ | lɯ poŋ", "1 | 3"),
    }),
    102: ("hair", "left", {
        "MN": ("d̪umɯt̪", "1"), "BR": ("d̪umɯt̪", "1"),
        "RM": ("d̪um", "1"), "ML": ("d̪uma", "1"),
        "PL": ("d̪umɯ", "1"), "AS": ("d̪umɯt̪", "1"),
        "PD": ("d̪umɯt̪", "1"), "SM": ("d̪umɯt̪", "1"),
        "BK": ("d̪ummɯ", "1"),
    }),
    103: ("eye", "left", {
        "MN": ("amik", "1"), "BR": ("amit̪", "1"), "RM": ("mikh", "1"),
        "ML": ("amik", "1"), "PL": ("aɲik", "1"), "AS": ("imit̪", "1"),
        "PD": ("amik", "1"), "SM": ("amik", "1"), "BK": ("mikh", "1"),
    }),
    104: ("nose", "left", {
        "MN": ("ɲobuŋ", "1"), "BR": ("ɲobuŋ", "1"),
        "RM": ("ɲɛpum", "1"), "ML": ("nubuŋ", "1"),
        "PL": ("ɲapuŋ", "1"), "AS": ("ɲobuŋ", "1"),
        "PD": ("ɲobuŋ", "1"), "SM": ("ɲobuŋ", "1"), "BK": ("ɲapum", "1"),
    }),
    105: ("ear", "middle", {
        "MN": ("ɲoɾuŋ", "1"), "BR": ("ɲoɾuŋ", "1"),
        "RM": ("ɲɛɾuŋ", "1"), "ML": ("ɾaɲu", "2"),
        "PL": ("ɲaɾu", "1"), "AS": ("ɲoɾuŋ", "1"),
        "PD": ("ɲoɾuŋ", "1"), "SM": ("ɲoɾuŋ", "1"), "BK": ("ɲaɾuŋ", "1"),
    }),
    106: ("cheek", "middle", {
        "MN": ("molum", "1"), "BR": ("mimo", "2"),
        "RM": ("ɾud̪in", "3"), "ML": ("kemkem", "4"),
        "PL": ("ɾud̪in", "3"), "AS": ("molum", "1"),
        "PD": ("mopum", "1"), "SM": ("mopum", "1"), "BK": ("ɾud̪in", "3"),
    }),
    107: ("chin", "middle", {
        "MN": ("hoglə", "1"), "BR": ("ʃokkoɾ", "2"),
        "RM": ("ʃokloŋ", "2"), "ML": ("tʃokku", "2"),
        "PL": ("tʃokt̪am", "2"), "AS": ("tʃokko", "2"),
        "PD": ("ʃokkoɾ", "2"), "SM": ("ahok", "3"), "BK": ("malə", "4"),
    }),
    108: ("mouth", "middle", {
        "MN": ("nɑʔaŋ", "1"), "BR": ("nappaŋ | na ppaŋ", "1 | 2"),
        "RM": ("nappaŋ | nappaŋ", "1 | 2"), "ML": ("tʃaŋtʃi", "3"),
        "PL": ("nappha", "2"), "AS": ("ɲjappaŋ", "2"),
        "PD": ("nappaŋ | nappaŋ", "1 | 2"),
        "SM": ("nappaŋ | nappaŋ", "1 | 2"), "BK": ("ɲjappaŋ", "2"),
    }),
    109: ("tongue", "middle", {
        "MN": ("ajo", "1"), "BR": ("ajo", "1"), "RM": ("ajo", "1"),
        "ML": ("ʃid̪al", "2"), "PL": ("ajo", "1"), "AS": ("ajo", "1"),
        "PD": ("ajo", "1"), "SM": ("ajo", "1"), "BK": ("ajo", "1"),
    }),
    110: ("tooth", "right", {
        "MN": ("iːaŋ", "1"), "BR": ("hid̪uŋ", "2"),
        "RM": ("hidʒuŋ", "2"), "ML": ("ʃippa", "3"),
        "PL": ("idʒu", "4"), "AS": ("ipaŋ", "1"),
        "PD": ("ipaŋ", "1"), "SM": ("ipaŋ", "1"), "BK": ("hidʒuŋ", "2"),
    }),
    111: ("elbow", "right", {
        "MN": ("lagd̪ukoɲiŋ", "1"), "BR": ("lad̪u", "1"),
        "RM": ("loʔd̪u", "1"), "ML": ("lagdʒu", "1"),
        "PL": ("lagd̪u", "1"), "AS": ("lagd̪ukoɲiŋ", "1"),
        "PD": ("lagd̪ukoɲiŋ", "1"), "SM": ("lagd̪u", "1"),
        "BK": ("lukd̪u", "1"),
    }),
    112: ("hand", "right", {
        "MN": ("alak", "1"), "BR": ("alak", "1"), "RM": ("alokh", "1"),
        "ML": ("alak", "1"), "PL": ("alak", "1"), "AS": ("alak", "1"),
        "PD": ("alak", "1"), "SM": ("alag", "1"), "BK": ("alokh", "1"),
    }),
    113: ("palm", "right", {
        "MN": ("lakɯjo", "1"), "BR": ("alak", "1"),
        "RM": ("lokpjio", "1"), "ML": ("lakpiu", "1"),
        "PL": ("laktʃo", "1"), "AS": ("lakpjo", "1"),
        "PD": ("lokpjio", "1"), "SM": ("lakpjo", "1"), "BK": ("lokpjio", "1"),
    }),
    114: ("finger", "right", {
        "MN": ("lakkeŋ", "1"), "BR": ("latʃəŋ", "1"),
        "RM": ("lokʃəŋ", "1"), "ML": ("lakke", "1"),
        "PL": ("laktʃə", "1"), "AS": ("lakkeŋ", "1"),
        "PD": ("lakkeŋ", "1"), "SM": ("lakkeŋ", "1"), "BK": ("lokʃəŋ", "1"),
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
            pdf_page = "23" if item == 100 and code in {"MN", "BR", "RM"} else "24"
            printed_page = "19" if pdf_page == "23" else "20"
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": pdf_page,
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
    assert len(rows) == 15 * 9 == 135
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
