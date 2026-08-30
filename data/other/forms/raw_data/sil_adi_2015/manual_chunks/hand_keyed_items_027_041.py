#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 27--41."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_027_041_hand_keyed.tsv"
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
    27: ("year", "left", {
        "MN": ("d̪ɯt̪ak", "1"), "BR": ("d̪ɯt̪ak", "1"),
        "RM": ("eɲɯ", "2"), "ML": ("t̪aɾak", "3"),
        "PL": ("aɲɯ", "2"), "AS": ("d̪ɯt̪ak", "1"),
        "PD": ("d̪ɯt̪ak", "1"), "SM": ("d̪ɯt̪ak", "1"),
        "BK": ("ɲiŋ", "4"),
    }),
    28: ("day", "left", {
        "MN": ("loŋa", "1"), "BR": ("lo", "1"), "RM": ("alo", "1"),
        "ML": ("ane", "2"), "PL": ("alo", "1"), "AS": ("loŋə", "1"),
        "PD": ("loŋə", "1"), "SM": ("loŋə", "1"), "BK": ("lo", "1"),
    }),
    29: ("morning", "left", {
        "MN": ("ɾo", "1"), "BR": ("ɾo", "1"), "RM": ("aɾo", "1"),
        "ML": ("anap", "2"), "PL": ("aɾo", "1"),
        "AS": ("ɾokom", "1"), "PD": ("ɾo", "1"),
        "SM": ("ɾo", "1"), "BK": ("aɾo", "1"),
    }),
    30: ("noon", "left", {
        "MN": ("loŋa", "1"), "BR": ("d̪oɲɯ̃ kɯd̪ɯ", "2"),
        "RM": ("ʃɯjum", "3"), "ML": ("nəɾa", "4"),
        "PL": ("aloloji", "1"), "AS": ("loŋəjiɾaŋ", "1"),
        "PD": ("loŋəɾadʒaŋ", "1"), "SM": ("loŋəjiɾaŋ", "1"),
        "BK": ("lopoŋ", "1"),
    }),
    31: ("evening", "left", {
        "MN": ("jumd̪əŋ", "1"), "BR": ("jumə", "1"),
        "RM": ("ʃujum", "1"), "ML": ("ajem | ajem", "1 | 2"),
        "PL": ("ad̪um", "2"), "AS": ("jumə", "1"),
        "PD": ("ad̪əŋ", "2"), "SM": ("jumd̪əŋ", "1"),
        "BK": ("ajum | ajum", "1 | 2"),
    }),
    32: ("night", "left", {
        "MN": ("jo", "1"), "BR": ("jo", "1"), "RM": ("kənə", "2"),
        "ML": ("aju", "1"), "PL": ("kənə", "2"),
        "AS": ("jomaŋ", "1"), "PD": ("jo", "1"),
        "SM": ("jo", "1"), "BK": ("ajo", "1"),
    }),
    33: ("paddy rice", "middle", {
        "MN": ("ammo", "1"), "BR": ("anʃik", "2"), "RM": ("am", "1"),
        "ML": ("pimɾumu", "3"), "PL": ("amʃɯk | amʃɯk", "1 | 2"),
        "AS": ("ammo", "1"), "PD": ("am", "1"), "SM": ("ammo", "1"),
        "BK": ("amʃɯk | amʃɯk", "1 | 2"),
    }),
    34: ("uncooked rice", "middle", {
        "MN": ("d̪obɪn", "1"), "BR": ("abɯn", "1"),
        "RM": ("amə", "2"), "ML": ("d̪ukɯ", "3"),
        "PL": ("ambin", "1"), "AS": ("ambin", "1"),
        "PD": ("ambɯn", "1"), "SM": ("ambɯn", "1"), "BK": ("amə", "2"),
    }),
    35: ("cooked rice", "middle", {
        "MN": ("amah", "1"), "BR": ("amə | apin", "1 | 4"),
        "RM": ("akke", "2"), "ML": ("d̪una", "3"),
        "PL": ("d̪okke", "2"), "AS": ("amə | apin", "1 | 4"),
        "PD": ("apim", "4"), "SM": ("apin", "4"), "BK": ("akke", "2"),
    }),
    36: ("Wheat", "middle", {
        "MN": ("", "0"), "BR": ("", "0"), "RM": ("ompiɾ", "1"),
        "ML": ("", "0"), "PL": ("", "0"), "AS": ("", "0"),
        "PD": ("", "0"), "SM": ("gẽhu", "2"), "BK": ("omjiŋ", "1"),
    }),
    37: ("corn", "right", {
        "MN": ("həʔa", "1"), "BR": ("papo", "2"), "RM": ("pepo", "2"),
        "ML": ("ɾokʃin", "4"), "PL": ("t̪apu", "3"),
        "AS": ("ʃapə | ʃapə", "2 | 3"), "PD": ("ʃapa", "2"),
        "SM": ("həpa | həpa", "1 | 2"), "BK": ("pepo", "2"),
    }),
    38: ("potato", "right", {
        "MN": ("", "0"), "BR": ("alu", "1"), "RM": ("d̪obuɾ", "2"),
        "ML": ("alu", "1"), "PL": ("d̪obuɾ | popse", "2 | 3"),
        "AS": ("alu", "1"), "PD": ("alu", "1"),
        "SM": ("alugut̪i", "1"), "BK": ("poptʃe", "3"),
    }),
    39: ("cauliflower", "right", {
        "MN": ("", "0"), "BR": ("", "0"), "RM": ("kobi", "1"),
        "ML": ("", "0"), "PL": ("kobi", "1"), "AS": ("kobi", "1"),
        "PD": ("phulkobi", "1"), "SM": ("kobi", "1"), "BK": ("kobi", "1"),
    }),
    40: ("cabbage", "right", {
        "MN": ("", "0"), "BR": ("", "0"), "RM": ("kobi", "1"),
        "ML": ("", "0"), "PL": ("kobi", "1"), "AS": ("kobi", "1"),
        "PD": ("band̪akobi", "1"), "SM": ("kobi", "1"), "BK": ("kobi", "1"),
    }),
    41: ("eggplant", "right", {
        "MN": ("bajom", "1"), "BR": ("bajom", "1"),
        "RM": ("bajom", "1"), "ML": ("bajom", "1"),
        "PL": ("bajom", "1"), "AS": ("bajon", "1"),
        "PD": ("bajom", "1"), "SM": ("bajom", "1"), "BK": ("bajum", "1"),
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
            pdf_page = "18" if item == 27 and code != "BK" else "19"
            printed_page = "14" if pdf_page == "18" else "15"
            blank = not form
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
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
    assert len(rows) == 15 * 9 == 135
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
