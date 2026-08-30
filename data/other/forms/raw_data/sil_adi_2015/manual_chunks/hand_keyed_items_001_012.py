#!/usr/bin/env python3
"""Write the visually checked Adi Appendix B cells for items 1--12.

The lexical strings below were keyed while reading the 400-dpi rendering of
physical PDF page 17.  The PDF text layer was consulted only as a character
input scaffold; every cell was accepted only after visual comparison.
"""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_001_012_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
    "text scaffold not accepted without cell visual match"
)
SITES = {
    "MN": "Minyong",
    "BR": "Bori",
    "RM": "Ramo",
    "ML": "Milang",
    "PL": "Pailibo",
    "AS": "Ashing (Bogum Bokang)",
    "PD": "Padam",
    "SM": "Shimong",
    "BK": "Bokar",
}

# item: (gloss, page column, {site: (form(s), cognate label(s))})
# Multiple responses within one conceptual cell are separated by `` | ``.
DATA = {
    1: ("sky", "left", {
        "MN": ("t̪aləŋ", "1"), "BR": ("t̪aləŋ", "1"),
        "RM": ("med̪omo", "2"), "ML": ("t̪allə", "1"),
        "PL": ("t̪allə", "1"), "AS": ("t̪aləŋ", "1"),
        "PD": ("t̪aləŋ", "1"), "SM": ("t̪aləŋ", "1"),
        "BK": ("med̪oŋmo", "2"),
    }),
    2: ("sun", "left", {
        "MN": ("d̪oŋi", "1"), "BR": ("d̪oŋi", "1"),
        "RM": ("d̪oŋi", "1"), "ML": ("məɾuŋ", "2"),
        "PL": ("d̪oŋi", "1"), "AS": ("d̪oŋi", "1"),
        "PD": ("d̪oŋi", "1"), "SM": ("d̪oŋi", "1"),
        "BK": ("d̪õɲi", "1"),
    }),
    3: ("moon", "left", {
        "MN": ("polo", "1"), "BR": ("polo", "1"),
        "RM": ("põlo", "1"), "ML": ("polu", "1"),
        "PL": ("polo", "1"), "AS": ("polo", "1"),
        "PD": ("polo", "1"), "SM": ("polo", "1"),
        "BK": ("põlo", "1"),
    }),
    4: ("star", "middle", {
        "MN": ("t̪akaɾ", "1"), "BR": ("t̪akəɾ", "1"),
        "RM": ("t̪akʌɾ", "1"), "ML": ("t̪akaɾ", "1"),
        "PL": ("t̪akəɾ", "1"), "AS": ("t̪akəɾ", "1"),
        "PD": ("t̪akaɾ", "1"), "SM": ("t̪akaɾ", "1"),
        "BK": ("t̪akaɾ", "1"),
    }),
    5: ("cloud", "middle", {
        "MN": ("d̪omuk", "1"), "BR": ("hapon", "2"),
        "RM": ("d̪omuk", "1"), "ML": ("amuk", "1"),
        "PL": ("d̪omuk", "1"), "AS": ("d̪omuk", "1"),
        "PD": ("d̪omuk", "1"), "SM": ("d̪omuk", "1"),
        "BK": ("d̪õmuk", "1"),
    }),
    6: ("rain", "middle", {
        "MN": ("pəd̪oŋ", "1"), "BR": ("pəd̪oŋ", "1"),
        "RM": ("med̪oŋ", "1"), "ML": ("badʒo", "2"),
        "PL": ("ɲid̪o", "3"), "AS": ("pəd̪oŋ", "1"),
        "PD": ("pəd̪oŋ", "1"), "SM": ("pəd̪oŋ", "1"),
        "BK": ("med̪oŋ", "1"),
    }),
    7: ("rainbow", "middle", {
        "MN": ("muɾeŋ", "1"), "BR": ("muɾe", "1"),
        "RM": ("uɾe", "1"), "ML": ("bəkəbəle", "2"),
        "PL": ("uɾe", "1"), "AS": ("mɯɾeŋ", "1"),
        "PD": ("muɾe", "1"), "SM": ("muɾeŋ", "1"),
        "BK": ("uɾe", "1"),
    }),
    8: ("wind", "middle", {
        "MN": ("d̪oji", "1"), "BR": ("d̪oji", "1"),
        "RM": ("ɲilu | ɲilu", "2 | 3"), "ML": ("ləluŋ", "2"),
        "PL": ("ilu", "3"), "AS": ("d̪oji", "1"),
        "PD": ("d̪oji | aʃaɾ", "1 | 4"), "SM": ("d̪oji", "1"),
        "BK": ("ɲuluŋ", "2"),
    }),
    9: ("lightning", "right", {
        "MN": ("jaɾi", "1"), "BR": ("joɾi", "1"),
        "RM": ("d̪ojak", "2"), "ML": ("maɾlɯŋkapən", "3"),
        "PL": ("d̪ojak", "2"), "AS": ("jaɾi", "1"),
        "PD": ("jaɾi", "1"), "SM": ("jaːɾi", "1"),
        "BK": ("d̪ojap", "2"),
    }),
    10: ("thunder", "right", {
        "MN": ("d̪omɯɾ", "1"), "BR": ("d̪omɯɾ", "1"),
        "RM": ("d̪oŋgum", "3"), "ML": ("dʒomaɾ", "1"),
        "PL": ("d̪obum", "2"), "AS": ("d̪omɯɾjaɾi", "1"),
        "PD": ("d̪omɯɾ", "1"), "SM": ("d̪oːmɯɾ", "1"),
        "BK": ("d̪oŋgum", "3"),
    }),
    11: ("sea", "right", {
        "MN": ("", "0"), "BR": ("", "0"),
        "RM": ("t̪əbənaiʃ", "1"), "ML": ("", "0"),
        "PL": ("", "0"), "AS": ("", "0"),
        "PD": ("ʃɯjəŋ", "2"), "SM": ("", "0"),
        "BK": ("", "0"),
    }),
    12: ("mountain", "right", {
        "MN": ("dit̪ə", "2"), "BR": ("ad̪i", "1"),
        "RM": ("d̪it̪uŋ", "2"), "ML": ("ade", "1"),
        "PL": ("ad̪i", "1"), "AS": ("ad̪i", "1"),
        "PD": ("d̪ɯt̪ə", "2"), "SM": ("d̪it̪ə", "2"),
        "BK": ("d̪it̪uŋ", "2"),
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
                "Site_Name": name, "PDF_Page": "17", "Printed_Page": "13",
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
    assert len(rows) == 12 * 9 == 108
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
