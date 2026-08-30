#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 56--70."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_056_070_hand_keyed.tsv"
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
    56: ("sugarcane", "left", {
        "MN": ("t̪abat̪", "1"), "BR": ("t̪abət̪", "1"),
        "RM": ("bəpi", "2"), "ML": ("pɯɾɯp", "3"),
        "PL": ("bapak", "2"), "AS": ("t̪abət̪", "1"),
        "PD": ("t̪abat̪", "1"), "SM": ("t̪abat̪", "1"), "BK": ("bapi", "2"),
    }),
    57: ("betelnut", "left", {
        "MN": ("", "0"), "BR": ("", "0"), "RM": ("t̪amul", "1"),
        "ML": ("", "0"), "PL": ("", "0"), "AS": ("", "0"),
        "PD": ("gujə", "2"), "SM": ("", "0"), "BK": ("", "0"),
    }),
    58: ("lime (for betelnut)", "left", {
        "MN": ("t̪aɲio", "1"), "BR": ("t̪aɲo", "1"),
        "RM": ("tʃuna", "2"), "ML": ("ɯŋʃun", "3"),
        "PL": ("t̪aɲo", "1"), "AS": ("ɲoku", "4"),
        "PD": ("tʃun", "2"), "SM": ("ɲoku", "4"), "BK": ("t̪aɲo", "1"),
    }),
    59: ("liquor", "left", {
        "MN": ("aʔoŋ", "1"), "BR": ("apoŋ", "1"),
        "RM": ("opõŋ | opõŋ", "1 | 2"), "ML": ("aju", "3"),
        "PL": ("opo", "2"), "AS": ("apoŋ", "1"),
        "PD": ("apoŋ", "1"), "SM": ("apoŋ", "1"), "BK": ("oh", "4"),
    }),
    60: ("milk", "left", {
        "MN": ("gakɯɾ", "1"), "BR": ("at̪ʃuŋ", "2"),
        "RM": ("om", "3"), "ML": ("", "0"), "PL": ("omo", "3"),
        "AS": ("gakɯɾ", "1"), "PD": ("gakɯɾ", "1"),
        "SM": ("aɲun", "4"), "BK": ("omə", "3"),
    }),
    61: ("oil", "middle", {
        "MN": ("t̪ulaŋ", "1"), "BR": ("t̪ulaŋ", "1"),
        "RM": ("t̪ulaŋ", "1"), "ML": ("t̪el", "2"),
        "PL": ("", "0"), "AS": ("t̪ulaŋ", "1"),
        "PD": ("t̪ulaŋ", "1"), "SM": ("t̪ulaŋ", "1"), "BK": ("t̪el", "2"),
    }),
    62: ("meat", "middle", {
        "MN": ("mənə", "1"), "BR": ("ad̪ɯn", "2"),
        "RM": ("id̪in", "2"), "ML": ("ad̪ɯn", "2"),
        "PL": ("ad̪in", "2"), "AS": ("ad̪ɯn", "2"),
        "PD": ("ad̪ɯn", "2"), "SM": ("mənə", "1"), "BK": ("id̪in", "2"),
    }),
    63: ("salt", "middle", {
        "MN": ("alo", "1"), "BR": ("alo", "1"), "RM": ("olo", "1"),
        "ML": ("t̪apu", "2"), "PL": ("alo", "1"), "AS": ("alo", "1"),
        "PD": ("alo", "1"), "SM": ("alo", "1"), "BK": ("olo", "1"),
    }),
    64: ("onion", "middle", {
        "MN": ("d̪ilap", "1"), "BR": ("", "0"), "RM": ("pɪjadʒ", "2"),
        "ML": ("d̪əlap", "1"), "PL": ("dʒakup", "3"), "AS": ("", "0"),
        "PD": ("d̪ilap", "1"), "SM": ("d̪ilap", "1"), "BK": ("dʒaʃuŋ", "3"),
    }),
    65: ("garlic", "middle", {
        "MN": ("", "0"), "BR": ("t̪alap", "1"),
        "RM": ("dʒəkukh", "2"), "ML": ("", "0"),
        "PL": ("t̪alap", "1"), "AS": ("t̪alap", "1"),
        "PD": ("", "0"), "SM": ("t̪alap", "1"), "BK": ("dʒakuk", "2"),
    }),
    66: ("red pepper, chilli", "right", {
        "MN": ("maɾhɯ", "1"), "BR": ("dʒaluk", "2"),
        "RM": ("dʒəlukh", "2"), "ML": ("maɾʃɯ", "1"),
        "PL": ("dʒaluk", "2"), "AS": ("maɾʃɯ", "1"),
        "PD": ("maɾtʃɯ", "1"), "SM": ("maɾhi", "1"), "BK": ("dʒaluk", "2"),
    }),
    67: ("elephant", "right", {
        "MN": ("hɯt̪ə", "1"), "BR": ("ʃitə", "1"),
        "RM": ("ʃɯt̪ə", "1"), "ML": ("ʃɯt̪a", "1"),
        "PL": ("ʃot̪ə", "1"), "AS": ("ʃɯt̪ə", "1"),
        "PD": ("ʃɯt̪a", "1"), "SM": ("hit̪e", "1"),
        "BK": ("mojiŋʃot̪ə", "2"),
    }),
    68: ("tiger", "right", {
        "MN": ("hɯmjo", "1"), "BR": ("ʃiɲĩo", "1"),
        "RM": ("ʃimjo", "1"), "ML": ("pat̪hɯ", "2"),
        "PL": ("ɲoɾe", "3"), "AS": ("ʃimijo", "1"),
        "PD": ("ʃimijo", "1"), "SM": ("himjio", "1"), "BK": ("ʃomjeo", "1"),
    }),
    69: ("bear", "right", {
        "MN": ("hɯt̪um", "2"), "BR": ("ʃit̪um", "2"),
        "RM": ("ʃut̪um", "2"), "ML": ("at̪ɯm", "1"),
        "PL": ("ʃot̪t̪um", "2"), "AS": ("ʃɯt̪um", "2"),
        "PD": ("ʃit̪um", "2"), "SM": ("hit̪um", "2"), "BK": ("ʃut̪um", "2"),
    }),
    70: ("deer", "right", {
        "MN": ("hɯd̪um", "2"), "BR": ("ʃid̪um", "2"),
        "RM": ("ʃud̪um", "2"), "ML": ("ad̪um", "1"),
        "PL": ("ʃod̪um", "2"), "AS": ("ʃɯd̪um", "2"),
        "PD": ("ʃid̪um", "2"), "SM": ("hid̪um", "2"), "BK": ("ʃud̪um", "2"),
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
                "Site_Name": name, "PDF_Page": "21", "Printed_Page": "17",
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
    assert len(rows) == 15 * 9 == 135
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
