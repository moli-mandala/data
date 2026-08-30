#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 13--26."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_013_026_hand_keyed.tsv"
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

# item: (gloss, page column, cells). Cells are (form(s), cognate label(s)).
DATA = {
    13: ("water", "left", {
        "MN": ("aʃɯ", "1"), "BR": ("aʃi", "1"), "RM": ("iʃ", "1"),
        "ML": ("aʃɯ", "1"), "PL": ("isi", "1"), "AS": ("aʃɯ", "1"),
        "PD": ("aʃɯ", "1"), "SM": ("aʃɯ", "1"), "BK": ("isi", "1"),
    }),
    14: ("river", "left", {
        "MN": ("koɾoŋ", "1"), "BR": ("ʃijon", "3"),
        "RM": ("tʃot̪oŋ", "4"), "ML": ("koɾo", "1"),
        "PL": ("ʃit̪o", "3"), "AS": ("konə", "5"),
        "PD": ("koɾoŋ", "1"), "SM": ("koɾoŋ", "1"),
        "BK": ("iʃijumbuŋ", "2"),
    }),
    15: ("soil/ground", "left", {
        "MN": ("kedeŋ", "1"), "BR": ("ked̪e", "1"),
        "RM": ("ked̪e", "1"), "ML": ("kaɾ", "2"),
        "PL": ("ked̪e", "1"), "AS": ("kəd̪eŋ", "1"),
        "PD": ("kedeŋ", "1"), "SM": ("kədʒeŋ", "1"),
        "BK": ("kəd̪e", "1"),
    }),
    16: ("mud", "left", {
        "MN": ("hɯjuŋ", "1"), "BR": ("ʃijuŋ", "1"),
        "RM": ("ʃod̪o", "1"), "ML": ("sokh", "2"),
        "PL": ("sidʒa", "3"), "AS": ("d̪ejjuŋ", "1"),
        "PD": ("hijjuŋ | ʃujuŋ", "1 | 1"), "SM": ("hijjuŋ", "1"),
        "BK": ("ʃod̪o", "1"),
    }),
    17: ("dust", "left", {
        "MN": ("dʒekɯɾt̪amɯɾ", "1"), "BR": ("asukamuk", "2"),
        "RM": ("mɯd̪bu", "3"), "ML": ("amuk", "4"),
        "PL": ("mibbu", "3"), "AS": ("pəmuk", "4"),
        "PD": ("pəmuk", "4"), "SM": ("t̪akɯɾt̪amɯɾ", "1"),
        "BK": ("mid̪bu", "3"),
    }),
    18: ("stone", "middle", {
        "MN": ("əlɯŋ", "1"), "BR": ("əlɯŋ", "1"),
        "RM": ("ɯlɯŋ", "1"), "ML": ("d̪abu", "2"),
        "PL": ("ɯlɯ", "1"), "AS": ("ɯlɯŋ", "1"),
        "PD": ("əlɯŋ", "1"), "SM": ("d̪abu", "2"),
        "BK": ("ɯlɯŋ", "1"),
    }),
    19: ("sand", "middle", {
        "MN": ("hɯjɯ", "1"), "BR": ("ʃij", "2"),
        "RM": ("lɯjit̪", "4"), "ML": ("ʃapi", "3"),
        "PL": ("ʃili | ʃili | ʃili", "2 | 3 | 4"),
        "AS": ("ʃiji | ʃiji | ʃiji", "2 | 3 | 4"),
        "PD": ("bali", "5"), "SM": ("hɯjjə", "1"), "BK": ("ʃi", "2"),
    }),
    20: ("gold", "middle", {
        "MN": ("", "0"), "BR": ("", "0"),
        "RM": ("dʒət̪ət̪əɾbum", "1"), "ML": ("", "0"),
        "PL": ("ŋiseɾ", "2"), "AS": ("ʃuna", "3"),
        "PD": ("", "0"), "SM": ("", "0"), "BK": ("ʃəɾ", "2"),
    }),
    21: ("silver", "middle", {
        "MN": ("", "0"), "BR": ("", "0"),
        "RM": ("dʒet̪ət̪əɾbum", "1"), "ML": ("", "0"),
        "PL": ("ŋiseɾ", "2"), "AS": ("tʃand̪i", "3"),
        "PD": ("", "0"), "SM": ("", "0"), "BK": ("ɲi", "4"),
    }),
    22: ("today", "middle", {
        "MN": ("hɯlo", "1"), "BR": ("ʃɯlo", "1"),
        "RM": ("ʃolo", "1"), "ML": ("ɯnə", "2"),
        "PL": ("ʃɯlo", "1"), "AS": ("ʃilo", "1"),
        "PD": ("ʃilo", "1"), "SM": ("hilo", "1"), "BK": ("ʃolo", "1"),
    }),
    23: ("yesterday", "right", {
        "MN": ("məlo", "1"), "BR": ("məlo", "1"),
        "RM": ("mojo", "1"), "ML": ("banə", "2"),
        "PL": ("məlo", "1"), "AS": ("məlo", "1"),
        "PD": ("məlo", "1"), "SM": ("məlo", "1"), "BK": ("məjo", "1"),
    }),
    24: ("tomorrow", "right", {
        "MN": ("ɲampo", "1"), "BR": ("ɾoɾo", "2"),
        "RM": ("aɾo", "2"), "ML": ("tʃõnaph", "3"),
        "PL": ("aɾo", "2"), "AS": ("ɲampo", "1"),
        "PD": ("ɲampo", "1"), "SM": ("ɲampo", "1"), "BK": ("aɾe", "2"),
    }),
    25: ("week", "right", {
        "MN": ("", "0"), "BR": ("ɲopə", "1"),
        "RM": ("hopta", "2"), "ML": ("", "0"),
        "PL": ("ɲopə", "1"), "AS": ("", "0"),
        "PD": ("hopta", "2"), "SM": ("hopta", "2"), "BK": ("", "0"),
    }),
    26: ("month", "right", {
        "MN": ("polo", "1"), "BR": ("polo", "1"),
        "RM": ("põlo", "1"), "ML": ("polu", "1"),
        "PL": ("polo", "1"), "AS": ("polo", "1"),
        "PD": ("polo", "1"), "SM": ("polo", "1"), "BK": ("põlo", "1"),
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
            # Item 13 begins at the foot of p.17 and continues on p.18.
            pdf_page = "17" if item == 13 and code in {"MN", "BR"} else "18"
            printed_page = "13" if pdf_page == "17" else "14"
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
    assert len(rows) == 14 * 9 == 126
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
