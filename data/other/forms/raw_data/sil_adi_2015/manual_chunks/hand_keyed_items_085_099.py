#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 85--99."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_085_099_hand_keyed.tsv"
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
    85: ("pig", "left", {
        "MN": ("əjək", "2"), "BR": ("əjək", "2"), "RM": ("əjək", "2"),
        "ML": ("ajek", "2"), "PL": ("əjək", "2"), "AS": ("əjək", "2"),
        "PD": ("eek", "1"), "SM": ("eek", "1"), "BK": ("əjək", "2"),
    }),
    86: ("rat", "left", {
        "MN": ("kəbuŋ", "1"), "BR": ("kəbuŋ", "1"),
        "RM": ("kubuŋ | kubuŋ", "1 | 2"), "ML": ("gabuŋ", "1"),
        "PL": ("kobu", "2"), "AS": ("kəbuŋ", "1"),
        "PD": ("kəbuŋ", "1"), "SM": ("kəbuŋ", "1"),
        "BK": ("kobuŋ | kobuŋ", "1 | 2"),
    }),
    87: ("chicken", "left", {
        "MN": ("pəɾok", "1"), "BR": ("pəɾok", "1"),
        "RM": ("poɾok", "1"), "ML": ("atʃu", "2"),
        "PL": ("poɾok", "1"), "AS": ("pəɾok", "1"),
        "PD": ("pəɾok", "1"), "SM": ("pəɾok", "1"), "BK": ("poɾok", "1"),
    }),
    88: ("egg", "left", {
        "MN": ("ɾokʔɯ", "1"), "BR": ("pəpɯ", "3"),
        "RM": ("pɯpɯ", "3"), "ML": ("tʃitʃi", "2"),
        "PL": ("pɯpɯ", "3"), "AS": ("ɾokpɯ", "1"),
        "PD": ("ɾokpɯ", "1"), "SM": ("ɾokpɯ", "1"), "BK": ("pɯpɯ", "3"),
    }),
    89: ("fish", "left", {
        "MN": ("əŋo", "1"), "BR": ("əŋo", "1"), "RM": ("õŋo", "1"),
        "ML": ("aŋu", "1"), "PL": ("məne", "2"), "AS": ("əŋo", "1"),
        "PD": ("əŋo", "1"), "SM": ("əŋo", "1"), "BK": ("õŋo", "1"),
    }),
    90: ("duck", "middle", {
        "MN": ("pədʒap", "1"), "BR": ("pədʒap", "1"),
        "RM": ("hãʃ", "2"), "ML": ("pədʒap", "1"),
        "PL": ("pədʒap", "1"), "AS": ("pədʒap", "1"),
        "PD": ("pədʒap", "1"), "SM": ("pədʒap", "1"), "BK": ("bat̪ok", "3"),
    }),
    91: ("bird", "middle", {
        "MN": ("pət̪t̪aŋ", "1"), "BR": ("pət̪aŋ", "1"),
        "RM": ("pət̪aŋ", "1"), "ML": ("t̪apiu", "2"),
        "PL": ("pət̪a", "1"), "AS": ("pət̪t̪aŋ", "1"),
        "PD": ("pət̪t̪aŋ", "1"), "SM": ("pət̪t̪aŋ", "1"),
        "BK": ("pət̪aŋ", "1"),
    }),
    92: ("insect", "middle", {
        "MN": ("t̪aʔumt̪aɾuk", "2"), "BR": ("t̪apum", "1"),
        "RM": ("t̪əpum", "1"), "ML": ("t̪apum", "1"),
        "PL": ("apum", "1"), "AS": ("t̪apum", "1"),
        "PD": ("t̪akomt̪aɾi", "2"), "SM": ("t̪akomt̪aɾi", "2"),
        "BK": ("t̪apum", "1"),
    }),
    93: ("cockroach", "middle", {
        "MN": ("t̪akʔha", "1"), "BR": ("", "0"),
        "RM": ("tʃatʃɯbaj", "4"), "ML": ("gabuŋnəbaŋ", "3"),
        "PL": ("tʃapʃiabi", "4"), "AS": ("t̪aːʃi", "2"),
        "PD": ("t̪akʃi", "2"), "SM": ("t̪akʃi", "2"),
        "BK": ("pĩanẽkopkop", "5"),
    }),
    94: ("bee", "middle", {
        "MN": ("t̪aŋut̪", "1"), "BR": ("t̪aŋut̪", "1"),
        "RM": ("t̪əjt̪uŋ", "2"), "ML": ("t̪au", "3"),
        "PL": ("t̪aɲit̪", "1"), "AS": ("t̪aŋut̪", "1"),
        "PD": ("t̪aŋut̪", "1"), "SM": ("t̪aŋut̪", "1"), "BK": ("t̪ũŋ", "2"),
    }),
    95: ("fly", "right", {
        "MN": ("t̪ajiŋ", "1"), "BR": ("t̪amit̪", "1"),
        "RM": ("t̪əmit̪", "1"), "ML": ("amat̪h", "2"),
        "PL": ("t̪aji", "1"), "AS": ("t̪amit̪", "1"),
        "PD": ("t̪amit̪", "1"), "SM": ("t̪amit̪", "1"), "BK": ("t̪amit̪", "1"),
    }),
    96: ("spider", "right", {
        "MN": ("t̪aɾum", "1"), "BR": ("t̪aːɾun", "1"),
        "RM": ("t̪aɾumsom", "1"), "ML": ("poput̪aɾam", "1"),
        "PL": ("t̪aɾum", "1"), "AS": ("t̪aɾun", "1"),
        "PD": ("mopɯt̪aɾum", "1"), "SM": ("popɯt̪aɾum", "1"),
        "BK": ("t̪aːɾun", "1"),
    }),
    97: ("ant", "right", {
        "MN": ("t̪aɾuk", "1"), "BR": ("t̪aɾuk", "1"),
        "RM": ("t̪əɾuk", "1"), "ML": ("paŋkəɾ", "2"),
        "PL": ("t̪aɾuk", "1"), "AS": ("t̪aɾuk", "1"),
        "PD": ("t̪aɾuk", "1"), "SM": ("t̪aɾuk", "1"), "BK": ("t̪aɾuk", "1"),
    }),
    98: ("mosquito", "right", {
        "MN": ("t̪ahuɾuŋgu", "1"), "BR": ("t̪aɾutʃuŋgu", "1"),
        "RM": ("t̪əɾuŋ", "1"), "ML": ("t̪aɾuʃuŋgu", "1"),
        "PL": ("t̪aɾu", "1"), "AS": ("t̪aɾutʃuŋgu", "1"),
        "PD": ("t̪aɾuhuŋhu", "1"), "SM": ("t̪aɾuhuŋhu", "1"),
        "BK": ("t̪amit̪", "2"),
    }),
    99: ("head", "right", {
        "MN": ("t̪ukku", "1"), "BR": ("d̪uppoŋ", "2"),
        "RM": ("d̪umpɯɾ", "2"), "ML": ("d̪umpo", "2"),
        "PL": ("d̪umpo", "2"), "AS": ("d̪umpoŋ", "2"),
        "PD": ("t̪ukku", "1"), "SM": ("d̪umpoŋ", "2"), "BK": ("d̪uppɯɾ", "2"),
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
            pdf_page = "22" if item == 85 and code in {"MN", "BR", "RM", "ML", "PL"} else "23"
            printed_page = "18" if pdf_page == "22" else "19"
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
