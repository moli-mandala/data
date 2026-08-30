#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 96-100."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_096_100_hand_keyed.tsv")
DECLARATION = "hand-keyed-from-rendered-source; PDF-text-OCR-legacy-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; 900/1200-dpi "
    "crops used for every cell; PDF text/OCR/legacy not accepted"
)
SITES = (
    "BNM", "BNT", "RNK", "RNS_Sisaikhara", "RNS_Sisana", "RKM", "RKB",
    "TkN", "KkP", "SkP", "DKS", "DDK", "DGC", "DkR", "CCC", "HIN",
)
SOURCE_CODES = {
    "BNM": "BNM", "BNT": "BNT", "RNK": "RNK", "RNS_Sisaikhara": "RNS",
    "RNS_Sisana": "RNS", "RKM": "RkM", "RKB": "RKB", "TkN": "TkN",
    "KkP": "KkP", "SkP": "SkP", "DKS": "DKS", "DDK": "DDK",
    "DGC": "DGC", "DkR": "DkR", "CCC": "CCC", "HIN": "HIN",
}
OCCURRENCE = {site: "" for site in SITES}
OCCURRENCE.update({"RNS_Sisaikhara": "1", "RNS_Sisana": "2"})
FIELDS = [
    "Item", "Gloss", "Site_Key", "Source_Code", "Source_Code_Occurrence", "Scope",
    "PDF_Page", "Printed_Page", "Column", "Source_Group_Labels",
    "Manual_Transcription", "Manual_Form_Count", "Source_Qualifier", "Review_Status",
    "Confidence", "Site_Assignment_Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def cell(form, labels="1", page="49", printed="44", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    96: ("snake", {
        "HIN": cell("sãp", page="48", printed="43", column="right"),
        "RNK": cell("sãp", page="48", printed="43", column="right"),
        "RNS_Sisaikhara": cell("sãp", page="48", printed="43", column="right"),
        "BNM": cell("sãp", page="48", printed="43", column="right"),
        "SkP": cell("sãp", page="48", printed="43", column="right"),
        "RKB": cell("sãp", page="48", printed="43", column="right"),
        "TkN": cell("sãp", page="48", printed="43", column="right"),
        "DKS": cell("sãp", page="48", printed="43", column="right"),
        "BNT": cell("sãp", page="48", printed="43", column="right"),
        "RKM": cell("sãp", page="48", printed="43", column="right"),
        "RNS_Sisana": cell("sãp", page="48", printed="43", column="right"),
        "CCC": cell("saːp", page="48", printed="43", column="right"),
        "KkP": cell("samp", page="48", printed="43", column="right"),
        "DGC": cell("sʌpuwa", "2", page="48", printed="43", column="right"),
        "DkR": cell("sapuã", "2", page="48", printed="43", column="right"),
        "DDK": cell("sʌpua", "2", page="48", printed="43", column="right"),
    }),
    97: ("monkey", {
        "HIN": cell("bʌndʌɾ", page="48", printed="43", column="right"),
        "BNM": cell("bʌndʌɾ", page="48", printed="43", column="right"),
        "BNT": cell("bʌndʌɾ", page="48", printed="43", column="right"),
        "RNK": cell("bʌndʌɾa", page="48", printed="43", column="right"),
        "RNS_Sisaikhara": cell("bʌndʌɾa", page="48", printed="43", column="right"),
        "DGC": cell("bʌndʌɾa", page="48", printed="43", column="right"),
        "DkR": cell("bʌndʌɾa", page="48", printed="43", column="right"),
        "SkP": cell("bʌndʌɾa", page="48", printed="43", column="right"),
        "RKB": cell("bʌndʌɾa", page="48", printed="43", column="right"),
        "TkN": cell("bʌndʌɾa"),
        "DKS": cell("bʌndʌɾa"),
        "RKM": cell("bʌndʌɾa"),
        "RNS_Sisana": cell("bʌndʌɾa"),
        "DDK": cell("bʌndʌɾa"),
        "KkP": cell("bʌndʌɾa"),
        "CCC": cell("banʌɾ"),
    }),
    98: ("mosquito", {
        "HIN": cell("mʌtʃʰʌɾ"),
        "RNK": cell("mʌtʃʰʌɾ"),
        "RNS_Sisaikhara": cell("mʌtʃʰʌɾ"),
        "BNM": cell("mʌtʃʰʌɾ"),
        "RKB": cell("mʌtʃʰʌɾ"),
        "BNT": cell("mʌtʃʰʌɾ"),
        "TkN": cell("matʃʰʌɾ"),
        "RKM": cell("mʌtʃʰːʌɾ"),
        "RNS_Sisana": cell("mʌtʃʰːʌɾ"),
        "DGC": cell("mʌs", "2"),
        "DkR": cell("mʌs", "2"),
        "SkP": cell("mʌs", "2"),
        "DKS": cell("mʌs", "2"),
        "CCC": cell("mʌs", "2"),
        "DDK": cell("mas", "2"),
        "KkP": cell("mãsa", "2"),
    }),
    99: ("ant", {
        "HIN": cell("tʃĩtĩ"),
        "RNK": cell("tʃĩtĩ"),
        "RNS_Sisaikhara": cell("tʃĩtĩ"),
        "BNM": cell("tʃĩtĩ"),
        "BNT": cell("tʃĩtĩ"),
        "RKB": cell("tʃĩti"),
        "SkP": cell("tʃæ̃ti"),
        "TkN": cell("tʃiti"),
        "RKM": cell("tʃiti"),
        "RNS_Sisana": cell("tʃiti"),
        "KkP": cell("tʃʰeⁱnti"),
        "CCC": cell("tʃihuti"),
        "DkR": cell("tʃimʈa", "2"),
        "DGC": cell("tʃimʈa", "2"),
        "DDK": cell("tʃimʈa", "2"),
        "DKS": cell("tʃimʌʈ", "2"),
    }),
    100: ("spider", {
        "HIN": cell("mʌkʌɾi"),
        "RNK": cell("mʌkʌɾa"),
        "RNS_Sisaikhara": cell("mʌkʌɾa"),
        "DkR": cell("mʌkʌɾa"),
        "SkP": cell("mʌkʌɾa"),
        "RKB": cell(
            "mʌkʌɾa / dʒara", "1 / 2", column="left / right",
            qualifier="second response: (web)",
        ),
        "DKS": cell("mʌkʌɾa"),
        "BNT": cell("mʌkʌɾa"),
        "RKM": cell("mʌkʌɾa"),
        "RNS_Sisana": cell("mʌkʌɾa", column="right"),
        "DDK": cell("mʌkʌɾa", column="right"),
        "BNM": cell("mʌkːʌɾi", column="right"),
        "DGC": cell("mokʌɾa", column="right"),
        "TkN": cell("mʌkʌɾija", column="right"),
        "CCC": cell("makara", column="right"),
        "KkP": cell("tʃʰiŋɡoɾa", "3", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(96, 101):
        gloss, cells = ITEMS[item]
        assert set(cells) == set(SITES)
        for site in SITES:
            form, labels, page, printed, column, qualifier = cells[site]
            blank = form is None
            uncertainty = ""
            site_confidence = "high"
            if site.startswith("RNS_"):
                site_confidence = "medium"
                uncertainty = (
                    "duplicate source code RNS; within-group occurrence order assigned to "
                    "metadata row order; unmatched extra group occurrence assigned to "
                    "metadata row 1 (Sisaikhara)"
                )
            if blank:
                uncertainty = "site code absent from the complete printed item block"
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": SOURCE_CODES[site],
                "Source_Code_Occurrence": OCCURRENCE[site],
                "Scope": "control" if site == "HIN" else "target",
                "PDF_Page": page, "Printed_Page": printed, "Column": column,
                "Source_Group_Labels": labels, "Manual_Transcription": form or "",
                "Manual_Form_Count": str(len(form.split(" / "))) if form else "0",
                "Source_Qualifier": qualifier,
                "Review_Status": "source_blank" if blank else "attested",
                "Confidence": "high", "Site_Assignment_Confidence": site_confidence,
                "Uncertainty": uncertainty, "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 80
    assert sum(row["Review_Status"] == "attested" for row in rows) == 80
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 0
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 76
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
