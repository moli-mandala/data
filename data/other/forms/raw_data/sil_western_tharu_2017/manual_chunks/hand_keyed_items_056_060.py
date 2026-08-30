#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 56-60."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_056_060_hand_keyed.tsv")
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


def cell(form, labels, page=42, printed=37, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    56: ("smoke", {
        "HIN": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "RNK": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "RNS_Sisaikhara": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "BNM": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "DGC": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "DkR": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "TkN": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "BNT": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "RKM": cell("dʰũa", "1", page=41, printed=36, column="right"),
        "SkP": cell("dʰuwã", "1", page=41, printed=36, column="right"),
        "DDK": cell("dʰuwã", "1", page=41, printed=36, column="right"),
        "RKB": cell("dʰuã", "1", page=41, printed=36, column="right"),
        "DKS": cell("dʰuã", "1", page=41, printed=36, column="right"),
        "RNS_Sisana": cell("dʰuã", "1", page=41, printed=36, column="right"),
        "KkP": cell("dʰũwa", "1", page=41, printed=36, column="right"),
        "CCC": cell("dʰuːʌ", "1", page=41, printed=36, column="right"),
    }),
    57: ("ash", {
        "HIN": cell("ɾakʰ", "1", page=41, printed=36, column="right"),
        "RNK": cell("bʰua", "2", page=41, printed=36, column="right"),
        "RNS_Sisaikhara": cell("bʰua", "2", page=41, printed=36, column="right"),
        "DGC": cell("bʰua", "2", page=41, printed=36, column="right"),
        "TkN": cell("bʰua", "2", page=41, printed=36, column="right"),
        "DkR": cell("bʰui", "2", page=41, printed=36, column="right"),
        "DKS": cell("bʰui", "2", page=41, printed=36, column="right"),
        "RKB": cell("bʰũa", "2", page=41, printed=36, column="right"),
        "RKM": cell("bʰʊa", "2", page=41, printed=36, column="right"),
        "RNS_Sisana": cell("bʰʊa", "2", page=41, printed=36, column="right"),
        "DDK": cell("bʰʊa", "2", page=41, printed=36, column="right"),
        "SkP": cell("bʰʊa", "2", page=41, printed=36, column="right"),
        "KkP": cell("bʰuwa", "2", page=41, printed=36, column="right"),
        "BNM": cell("tʃʰaɾ", "3", page=41, printed=36, column="right"),
        "BNT": cell("tʃʰaɾ", "3", page=41, printed=36, column="right"),
        "CCC": cell("tʃʰaɖu", "3", page=41, printed=36, column="right"),
    }),
    58: ("mud", {
        "HIN": cell("miʈːi", "1", page=41, printed=36, column="right"),
        "RNK": cell("mʌʈːi", "1", page=41, printed=36, column="right"),
        "RNS_Sisaikhara": cell("mʌʈːi", "1", page=41, printed=36, column="right"),
        "BNM": cell("mʌʈːi", "1", page=41, printed=36, column="right"),
        "RKB": cell("mʌʈːi", "1", page=41, printed=36, column="right"),
        "RKM": cell("mʌʈːi", "1"),
        "RNS_Sisana": cell("mʌʈːi", "1"),
        "DGC": cell("maʈi", "1"),
        "DkR": cell("maʈi", "1"),
        "SkP": cell("maʈi", "1"),
        "DDK": cell("maʈi", "1"),
        "TkN": cell("miʈːi", "1"),
        "DKS": cell("maʈːi", "1"),
        "BNT": cell("mʌʈːɪ", "1"),
        "KkP": cell("kĩntʃʰa", "2"),
        "CCC": cell(None, "", page="41 / 42", printed="36 / 37", column="right / left"),
    }),
    59: ("dust", {
        "HIN": cell("dʰul", "1"),
        "DGC": cell("dʰuɾ", "1"),
        "DkR": cell("dʰuɾ", "1"),
        "SkP": cell("dʰuɾ", "1"),
        "DDK": cell("dʰuɾ", "1"),
        "KkP": cell("dʰuɾ", "1"),
        "DKS": cell("duɾa", "1"),
        "RNK": cell("dʰutʃʰãɾ", "2"),
        "RNS_Sisaikhara": cell("dʰudʰʌ̃ɾ", "2"),
        "BNM": cell("dʰudʌɾ", "2"),
        "TkN": cell("dʰudʰʌɾ", "2"),
        "BNT": cell("dʰudʰʌɾ", "2"),
        "RNS_Sisana": cell("dʰudʰʌɾ", "2"),
        "RKM": cell("dʰũdʰʌɾ", "2"),
        "RKB": cell("dʰũdʰʌɾ", "2"),
        "CCC": cell(None, ""),
    }),
    60: ("gold", {
        "HIN": cell("sona", "1"),
        "RNK": cell("sona", "1"),
        "RNS_Sisaikhara": cell("sona", "1"),
        "BNM": cell("sona", "1"),
        "BNT": cell("sona", "1"),
        "DkR": cell("son", "1"),
        "DKS": cell("son", "1"),
        "CCC": cell("son", "1"),
        "DDK": cell("son", "1"),
        "DGC": cell("son", "1"),
        "KkP": cell("son", "1"),
        "RKB": cell("sono", "1"),
        "TkN": cell("sono", "1"),
        "RKM": cell("sono", "1"),
        "RNS_Sisana": cell("sono", "1"),
        "SkP": cell("swan", "1"),
    }),
}


def main() -> None:
    rows = []
    for item in range(56, 61):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 78
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 78
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 73
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
