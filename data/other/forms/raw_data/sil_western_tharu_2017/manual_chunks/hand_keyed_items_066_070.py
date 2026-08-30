#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 66-70."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_066_070_hand_keyed.tsv")
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


def cell(form, labels, page=43, printed=38, column="right", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    66: ("fruit", {
        "HIN": cell("pʰʌl", "1", column="left"),
        "RNK": cell("pʰʌl", "1", column="left"),
        "RNS_Sisaikhara": cell("pʰʌl", "1", column="left"),
        "BNM": cell("pʰʌl", "1", column="left"),
        "DkR": cell("pʰʌl", "1", column="left"),
        "SkP": cell("pʰʌl", "1", column="left"),
        "TkN": cell("pʰʌl", "1", column="left"),
        "BNT": cell("pʰʌl", "1", column="left"),
        "RNS_Sisana": cell("pʰʌl", "1", column="left"),
        "CCC": cell("pʰʌl", "1", column="left"),
        "RKM": cell("pʰʌɾa", "1", column="left"),
        "DGC": cell("pʰaɾa", "1", column="left"),
        "KkP": cell("pʰaɾa", "1", column="left"),
        "RKB": cell("bʰʌɾa", "1", column="left"),
        "DKS": cell("bʰaɾa", "1", column="left"),
        "DDK": cell(None, "", column="left"),
    }),
    67: ("mango", {
        "HIN": cell("am", "1", column="left"),
        "RNK": cell("am", "1"),
        "RNS_Sisaikhara": cell("am", "1"),
        "BNM": cell("am", "1"),
        "DGC": cell("am", "1"),
        "DkR": cell("am", "1"),
        "SkP": cell("am", "1"),
        "RKB": cell("am", "1"),
        "TkN": cell("am", "1"),
        "DKS": cell("am", "1"),
        "BNT": cell("am", "1"),
        "RKM": cell("am", "1"),
        "RNS_Sisana": cell("am", "1"),
        "CCC": cell("am", "1"),
        "DDK": cell("am", "1"),
        "KkP": cell("amb", "1"),
    }),
    68: ("banana", {
        "HIN": cell("kela", "1"),
        "BNT": cell("kela", "1"),
        "DGC": cell("keɾa", "1"),
        "DkR": cell("keɾa", "1"),
        "TkN": cell("keɾa", "1"),
        "DKS": cell("keɾa", "1"),
        "CCC": cell("keɾa", "1"),
        "DDK": cell("keɾa", "1"),
        "SkP": cell("kjaɾa", "1"),
        "RNS_Sisaikhara": cell("tʃʰijã", "2"),
        "RKM": cell("tʃʰijã", "2"),
        "RNK": cell("tʃʰijã", "2"),
        "RKB": cell("tʃʰija", "2"),
        "RNS_Sisana": cell("tʃʰija", "2"),
        "KkP": cell("tʃʰia", "2"),
        "BNM": cell("ɡeɾkibʰʌɾi", "3"),
    }),
    69: ("wheat", {
        "HIN": cell("ɡehũ", "1"),
        "RNK": cell("ɡehũ", "1"),
        "RNS_Sisaikhara": cell("ɡehũ", "1"),
        "BNM": cell("ɡehũ", "1"),
        "BNT": cell("ɡehũ", "1"),
        "RKM": cell("ɡehũ", "1"),
        "KkP": cell("ɡehũ", "1"),
        "DGC": cell("ɡohũ", "1"),
        "DkR": cell("ɡohũ", "1"),
        "DDK": cell("ɡohũ", "1"),
        "SkP": cell("ɡʊhõ", "1"),
        "DKS": cell("ɡʊhõ", "1"),
        "RKB": cell("ɡehõ", "1"),
        "TkN": cell("ɡehu", "1"),
        "RNS_Sisana": cell("ɡehõ", "1"),
        "CCC": cell(None, ""),
    }),
    70: ("millet", {
        "HIN": cell("dʒʌvaɾ / dʒɔ", "1 / 4", page="43 / 44", printed="38 / 39", column="right / left"),
        "DkR": cell("dʒolʌɾi", "1"),
        "SkP": cell("dʒwaɾ", "1", page=44, printed=39, column="left"),
        "RKM": cell("dʒwaɾ", "1", page=44, printed=39, column="left"),
        "RNS_Sisaikhara": cell("dʒwaɾ / tʃʊɾi", "1 / 2", page=44, printed=39, column="left"),
        "TkN": cell("dʒʌbʌɾ", "1", page=44, printed=39, column="left"),
        "RNK": cell("tʃʌɾi", "2", page=44, printed=39, column="left"),
        "RKB": cell("tʃʌɾi", "2", page=44, printed=39, column="left"),
        "BNT": cell("tʃʌɾe", "2", page=44, printed=39, column="left"),
        "BNM": cell("bʌɾe", "3", page=44, printed=39, column="left"),
        "KkP": cell("dʒoᵘ", "4", page=44, printed=39, column="left"),
        "DKS": cell("dʒaᵘ", "4", page=44, printed=39, column="left"),
        "CCC": cell("koːdo", "5", page=44, printed=39, column="left"),
        "DDK": cell("bʌdʒʌɾa", "6", page=44, printed=39, column="left"),
        "DGC": cell("bʌdʒʌɾa", "6", page=44, printed=39, column="left"),
        "RNS_Sisana": cell(None, "", page="43 / 44", printed="38 / 39", column="right / left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(66, 71):
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
            if blank and not site.startswith("RNS_"):
                uncertainty = "site code absent from the complete printed item block"
            if blank and site.startswith("RNS_"):
                uncertainty += "; no second RNS occurrence in the complete printed item block"
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 77
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 3
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 79
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
