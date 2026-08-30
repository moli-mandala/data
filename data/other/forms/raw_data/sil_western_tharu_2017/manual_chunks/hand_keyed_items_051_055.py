#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 51-55."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_051_055_hand_keyed.tsv")
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


def cell(form, labels, page=41, printed=36, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 900/1200-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    51: ("wind", {
        "HIN": cell("hʌwa", "1", page=40, printed=35, column="right"),
        "BNM": cell("hʌwa", "1", page=40, printed=35, column="right"),
        "DkR": cell("hʌwa", "1", page=40, printed=35, column="right"),
        "SkP": cell("hʌw", "1", page=40, printed=35, column="right"),
        "CCC": cell("hawa", "1", page=40, printed=35, column="right"),
        "RNK": cell("bjaɾ", "2", page=40, printed=35, column="right"),
        "RNS_Sisaikhara": cell("bjaɾ", "2", page=40, printed=35, column="right"),
        "RKB": cell("bjaɾ", "2", page=40, printed=35, column="right"),
        "RKM": cell("bjaɾ", "2", page=40, printed=35, column="right"),
        "RNS_Sisana": cell("bjaɾ", "2", page=40, printed=35, column="right"),
        "TkN": cell("biaɾ", "2", page=40, printed=35, column="right"),
        "DKS": cell("bʌjal", "2", page=40, printed=35, column="right"),
        "DDK": cell("bʌjal", "2", page=40, printed=35, column="right"),
        "DGC": cell("bʌjaɾ", "2", page=40, printed=35, column="right"),
        "KkP": cell("bajaɾ", "2", page=40, printed=35, column="right"),
        "BNT": cell(None, "", page=40, printed=35, column="right"),
    }),
    52: ("stone", {
        "HIN": cell("pʌtʰːʌɾ", "1", page=40, printed=35, column="right"),
        "RNK": cell("pʌtʰːʌɾ", "1", page=40, printed=35, column="right"),
        "RNS_Sisaikhara": cell("pʌtʰːʌɾ", "1", page=40, printed=35, column="right"),
        "BNM": cell("pʌtʰːʌɾ", "1", page=40, printed=35, column="right"),
        "DGC": cell("pʌtʰʌɾa", "1", page=40, printed=35, column="right"),
        "RNS_Sisana": cell("pʌtʰʌɾa", "1", page=40, printed=35, column="right"),
        "DDK": cell("pʌtʰʌɾa / dũŋɡa", "1 / 2", page="40 / 41", printed="35 / 36", column="right / left"),
        "DkR": cell("pʌtʰːʌɾa", "1", page=40, printed=35, column="right"),
        "SkP": cell("pʌtʰɾa", "1", page=40, printed=35, column="right"),
        "RKB": cell("pʌtʰʌɾ", "1", page=40, printed=35, column="right"),
        "TkN": cell("pʌtːʌɾ", "1"),
        "DKS": cell("pʌtʌjʌɾa", "1"),
        "BNT": cell("pʌtʰʌɾija", "1"),
        "RKM": cell("pʌtʰːʌɾ", "1"),
        "CCC": cell("pataɾa", "1"),
        "KkP": cell("pʌtʰaɾa", "1"),
    }),
    53: ("path", {
        "HIN": cell("ɾasta / maɾɡ", "1 / 5"),
        "RNS_Sisaikhara": cell("ɾasta / ɾʌtːa", "1 / 3"),
        "RNK": cell("ɾʌtːa / ɾʌtːa", "1 / 3"),
        "RNS_Sisana": cell("ɾaha", "3"),
        "BNM": cell("ɾʌsta", "1"),
        "DGC": cell("ɖʌɡʌɾa", "2"),
        "DkR": cell("ɖʌɡʌɾ", "2"),
        "SkP": cell("ɖʌɡʌɾ", "2"),
        "RKB": cell("ɖʌɡʌɾ", "2"),
        "DKS": cell("ɖʌɡʌɾ", "2"),
        "DDK": cell("ɖʌɡʌɾ", "2"),
        "KkP": cell("daɡaɾ", "2"),
        "TkN": cell("ɾʌha", "3"),
        "RKM": cell("ɾah", "3"),
        "CCC": cell("paheːɖa", "4"),
        "BNT": cell(None, ""),
    }),
    54: ("sand", {
        "HIN": cell("balu / ɾet", "1 / 2"),
        "DGC": cell("balu", "1"),
        "DkR": cell("balu", "1"),
        "DKS": cell("balu", "1"),
        "CCC": cell("balu", "1"),
        "DDK": cell("balu", "1"),
        "SkP": cell("baɾu", "1"),
        "KkP": cell("baɾu", "1"),
        "TkN": cell("ɾet", "2"),
        "RNK": cell("ɾeta", "2"),
        "RNS_Sisaikhara": cell("ɾeta", "2"),
        "BNM": cell("ɾeta", "2"),
        "BNT": cell("ɾeta", "2"),
        "RKM": cell("ɾeta", "2"),
        "RNS_Sisana": cell("ɾeta", "2"),
        "RKB": cell("ɾota", "2"),
    }),
    55: ("fire", {
        "HIN": cell("aɡ", "1"),
        "BNM": cell("aɡ", "1"),
        "RNK": cell("aɡi", "1"),
        "RNS_Sisaikhara": cell("aɡi", "1"),
        "DGC": cell("aɡi", "1"),
        "DkR": cell("aɡi", "1", column="right"),
        "SkP": cell("aɡi", "1", column="right"),
        "RKB": cell("aɡi", "1", column="right"),
        "TkN": cell("aɡi", "1", column="right"),
        "DKS": cell("aɡi", "1", column="right"),
        "BNT": cell("aɡi", "1", column="right"),
        "RKM": cell("aɡi", "1", column="right"),
        "RNS_Sisana": cell("aɡi", "1", column="right"),
        "CCC": cell("aɡi", "1", column="right"),
        "DDK": cell("aɡi", "1", column="right"),
        "KkP": cell("aɡi", "1", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(51, 56):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 83
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
