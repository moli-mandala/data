#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 156-160."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_156_160_hand_keyed.tsv")
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


def cell(form, labels="1", page="59", printed="54", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600/2400-dpi rendered-page crops
# before any comparison with the legacy CSV.
ITEMS = {
    156: ("six", {
        "HIN": cell("tʃʰe"),
        "RNK": cell("tʃʰe"),
        "RNS_Sisaikhara": cell("tʃʰe"),
        "BNM": cell("tʃʰe"),
        "DGC": cell("tʃʰe"),
        "DkR": cell("tʃʰe"),
        "SkP": cell("tʃʰe"),
        "RKB": cell("tʃʰe"),
        "DKS": cell("tʃʰe"),
        "BNT": cell("tʃʰe"),
        "KkP": cell("tʃʰe"),
        "TkN": cell("tʃʰʌ"),
        "RNS_Sisana": cell("tʃʰʌ"),
        "RKM": cell("tʃʰʌⁱ"),
        "CCC": cell("tʃʰo"),
        "DDK": cell("tʃʰɔᶸ"),
    }),
    157: ("seven", {
        **{site: cell("sat") for site in SITES},
        "DDK": cell("sat", column="right"),
        "KkP": cell("sat", column="right"),
        "DGC": cell("saʈ", column="right"),
    }),
    158: ("eight", {
        **{site: cell("aʈʰ", column="right") for site in SITES},
        "CCC": cell("at", column="right"),
    }),
    159: ("nine", {
        **{site: cell("nɔ", column="right") for site in SITES},
        "CCC": cell("nou", column="right"),
    }),
    160: ("ten", {
        **{site: cell("dʌs", column="right") for site in SITES},
        "KkP": cell("dʌs", page="60", printed="55", column="left"),
        "DGC": cell("ɖʌs", page="60", printed="55", column="left"),
        "CCC": cell("das", page="60", printed="55", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(156, 161):
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
                    "metadata row order"
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
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 80
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 75
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
