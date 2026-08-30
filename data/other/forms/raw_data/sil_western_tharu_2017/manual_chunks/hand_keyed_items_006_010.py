#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 6-10."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_006_010_hand_keyed.tsv")
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


def cell(form, labels, page, printed, column, qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1200-dpi crops before legacy reconciliation.
# Parenthetical comparison numbers remain literal qualifiers, not analysis.
ITEMS = {
    6: ("ear", {site: cell("kan", "1", 32, 27, "left") for site in SITES}),
    7: ("nose", {site: cell("nak", "1", 32, 27, "right") for site in SITES}),
    8: ("mouth", {
        "HIN": cell("mũh", "1", 32, 27, "right"),
        "RNS_Sisaikhara": cell("mũh", "1", 32, 27, "right"),
        "BNT": cell("mũh", "1", 32, 27, "right", "(4)"),
        "DkR": cell("mũh", "1", 32, 27, "right", "(4)"),
        "DDK": cell("mũh", "1", 32, 27, "right", "(4)"),
        "BNM": cell("moh", "1", 32, 27, "right", "(4)"),
        "RKM": cell("moh", "1", 32, 27, "right", "(4)"),
        "SkP": cell("mʊh", "1", 32, 27, "right", "(4)"),
        "RNS_Sisana": cell("mʊh", "1", 32, 27, "right", "(4)"),
        "RKB": cell("muh", "1", 32, 27, "right", "(4)"),
        "TkN": cell("muh", "1", 32, 27, "right", "(4)"),
        "DKS": cell("muh", "1", 32, 27, "right", "(4)"),
        "RNK": cell("muh", "1", 32, 27, "right", "(4)"),
        "CCC": cell("muːhʌ", "1", 32, 27, "right"),
        "DGC": cell("mʊh", "1", 32, 27, "right"),
        "KkP": cell("mũ", "1", 32, 27, "right", "(4)"),
    }),
    9: ("teeth", {
        "HIN": cell("dãnt", "1", 32, 27, "right"),
        "RNS_Sisaikhara": cell("dãnt", "1", 32, 27, "right"),
        "RNK": cell("dãnt", "1", 32, 27, "right"),
        "DGC": cell("dãnt", "1", 32, 27, "right"),
        "DkR": cell("dãnt", "1", 32, 27, "right"),
        "RKB": cell("dãnt", "1", 32, 27, "right"),
        "DKS": cell("dãnt", "1", 32, 27, "right"),
        "BNT": cell("dãnt", "1", 32, 27, "right"),
        "RNS_Sisana": cell("dãnt", "1", 32, 27, "right"),
        "DDK": cell("dãnt", "1", 32, 27, "right"),
        "KkP": cell("dãnt", "1", 32, 27, "right"),
        "BNM": cell("dand", "1", 32, 27, "right"),
        "SkP": cell("daːt", "1", 32, 27, "right"),
        "TkN": cell("daːt", "1", 32, 27, "right"),
        "RKM": cell("daːt", "1", 32, 27, "right"),
        "CCC": cell("daːt", "1", 32, 27, "right"),
    }),
    10: ("tongue", {
        site: cell("dʒibʰi" if site == "CCC" else "dʒibʰ", "1", 33, 28, "left")
        for site in SITES
    }),
}


def main() -> None:
    rows = []
    for item in range(6, 11):
        gloss, cells = ITEMS[item]
        assert set(cells) == set(SITES)
        for site in SITES:
            form, labels, page, printed, column, qualifier = cells[site]
            uncertainty = ""
            site_confidence = "high"
            if site.startswith("RNS_"):
                site_confidence = "medium"
                uncertainty = (
                    "duplicate source code RNS; occurrence 1/2 assigned to metadata row 1/2 "
                    "(Sisaikhara/Sisana)"
                )
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": SOURCE_CODES[site],
                "Source_Code_Occurrence": OCCURRENCE[site],
                "Scope": "control" if site == "HIN" else "target",
                "PDF_Page": page, "Printed_Page": printed, "Column": column,
                "Source_Group_Labels": labels, "Manual_Transcription": form,
                "Manual_Form_Count": str(len(form.split(" / "))),
                "Source_Qualifier": qualifier, "Review_Status": "attested",
                "Confidence": "high", "Site_Assignment_Confidence": site_confidence,
                "Uncertainty": uncertainty, "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 80
    assert sum(row["Review_Status"] == "attested" for row in rows) == 80
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 80
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
