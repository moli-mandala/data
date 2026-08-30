#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 151-155."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_151_155_hand_keyed.tsv")
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


def cell(form, labels="1", page="58", printed="53", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600/2400-dpi rendered-page crops
# before any comparison with the legacy CSV.
ITEMS = {
    151: ("one", {
        **{site: cell("ek", column="left") for site in SITES},
        "CCC": cell("ek"),
        "DDK": cell("ek"),
        "KkP": cell("ek"),
    }),
    152: ("two", {
        "HIN": cell("do"),
        "BNM": cell("do"),
        "BNT": cell("do"),
        "RNK": cell("dui", "2"),
        "RNS_Sisaikhara": cell("dui", "2"),
        "DGC": cell("dui", "2"),
        "DkR": cell("dui", "2"),
        "SkP": cell("dui", "2"),
        "RKB": cell("dui", "2"),
        "CCC": cell("dui", "2"),
        "KkP": cell("dui", "2"),
        "TkN": cell("dʊi", "2"),
        "DKS": cell("dʊi", "2"),
        "RKM": cell("dʊi", "2"),
        "RNS_Sisana": cell("dʊi", "2"),
        "DDK": cell("dʊi", "2"),
    }),
    153: ("three", {
        **{site: cell("tin") for site in SITES},
        "DGC": cell("ʈin"),
    }),
    154: ("four", {
        **{site: cell("tʃaɾ") for site in SITES},
        "CCC": cell("tʃaɾ", page="59", printed="54", column="left"),
        "DDK": cell("tʃaɾ", page="59", printed="54", column="left"),
        "KkP": cell("tʃaɾ", page="59", printed="54", column="left"),
    }),
    155: ("five", {
        "HIN": cell("pãntʃ", page="59", printed="54", column="left"),
        "RNS_Sisaikhara": cell("pãntʃ", page="59", printed="54", column="left"),
        "BNM": cell("pãntʃ", page="59", printed="54", column="left"),
        "DGC": cell("pãntʃ", page="59", printed="54", column="left"),
        "DkR": cell("pãntʃ", page="59", printed="54", column="left"),
        "SkP": cell("pãntʃ", page="59", printed="54", column="left"),
        "RKB": cell("pãntʃ", page="59", printed="54", column="left"),
        "TkN": cell("pãntʃ", page="59", printed="54", column="left"),
        "DKS": cell("pãntʃ", page="59", printed="54", column="left"),
        "BNT": cell("pãntʃ", page="59", printed="54", column="left"),
        "RNS_Sisana": cell("pãntʃ", page="59", printed="54", column="left"),
        "DDK": cell("pãntʃ", page="59", printed="54", column="left"),
        "KkP": cell("pãntʃ", page="59", printed="54", column="left"),
        "RNK": cell("patʃ", page="59", printed="54", column="left"),
        "RKM": cell("patʃ", page="59", printed="54", column="left"),
        "CCC": cell("paːtʃ", page="59", printed="54", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(151, 156):
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
