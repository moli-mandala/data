#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 161-165."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_161_165_hand_keyed.tsv")
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


def cell(form, labels="1", page="60", printed="55", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/2400-dpi rendered-page crops before
# any comparison with the legacy CSV.
ITEMS = {
    161: ("eleven", {
        **{site: cell("gjaɾʌh") for site in SITES},
        "SkP": cell("ɪgjaɾʌh"),
        "RKB": cell("gaɾʌh"),
        "BNT": cell("gaɾʌh"),
        "CCC": cell(None),
    }),
    162: ("twelve", {
        **{site: cell("baɾʌh") for site in SITES},
        "SkP": cell("baɾʌhʌɾ"),
        "CCC": cell(None),
    }),
    163: ("twenty", {
        **{site: cell("bis") for site in SITES},
        "KkP": cell("bis", column="right"),
    }),
    164: ("hundred", {
        **{site: cell("sɔ", column="right") for site in SITES},
        "CCC": cell("sai", column="right"),
    }),
    165: ("who", {
        "HIN": cell("kɔn", column="right"),
        "BNM": cell("kɔn", column="right"),
        "RKB": cell("kɔn", column="right"),
        "BNT": cell("kɔn", column="right"),
        "RKM": cell("kɔn", column="right"),
        "RNS_Sisaikhara": cell("kɔn", column="right"),
        "RNK": cell("kɔːn", column="right"),
        "RNS_Sisana": cell("kɔːn", column="right"),
        "DGC": cell("kʌʊn / ke", labels="1 / 3", column="right"),
        "KkP": cell("kʌʊn", column="right"),
        "SkP": cell("kʊn", column="right"),
        "TkN": cell("kon", column="right"),
        "CCC": cell("kun", column="right"),
        "DKS": cell("ke", labels="3", column="right"),
        "DkR": cell("ke", labels="3", column="right"),
        "DDK": cell("ke", labels="3", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(161, 166):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 78
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 79
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 74
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
