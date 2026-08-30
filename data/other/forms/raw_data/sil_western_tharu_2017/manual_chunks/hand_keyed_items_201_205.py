#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 201-205."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_201_205_hand_keyed.tsv")
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


def cell(form, labels="1", page="67", printed="62", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1800-dpi rendered-page crops before any
# comparison with the legacy TSV. Item 205 continues onto physical p.68.
ITEMS = {
    201: ("he sees; he saw", {
        "HIN": cell("dekʰta / dekʰa"),
        "BNM": cell("dekʰta / dekʰa"),
        "RNK": cell("dekʰʌthæ / dekʰi"),
        "RNS_Sisaikhara": cell("dekʰʌthæ / dekʰi"),
        "DGC": cell(
            "dekʰnu / her", labels="1 / 3",
            qualifier="second response followed by (see)"
        ),
        "DkR": cell(
            "dekʰun / her", labels="1 / 3",
            qualifier=(
                "first response followed by (look); second response followed by (see)"
            ),
        ),
        "RKB": cell("dekhʌto / dekhʌlɔ"),
        "TkN": cell("dekʰho"),
        "DKS": cell("dekʰʌnʊ / dekʰʌli"),
        "BNT": cell("dekʰʌta / dekʰu"),
        "RKM": cell("dekʰ / dekʰo"),
        "RNS_Sisana": cell("dekʰle / dekʰo"),
        "DDK": cell("dekʰʌt / dekʰʌlæ"),
        "KkP": cell("dekʰʌl"),
        "SkP": cell("dʌjakʰo / dʌjakʌl", labels="2"),
        "CCC": cell("herʌi", labels="3"),
    }),
    202: ("I", {
        "HIN": cell("mæ̃"),
        "RNK": cell("mæ̃"),
        "RNS_Sisaikhara": cell("mæ̃"),
        "BNM": cell("mæ̃"),
        "RKB": cell("mæ̃"),
        "TkN": cell("mæ̃"),
        "BNT": cell("mæ̃"),
        "RKM": cell("mæ̃"),
        "RNS_Sisana": cell("mæ̃"),
        "KkP": cell("mæ"),
        "DkR": cell("mej", labels="2"),
        "SkP": cell("mæi", labels="2"),
        "DGC": cell("mæi", labels="2"),
        "CCC": cell("muːi", labels="2"),
        "DDK": cell("mʌi", labels="2"),
        "DKS": cell("mʌi", labels="2"),
    }),
    203: ("you (sg. informal)", {
        "HIN": cell("tʊm / tu", labels="1 / 2", column="right"),
        "BNM": cell("tʊm / tu", labels="1 / 2", column="right"),
        "KkP": cell("tʊm", column="right"),
        "RNK": cell("tu", labels="2", column="right"),
        "RNS_Sisaikhara": cell("tu", labels="2", column="right"),
        "BNT": cell("tu", labels="2", column="right"),
        "RNS_Sisana": cell("tu", labels="2", column="right"),
        "DGC": cell("tʌĩ", labels="3", column="right"),
        "SkP": cell("tæi", labels="3", column="right"),
        "CCC": cell("tuːi", labels="3", column="right"),
        "DDK": cell("tæ̃i", labels="3", column="right"),
        "DkR": cell("tæ̃", labels="4", column="right"),
        "TkN": cell("tæ̃", labels="4", column="right"),
        "RKM": cell("tæ̃", labels="4", column="right"),
        "RKB": cell("tæ", labels="4", column="right"),
        "DKS": cell("tæ", labels="4", column="right"),
    }),
    204: ("you (sg. formal)", {
        "HIN": cell("ap", column="right"),
        "RNK": cell("tu", labels="2", column="right", qualifier="response followed by (203)"),
        "RNS_Sisaikhara": cell(
            "tu", labels="2", column="right", qualifier="response followed by (203)"
        ),
        "RNS_Sisana": cell(
            "tu", labels="2", column="right", qualifier="response followed by (203)"
        ),
        "DKS": cell("ʈʊ", labels="2", column="right"),
        "DDK": cell("tũ", labels="2", column="right"),
        "BNM": cell("tum", labels="3", column="right", qualifier="response followed by (203)"),
        "KkP": cell("tum", labels="3", column="right", qualifier="response followed by (203)"),
        "BNT": cell("tum", labels="3", column="right"),
        "RKM": cell("tʊm", labels="3", column="right"),
        "DGC": cell("tʌĩ", labels="4", column="right", qualifier="response followed by (203)"),
        "SkP": cell("tæi", labels="4", column="right", qualifier="response followed by (203)"),
        "DkR": cell("tæ̃", labels="5", column="right", qualifier="response followed by (203)"),
        "TkN": cell("tæ̃", labels="5", column="right", qualifier="response followed by (203)"),
        "RKB": cell("tæ", labels="5", column="right", qualifier="response followed by (203)"),
        "CCC": cell("jʌpʌnahike", labels="6", column="right"),
    }),
    205: ("he", {
        "HIN": cell("vʌh", column="right"),
        "RNK": cell("vʌh", column="right"),
        "BNT": cell("vʌh", column="right"),
        "RNS_Sisaikhara": cell("vo", column="right"),
        "BNM": cell("vo", column="right"),
        "RNS_Sisana": cell("vo", column="right", qualifier="response followed by (174)"),
        "RKB": cell("boh", column="right"),
        "RKM": cell("bʌh", column="right"),
        "TkN": cell("ba", column="right"),
        "DkR": cell("u", labels="2", column="right", qualifier="response followed by (171)"),
        "SkP": cell("u", labels="2", column="right", qualifier="response followed by (171)"),
        "DDK": cell("u", labels="2", column="right", qualifier="response followed by (171)"),
        "DGC": cell("u", labels="2", column="right", qualifier="response followed by (171)"),
        "KkP": cell("u", labels="2", column="right", qualifier="response followed by (171)"),
        "DKS": cell("u", labels="2", column="right", qualifier="response followed by (171)"),
        "CCC": cell(
            "ua", labels="3", page="68", printed="63", column="left",
            qualifier="response followed by (174)"
        ),
    }),
}


def main() -> None:
    rows = []
    for item in range(201, 206):
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
                    "duplicate source code RNS; within-item and within-group occurrence "
                    "order provisionally assigned to metadata row order"
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 95
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 88
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
