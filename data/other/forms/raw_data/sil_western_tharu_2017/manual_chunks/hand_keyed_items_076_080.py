#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 76-80."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_076_080_hand_keyed.tsv")
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


def cell(form, labels, page="45", printed="40", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    76: ("turmeric", {
        "HIN": cell("hʌldi", "1"),
        "RNK": cell("hʌɾʌdi", "1"),
        "RNS_Sisaikhara": cell("hʌɾʌdi", "1"),
        "DkR": cell("hʌɾʌdi", "1"),
        "DKS": cell("hʌɾʌdi", "1"),
        "TkN": cell("hʌɾʌdi", "1"),
        "RKM": cell("hʌɾʌdi", "1"),
        "RNS_Sisana": cell("hʌɾʌdi", "1"),
        "DDK": cell("hʌɾʌdi", "1"),
        "BNM": cell("hʌɾʌd", "1"),
        "DGC": cell("hʌɾʌd", "1"),
        "BNT": cell("hʌɾʌd", "1"),
        "SkP": cell("hʌɾdi", "1"),
        "KkP": cell("hʌɾdi", "1"),
        "RKB": cell("hʌɾʌdi", "1"),
        "CCC": cell(None, ""),
    }),
    77: ("garlic", {
        "HIN": cell("lʌhsʊn", "1"),
        "BNM": cell("lʌhsʊn", "1"),
        "DkR": cell("lʌhsʊn", "1"),
        "BNT": cell("lʌhsʊn", "1"),
        "RNK": cell("lasun", "1"),
        "RNS_Sisaikhara": cell("lasun", "1"),
        "SkP": cell("lasun", "1"),
        "TkN": cell("lasun", "1"),
        "RKM": cell("lasun", "1"),
        "KkP": cell("lasun", "1"),
        "DGC": cell("lʌɾʌsʊn", "1"),
        "RKB": cell("lʌhsun", "1"),
        "DKS": cell("nʌhʌsun", "1"),
        "RNS_Sisana": cell("lasʌn", "1"),
        "CCC": cell("lʌhʌsun", "1"),
        "DDK": cell("lʌsun", "1"),
    }),
    78: ("onion", {
        "HIN": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "RNK": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "RNS_Sisaikhara": cell(
            "pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"
        ),
        "RNS_Sisana": cell(
            "pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"
        ),
        "BNM": cell(
            "pjadʒ / pjadʒ / pjadʒ / ɡʌɳʈʰi",
            "1 / 2 / 3 / 4", column="left / right",
        ),
        "DGC": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "DkR": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "RKB": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "TkN": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "DKS": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "RKM": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "DDK": cell("pjadʒ / pjadʒ / pjadʒ", "1 / 2 / 3", column="left / right"),
        "SkP": cell("pja", "1"),
        "CCC": cell("piadʒu", "2"),
        "KkP": cell("pedʒ", "3", column="right"),
        "BNT": cell("ɡʌɳʈʰi", "4", column="right"),
    }),
    79: ("cauliflower", {
        "HIN": cell("pʰulɡobʰi", "1", column="right"),
        "RNK": cell("pʰulɡobʰi", "1", column="right"),
        "RNS_Sisaikhara": cell("pʰulɡobʰi", "1", column="right"),
        "BNM": cell("pʰulɡobʰi", "1", column="right"),
        "DGC": cell("pʰulɡobʰi", "1", column="right"),
        "DkR": cell("pʰulɡobʰi", "1", column="right"),
        "SkP": cell("pʰulɡobʰi", "1", column="right"),
        "TkN": cell("pʰulɡobʰi", "1", column="right"),
        "DKS": cell("pʰulɡobʰi", "1", column="right"),
        "BNT": cell("pʰulɡobʰi", "1", column="right"),
        "RNS_Sisana": cell("pʰulɡobʰi", "1", column="right"),
        "DDK": cell("pʰulɡobʰi", "1", column="right"),
        "RKB": cell("pʰulɡobi", "1", column="right"),
        "RKM": cell("pʰulɡobi", "1", column="right"),
        "KkP": cell("pʰulɡobi", "1", column="right"),
        "CCC": cell(None, "", column="right"),
    }),
    80: ("tomato", {
        "HIN": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "RNK": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "RNS_Sisaikhara": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "BNM": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "DGC": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "DkR": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "SkP": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "TkN": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "RKM": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "RNS_Sisana": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "DDK": cell("ʈʌmaʈʌɾ", "1", column="right"),
        "RKB": cell("ʈimaʈʌɾ", "1", column="right"),
        "DKS": cell("ʈimaʈʌɾ", "1", page="46", printed="41"),
        "KkP": cell("ʈimaʈʌɾ", "1", page="46", printed="41"),
        "BNT": cell("ʈʌmʈʌmbʰʌʈa", "2", page="46", printed="41"),
        "CCC": cell("ɾambʰʌnʈa", "3", page="46", printed="41"),
    }),
}


def main() -> None:
    rows = []
    for item in range(76, 81):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 103
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 96
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
