#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 61-65."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_061_065_hand_keyed.tsv")
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


def cell(form, labels, page=42, printed=37, column="right", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    61: ("tree", {
        "HIN": cell("peɖ", "1", column="left"),
        "RNK": cell("peɖ / ɾukʰa", "1 / 2", column="left / right"),
        "RNS_Sisaikhara": cell("peɖ", "1", column="left"),
        "BNM": cell("peɖ", "1", column="left"),
        "RKB": cell("peɖ", "1", column="left"),
        "BNT": cell("peɖ", "1", column="left"),
        "RKM": cell("peɾ", "1", column="left"),
        "RNS_Sisana": cell("peɾ", "1"),
        "DGC": cell("peɾwa", "1"),
        "DKS": cell("peɾʌ / ɾukʰʌwa", "1 / 2"),
        "TkN": cell("ɾukʰa", "2"),
        "KkP": cell("ɾukʰa", "2"),
        "SkP": cell("ɾukʰːa", "2"),
        "DkR": cell("ɾukʰʌwa", "2"),
        "DDK": cell("ɾukʰʌwa", "2"),
        "CCC": cell("ɡatʃʰ", "3"),
    }),
    62: ("leaf", {
        "HIN": cell("pʌʈːa", "1"),
        "RNK": cell("pʌʈːa", "1"),
        "RNS_Sisaikhara": cell("pʌʈːa", "1"),
        "BNM": cell("pʌʈːa", "1"),
        "DkR": cell("pʌʈːa", "1"),
        "RKB": cell("pʌʈːa", "1"),
        "BNT": cell("pʌʈːa", "1"),
        "KkP": cell("pʌʈːa", "1"),
        "DGC": cell("pʌʈija / pata", "1 / 1"),
        "DDK": cell("pʌʈija", "1"),
        "SkP": cell("pata", "1"),
        "CCC": cell("pata", "1"),
        "TkN": cell("pʌta", "1"),
        "DKS": cell("pʌʈːija", "1"),
        "RKM": cell("pʌtːa", "1"),
        "RNS_Sisana": cell("pʌtːa", "1"),
    }),
    63: ("root", {
        "HIN": cell("tʌna / dʒʌɾ", "1 / 3"),
        "RNK": cell("hãŋɡa", "2"),
        "TkN": cell("hãŋɡa", "2"),
        "RKM": cell("hãŋɡa", "2"),
        "RNS_Sisaikhara": cell("hãŋɡa", "2"),
        "DkR": cell("hʌɡa / ɖahã", "2 / 4"),
        "SkP": cell("hãŋɡa", "2"),
        "RNS_Sisana": cell("hʌŋɡpa", "2"),
        "BNM": cell("dʒʌɖ", "3"),
        "DGC": cell("dʒʌɾ", "3"),
        "RKB": cell("dʒʌɾ", "3"),
        "DKS": cell("dʒʌɾ", "3"),
        "DDK": cell("dʒʌɾ / ɖahã", "3 / 4"),
        "BNT": cell("dʒʰʌɖ", "3"),
        "CCC": cell("dʒʌɾi", "3"),
        "KkP": cell("dʒʰʌɾa", "3"),
    }),
    64: ("thorn", {
        "HIN": cell("kãʈa", "1"),
        "DGC": cell("kãʈa", "1", page=43, printed=38, column="left"),
        "DkR": cell("kãʈa", "1", page=43, printed=38, column="left"),
        "RNK": cell("kãʈo", "1", page=43, printed=38, column="left"),
        "RNS_Sisaikhara": cell("kãʈo", "1", page=43, printed=38, column="left"),
        "BNM": cell("kãʈo", "1", page=43, printed=38, column="left"),
        "SkP": cell("kãʈ", "1", page=43, printed=38, column="left"),
        "RKB": cell("kaʈo", "1", page=43, printed=38, column="left"),
        "TkN": cell("kãʈo", "1", page=43, printed=38, column="left"),
        "RKM": cell("kãʈo", "1", page=43, printed=38, column="left"),
        "RNS_Sisana": cell("kãʈo", "1", page=43, printed=38, column="left"),
        "DKS": cell("kaʈ", "1", page=43, printed=38, column="left"),
        "KkP": cell("kaʈ / ɡaŋʈʰi", "1 / 1", page=43, printed=38, column="left"),
        "CCC": cell("kaːʈ", "1", page=43, printed=38, column="left"),
        "DDK": cell("kaʈa", "1", page=43, printed=38, column="left"),
        "BNT": cell("dʒʰaɾ", "2", page=43, printed=38, column="left"),
    }),
    65: ("flower", {
        "HIN": cell("pʰul", "1", page=43, printed=38, column="left"),
        "BNM": cell("pʰul", "1", page=43, printed=38, column="left"),
        "TkN": cell("pʰul", "1", page=43, printed=38, column="left"),
        "BNT": cell("pʰul", "1", page=43, printed=38, column="left"),
        "RNK": cell("pʰula", "1", page=43, printed=38, column="left"),
        "RNS_Sisaikhara": cell("pʰula", "1", page=43, printed=38, column="left"),
        "DGC": cell("pʰula", "1", page=43, printed=38, column="left"),
        "DkR": cell("pʰula", "1", page=43, printed=38, column="left"),
        "SkP": cell("pʰula", "1", page=43, printed=38, column="left"),
        "RKM": cell("pʰula", "1", page=43, printed=38, column="left"),
        "RNS_Sisana": cell("pʰula", "1", page=43, printed=38, column="left"),
        "CCC": cell("pʰula", "1", page=43, printed=38, column="left"),
        "DDK": cell("pʰula", "1", page=43, printed=38, column="left"),
        "KkP": cell("pʰula", "1", page=43, printed=38, column="left"),
        "RKB": cell("pʰula", "1", page=43, printed=38, column="left"),
        "DKS": cell("pʰula", "1", page=43, printed=38, column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(61, 66):
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
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 87
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 81
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
