#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 71-75."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_071_075_hand_keyed.tsv")
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


def cell(form, labels, column="left", qualifier=""):
    return form, labels, "44", "39", column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    71: ("rice", {
        "HIN": cell("tʃawʌl", "1"),
        "BNM": cell("tʃawʌl", "1"),
        "TkN": cell("tʃʌwʌl", "1"),
        "RNK": cell("tʃamʌɾ / tʃawʌɾ", "1 / 1"),
        "RKB": cell("tʃamʌɾ", "1"),
        "RKM": cell("tʃamʌɾ", "1"),
        "RNS_Sisaikhara": cell("tʃamʌɾ", "1"),
        "KkP": cell("tʃamʌɾ", "1"),
        "RNS_Sisana": cell("tʃawʌɾ", "1"),
        "BNT": cell("tʃawʌɾ", "1"),
        "DDK": cell("tʃawʌɾ", "1"),
        "DGC": cell("tʃauɾ", "1"),
        "DkR": cell("tʃauɾ", "1"),
        "SkP": cell("tʃʌɔɾ", "1"),
        "DKS": cell("tʃawuɾ", "1"),
        "CCC": cell(None, ""),
    }),
    72: ("potato", {
        "HIN": cell("alu", "1"),
        "RNK": cell("alu", "1"),
        "RNS_Sisaikhara": cell("alu", "1"),
        "BNM": cell("alu", "1"),
        "DGC": cell("alu", "1"),
        "DkR": cell("alu", "1"),
        "SkP": cell("alu", "1"),
        "RKB": cell("alu", "1"),
        "TkN": cell("alu", "1"),
        "DKS": cell("alu", "1"),
        "BNT": cell("alu", "1"),
        "RKM": cell("alu", "1"),
        "RNS_Sisana": cell("alu", "1"),
        "DDK": cell("alu", "1"),
        "KkP": cell("alu", "1"),
        "CCC": cell("alo", "1"),
    }),
    73: ("eggplant", {
        "HIN": cell("bæ̃ŋɡʌn", "1"),
        "RNK": cell("bʰʌʈa", "2", column="right"),
        "RNS_Sisaikhara": cell("bʰʌʈa", "2", column="right"),
        "BNM": cell("bʰʌʈa", "2", column="right"),
        "TkN": cell("bʰʌʈa", "2", column="right"),
        "RKM": cell("bʰʌʈa", "2", column="right"),
        "RNS_Sisana": cell("bʰʌʈa", "2", column="right"),
        "DGC": cell("bʰaʈa", "2", column="right"),
        "DkR": cell("bʰaʈa", "2", column="right"),
        "DKS": cell("bʰaʈa", "2", column="right"),
        "KkP": cell("bʰaʈa", "2", column="right"),
        "SkP": cell("bʰaʈʌa", "2", column="right"),
        "RKB": cell("bʌʈa", "2", column="right"),
        "BNT": cell("bʌʈa", "2", column="right"),
        "DDK": cell("bʰaɳʈa", "2", column="right"),
        "CCC": cell(None, "", column="left / right"),
    }),
    74: ("groundnut", {
        "HIN": cell("mũŋɡpʰʌli / mompʰʌli", "1 / 1", column="right"),
        "BNM": cell("muŋpʰʌli", "1", column="right"),
        "RNK": cell("mumpʌɾi", "1", column="right"),
        "RNS_Sisaikhara": cell("mumpʌɾi", "1", column="right"),
        "DGC": cell("mumpʰʌli", "1", column="right"),
        "DkR": cell("mumpʰʌli", "1", column="right"),
        "SkP": cell("mumpʰʌli", "1", column="right"),
        "TkN": cell("mumpʰʌli", "1", column="right"),
        "KkP": cell("mumpʰʌli", "1", column="right"),
        "RKB": cell("mʌŋɡpʰʌli", "1", column="right"),
        "DKS": cell("bʌmbʰʌli", "1", column="right"),
        "BNT": cell("mũpʰʌɾi", "1", column="right"),
        "RNS_Sisana": cell("mũpʰʌɾi", "1", column="right"),
        "RKM": cell("mumpʰʌɾi", "1", column="right"),
        "DDK": cell("mompʰʌli", "1", column="right"),
        "CCC": cell("bedam", "2", column="right"),
    }),
    75: ("chili", {
        "HIN": cell("miɾtʃ", "1", column="right"),
        "RNK": cell("miɾtʃ", "1", column="right"),
        "RNS_Sisaikhara": cell("miɾtʃ", "1", column="right"),
        "BNM": cell("miɾtʃ", "1", column="right"),
        "SkP": cell("miɾtʃ", "1", column="right"),
        "TkN": cell("miɾtʃ", "1", column="right"),
        "DKS": cell("miɾtʃ", "1", column="right"),
        "BNT": cell("miɾtʃ", "1", column="right"),
        "RKM": cell("miɾtʃ", "1", column="right"),
        "RNS_Sisana": cell("miɾtʃ", "1", column="right"),
        "DDK": cell("miɾtʃ", "1", column="right"),
        "DGC": cell("miɾtʃi / miɾtʃa", "1 / 1", column="right"),
        "DkR": cell("miɾtʃi", "1", column="right"),
        "KkP": cell("miɾtʃa", "1", column="right"),
        "RKB": cell("mitʃi", "1", column="right"),
        "CCC": cell("maɾtʃa", "1", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(71, 76):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 81
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
