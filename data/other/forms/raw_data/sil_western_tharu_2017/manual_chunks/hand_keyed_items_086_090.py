#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 86-90."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_086_090_hand_keyed.tsv")
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


def cell(form, labels="1", column="left", qualifier=""):
    return form, labels, "47", "42", column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    86: ("fish", {
        "HIN": cell("mʌtʃʰli"),
        "RNK": cell("mʌtʃʰːi"),
        "RNS_Sisaikhara": cell("mʌtʃʰːi"),
        "SkP": cell("mʌtʃʰːi"),
        "RKM": cell("mʌtʃʰːi"),
        "RNS_Sisana": cell("mʌtʃʰːi"),
        "BNM": cell("mʌtʃʰi"),
        "DGC": cell("mʌtʃʰʌɾi"),
        "RKB": cell("mʌtʃʰʌɾi"),
        "DKS": cell("mʌtʃʰʌɾi"),
        "CCC": cell("mʌtʃʰʌɾi"),
        "DDK": cell("mʌtʃʰʌɾi"),
        "KkP": cell("mʌtʃʰʌɾi"),
        "DkR": cell("mʌtʃʌhi"),
        "TkN": cell("mʌtʃʰʌli"),
        "BNT": cell("mʌdʒi"),
    }),
    87: ("chicken", {
        "HIN": cell("mʊɾɡi"),
        "DGC": cell("mʊɾɡi"),
        "DkR": cell("mʊɾɡi"),
        "DKS": cell("mʊɾɡi"),
        "BNT": cell("mʊɾɡi"),
        "DDK": cell("mʊɾɡi"),
        "KkP": cell("mʊɾɡi"),
        "RNK": cell("mʊɾɡija"),
        "RNS_Sisaikhara": cell("mʊɾɡija"),
        "TkN": cell("mʊɾɡija"),
        "BNM": cell("mʊɡi"),
        "SkP": cell("mʊɾɡa"),
        "RKB": cell("muɾɡi"),
        "RKM": cell("mʊɾɡija"),
        "RNS_Sisana": cell("mʊɾɡija"),
        "CCC": cell(None, ""),
    }),
    88: ("egg", {
        "HIN": cell("ʌɳɖa"),
        "RNK": cell("ʌɳɖa"),
        "RNS_Sisaikhara": cell("ʌɳɖa"),
        "BNM": cell("ʌɳɖa"),
        "DGC": cell("ʌɳɖa"),
        "SkP": cell("ʌɳɖa"),
        "BNT": cell("ʌɳɖa"),
        "RNS_Sisana": cell("ʌɳɖa"),
        "KkP": cell("ʌɳɖa"),
        "RKB": cell("aɖa"),
        "CCC": cell("aɖa"),
        "TkN": cell("ʌɳɖa"),
        "RKM": cell("ãɳɖa"),
        "DDK": cell("ãɳɽa"),
        "DkR": cell("ãɾa"),
        "DKS": cell("ãɾa", column="right"),
    }),
    89: ("cow", {
        "HIN": cell("ɡaj", column="right"),
        "BNM": cell("ɡaj", column="right"),
        "BNT": cell("ɡaj", column="right"),
        "RNK": cell("ɡaja", column="right"),
        "RNS_Sisaikhara": cell("ɡaja", column="right"),
        "RKB": cell("ɡaja", column="right"),
        "DkR": cell("ɡʌjːa", column="right"),
        "DDK": cell("ɡʌjːa", column="right"),
        "DGC": cell("ɡʌjːa", column="right"),
        "SkP": cell("ɡʌjã", column="right"),
        "RNS_Sisana": cell("ɡʌjːã", column="right"),
        "TkN": cell("ɡʌ̃jːa", column="right"),
        "DKS": cell("ɡɔja", column="right"),
        "CCC": cell("ɡae", column="right"),
        "KkP": cell("ɡãija", column="right"),
        "RKM": cell("ɡʌjijã", column="right"),
    }),
    90: ("buffalo", {
        "HIN": cell("bʰæs / bʰæ̃s", "1 / 1", column="right"),
        "DkR": cell("bʰæs", column="right"),
        "BNT": cell("bʰẽs", column="right"),
        "RNK": cell("bʰæsija", column="right"),
        "RNS_Sisaikhara": cell("bʰæsija", column="right"),
        "BNM": cell("bʰæsija", column="right"),
        "SkP": cell("bʰæsija", column="right"),
        "DGC": cell("bʰæ̃sʌnija", column="right"),
        "RKB": cell("bʰæsija", column="right"),
        "TkN": cell("bʰæsija", column="right"),
        "RKM": cell("bʰæsija", column="right"),
        "RNS_Sisana": cell("bʰæsija", column="right"),
        "DKS": cell("bʰæs", column="right"),
        "CCC": cell("bʰæsi", column="right"),
        "KkP": cell("bʰaⁱsa", column="right"),
        "DDK": cell("bʌdʰo", "2", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(86, 91):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 79
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 80
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
