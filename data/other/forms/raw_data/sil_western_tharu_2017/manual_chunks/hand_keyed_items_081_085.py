#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 81-85."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_081_085_hand_keyed.tsv")
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


def cell(form, labels, page="46", printed="41", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    81: ("cabbage", {
        "HIN": cell("bʌndɡobʰi", "1"),
        "RNK": cell("bʌndɡobʰi", "1"),
        "RNS_Sisaikhara": cell("bʌndɡobʰi", "1"),
        "BNM": cell("bʌndɡobʰi", "1"),
        "DkR": cell("bʌndɡobʰi", "1"),
        "SkP": cell("bʌndɡobʰi", "1"),
        "TkN": cell("bʌndɡobʰi", "1"),
        "DKS": cell("bʌndɡobʰi", "1"),
        "BNT": cell("bʌndɡobʰi", "1"),
        "RNS_Sisana": cell("bʌndɡobʰi", "1"),
        "DDK": cell("bʌndɡobʰi / ɡaɳʈʰɡobʰi", "1 / 2"),
        "DGC": cell("bʌndɡobʰi / patɡobʰi", "1 / 1"),
        "RKB": cell("bʌndɡobi", "1"),
        "RKM": cell("bʌndɡobi", "1"),
        "KkP": cell("bʌndɡobi", "1"),
        "CCC": cell(None, ""),
    }),
    82: ("oil", {
        "HIN": cell("tel", "1"),
        "RNK": cell("tel", "1"),
        "RNS_Sisaikhara": cell("tel", "1"),
        "BNM": cell("tel", "1"),
        "DkR": cell("tel", "1"),
        "RKB": cell("tel", "1"),
        "TkN": cell("tel", "1"),
        "DKS": cell("tel", "1"),
        "BNT": cell("tel", "1"),
        "RKM": cell("tel", "1"),
        "RNS_Sisana": cell("tel", "1"),
        "CCC": cell("tel", "1"),
        "DDK": cell("tel", "1"),
        "KkP": cell("tel", "1"),
        "DGC": cell("ʈel", "1"),
        "SkP": cell("tjal", "1"),
    }),
    83: ("salt", {
        "HIN": cell("nʌmʌk", "1"),
        "RNK": cell("nun / nun", "2 / 3", column="left / right"),
        "RNS_Sisaikhara": cell("nun / nun", "2 / 3", column="left / right"),
        "RNS_Sisana": cell("nun / nun", "2 / 3", column="left / right"),
        "RKB": cell("nun / nun", "2 / 3", column="left / right"),
        "TkN": cell("nun / nun", "2 / 3", column="left / right"),
        "DKS": cell("nun / nun", "2 / 3", column="left / right"),
        "RKM": cell("nun / nun", "2 / 3", column="left / right"),
        "CCC": cell("nun / nun", "2 / 3", column="left / right"),
        "DDK": cell("nun / nun", "2 / 3", column="left / right"),
        "KkP": cell("nun / nun", "2 / 3", column="left / right"),
        "BNM": cell("non / non", "2 / 3", column="right"),
        "DGC": cell("non / non", "2 / 3", column="right"),
        "DkR": cell("non / non", "2 / 3", column="right"),
        "BNT": cell("non / non", "2 / 3", column="right"),
        "SkP": cell("nwan", "2", column="right"),
    }),
    84: ("meat", {
        "HIN": cell("mãs / ɡoʃt", "1 / 3", column="right"),
        "CCC": cell("masu", "1", column="right"),
        "RNK": cell("sikaɾ", "2", column="right"),
        "RNS_Sisaikhara": cell("sikaɾ", "2", column="right"),
        "BNM": cell("sikaɾ", "2", column="right"),
        "DkR": cell("sikaɾ", "2", column="right"),
        "SkP": cell("sikaɾ", "2", column="right"),
        "RKM": cell("sikaɾ", "2", column="right"),
        "RNS_Sisana": cell("sikaɾ", "2", column="right"),
        "DDK": cell("sikaɾ", "2", column="right"),
        "RKB": cell("sikaɾ / buʈi", "2 / 4", column="right", qualifier="second response: (small piece)"),
        "DGC": cell("ʃikaɾ", "2", column="right"),
        "DKS": cell("ʃikaɾ", "2", column="right"),
        "BNT": cell("ʃikaɾ", "2", column="right"),
        "KkP": cell("sikaɾ / buʈi", "2 / 4", column="right"),
        "TkN": cell(None, "", column="right"),
    }),
    85: ("fat", {
        "HIN": cell("tʃʌɾbi", "1", column="right"),
        "RNK": cell("tadʒõ", "2", column="right"),
        "RNS_Sisaikhara": cell("tadʒõ", "2", column="right"),
        "SkP": cell("tadʒa", "2", column="right"),
        "KkP": cell("tadʒa", "2", column="right"),
        "RKB": cell("tadʒo", "2", column="right"),
        "TkN": cell("tadʒo", "2", column="right"),
        "RKM": cell("tadʒo", "2", column="right"),
        "RNS_Sisana": cell("tadʒo", "2", column="right"),
        "BNM": cell("muʈa", "3", column="right"),
        "BNT": cell("muʈa", "3", column="right"),
        "DkR": cell("moʈ", "3", column="right"),
        "DDK": cell("moʈ", "3", column="right"),
        "DGC": cell("moʈ", "3", page="47", printed="42"),
        "DKS": cell("muʈ", "3", page="47", printed="42"),
        "CCC": cell(None, "", page="46 / 47", printed="41 / 42", column="right / left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(81, 86):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 77
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 3
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 96
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 90
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
