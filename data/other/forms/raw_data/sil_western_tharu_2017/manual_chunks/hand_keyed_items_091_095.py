#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 91-95."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_091_095_hand_keyed.tsv")
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


def cell(form, labels="1", page="48", printed="43", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    91: ("milk", {
        "HIN": cell("dudʰ", page="47", printed="42", column="right"),
        "RNK": cell("dudʰ", page="47", printed="42", column="right"),
        "RNS_Sisaikhara": cell("dudʰ", page="47", printed="42", column="right"),
        "BNM": cell("dudʰ", page="47", printed="42", column="right"),
        "DGC": cell("dudʰ", page="47", printed="42", column="right"),
        "DkR": cell("dudʰ", page="47", printed="42", column="right"),
        "SkP": cell("dudʰ", page="47", printed="42", column="right"),
        "RKB": cell("dudʰ", page="47", printed="42", column="right"),
        "TkN": cell("dudʰ", page="47", printed="42", column="right"),
        "DKS": cell("dudʰ", page="47", printed="42", column="right"),
        "BNT": cell("dudʰ", page="47", printed="42", column="right"),
        "RNS_Sisana": cell("dudʰ", page="47", printed="42", column="right"),
        "DDK": cell("dudʰ", page="47", printed="42", column="right"),
        "KkP": cell("dudʰ", page="47", printed="42", column="right"),
        "RKM": cell("dud"),
        "CCC": cell("dudʰa"),
    }),
    92: ("horns", {
        "HIN": cell("sĩŋ"),
        "RNK": cell("sĩŋ"),
        "RNS_Sisaikhara": cell("sĩŋ"),
        "BNM": cell("sĩŋ"),
        "DGC": cell("sĩŋ"),
        "DkR": cell("sĩŋ"),
        "SkP": cell("sĩŋ"),
        "BNT": cell("sĩŋ"),
        "RNS_Sisana": cell("sĩŋ"),
        "DDK": cell("sĩŋ / kãʈa", "1 / 2"),
        "RKB": cell("siŋɡ"),
        "TkN": cell("siŋɡ"),
        "DKS": cell("siŋɡ"),
        "RKM": cell("sĩŋɡ"),
        "CCC": cell("siŋ"),
        "KkP": cell("siŋɡ"),
    }),
    93: ("tail", {
        "HIN": cell("pũtʃʰ / ɖum", "1 / 2"),
        "BNM": cell("pũtʃʰ"),
        "RKB": cell("pũtʃʰ"),
        "BNT": cell("pũtʃʰ"),
        "RKM": cell("pũtʃʰ"),
        "RNK": cell("pũtʃʰija"),
        "RNS_Sisaikhara": cell("pũtʃʰija"),
        "KkP": cell("pũtʃʰija"),
        "DGC": cell("putʃʰĩ"),
        "DKS": cell("putʃʰĩ"),
        "DkR": cell("pũtʃʰi"),
        "SkP": cell("putʃʰ"),
        "TkN": cell("putʃʰ"),
        "DDK": cell("putʃʰ"),
        "RNS_Sisana": cell("putʃʰija"),
        "CCC": cell("putʃʰi"),
    }),
    94: ("goat", {
        "HIN": cell("bʌkʌɾi"),
        "BNM": cell("bʌkʌɾi"),
        "BNT": cell("bʌkʌɾi"),
        "RNK": cell("bʌkʌɾija"),
        "RNS_Sisaikhara": cell("bʌkʌɾija"),
        "SkP": cell("bʌkʌɾija"),
        "RKB": cell("bʌkʌɾija"),
        "TkN": cell("bʌkʌɾija"),
        "RNS_Sisana": cell("bʌkʌɾija"),
        "KkP": cell("bʌkʌɾija"),
        "RKM": cell("bʌkʌɾja"),
        "DGC": cell("tʃʰʌɡɾija", "2"),
        "DkR": cell("tʃʰeɡʌɾa", "2", column="right"),
        "DKS": cell("tʃʰeɡʌɾi", "2", column="right"),
        "CCC": cell("tʃʰeɾi", "2", column="right"),
        "DDK": cell("tʃʰʌɡʌɾi", "2", column="right"),
    }),
    95: ("dog", {
        "HIN": cell("kʊtːa", column="right"),
        "RNK": cell("kʊtːa", column="right"),
        "RNS_Sisaikhara": cell("kʊtːa", column="right"),
        "BNM": cell("kʊtːa", column="right"),
        "SkP": cell("kʊtːa", column="right"),
        "TkN": cell("kʊtːa", column="right"),
        "BNT": cell("kʊtːa", column="right"),
        "RNS_Sisana": cell("kʊtːa", column="right"),
        "RKB": cell("kuʈːa", column="right"),
        "KkP": cell("kuʈːa", column="right"),
        "RKM": cell("kuʈːa", column="right"),
        "DGC": cell("kʊkʌɾa / kʊkʌɾa", "2 / 3", column="right"),
        "DkR": cell("kʊkʌɾa / kʊkʌɾa", "2 / 3", column="right"),
        "DKS": cell("kʊkʊɾ", "2", column="right"),
        "DDK": cell("kʊkʊɾ", "2", column="right"),
        "CCC": cell("kʊkʊɾu / kʊkʊɾu", "2 / 3", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(91, 96):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 80
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 0
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 85
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 79
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
