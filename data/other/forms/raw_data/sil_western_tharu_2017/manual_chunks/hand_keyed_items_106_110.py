#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 106-110."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_106_110_hand_keyed.tsv")
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


def cell(form, labels="1", page="51", printed="46", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    106: ("mother", {
        "HIN": cell("mata / mã", "1 / 5", page="50", printed="45", column="right"),
        "BNM": cell("abːu / ʌija", "2 / 3", page="50", printed="45", column="right"),
        "RNS_Sisaikhara": cell("ʌjːa", "3", page="50", printed="45", column="right"),
        "RNK": cell("ʌjːa", "3", page="50", printed="45", column="right"),
        "RNS_Sisana": cell("ʌjːa", "3", page="50", printed="45", column="right"),
        "RKB": cell("ɔja", "3", page="50", printed="45", column="right"),
        "TkN": cell("ajːa", "3", page="50", printed="45", column="right"),
        "RKM": cell("aija", "3", page="50", printed="45", column="right"),
        "BNT": cell("ʌia", "3", page="50", printed="45", column="right"),
        "DGC": cell("dai", "4", page="50", printed="45", column="right"),
        "SkP": cell("dai", "4", page="50", printed="45", column="right"),
        "DKS": cell("dai", "4", page="50", printed="45", column="right"),
        "KkP": cell("dai", "4", page="50", printed="45", column="right"),
        "DkR": cell("daji", "4", page="50", printed="45", column="right"),
        "DDK": cell("daji", "4", page="50", printed="45", column="right"),
        "CCC": cell(None, "", page="50", printed="45", column="right"),
    }),
    107: ("older brother", {
        "HIN": cell("bʌɾabʰai / dada", "1 / 2", page="50", printed="45", column="right"),
        "RNK": cell("dʌda", "2", page="50", printed="45", column="right"),
        "RNS_Sisaikhara": cell("dʌda", "2", page="50", printed="45", column="right"),
        "TkN": cell("dʌda", "2", page="50", printed="45", column="right"),
        "RKM": cell("dʌda", "2", page="50", printed="45", column="right"),
        "RNS_Sisana": cell("dʌda", "2", page="50", printed="45", column="right"),
        "BNM": cell("ʌdːa", "2", page="50", printed="45", column="right"),
        "BNT": cell("ʌdːa", "2", page="50", printed="45", column="right"),
        "DGC": cell("dada", "2", page="50", printed="45", column="right"),
        "DkR": cell(
            "dada / dadu", "2 / 2", page="50 / 51", printed="45 / 46",
            column="right / left",
        ),
        "SkP": cell("dada", "2", page="50", printed="45", column="right"),
        "KkP": cell("dada", "2"),
        "DKS": cell("dadu", "2"),
        "DDK": cell("dadu", "2"),
        "RKB": cell("dʌta", "2"),
        "CCC": cell(None, "", page="50 / 51", printed="45 / 46", column="right / left"),
    }),
    108: ("younger brother", {
        "HIN": cell("tʃʰoʈabʰai"),
        "RNK": cell("bʰʌjːa"),
        "RNS_Sisaikhara": cell("bʰʌjːa"),
        "RKM": cell("bʰʌjːa"),
        "RNS_Sisana": cell("bʰʌjːa"),
        "BNM": cell("bʰai"),
        "DGC": cell("bʰæjːa"),
        "DkR": cell("tʃʰuʈʌlibʰʌjːa"),
        "SkP": cell("bʰjːa"),
        "RKB": cell("bʰæja"),
        "DKS": cell("bʰæja"),
        "BNT": cell("bʰæja"),
        "TkN": cell("bʰʌjːa"),
        "CCC": cell("ʌbaⁱja"),
        "DDK": cell("bʰaⁱwa"),
        "KkP": cell("bʰaⁱja"),
    }),
    109: ("older sister", {
        "HIN": cell("didi"),
        "RNK": cell("didi"),
        "RNS_Sisaikhara": cell("didi"),
        "DGC": cell("didi"),
        "DkR": cell("didi"),
        "TkN": cell("didi"),
        "DKS": cell("didi"),
        "BNT": cell("didi / bʌhʌn", "1 / 3"),
        "RKM": cell("didi"),
        "RNS_Sisana": cell("didi"),
        "KkP": cell("didi"),
        "SkP": cell("dɪdi"),
        "RKB": cell("dɪdi"),
        "DDK": cell("ɖaɖi"),
        "BNM": cell("ʌtʃːi", "2"),
        "CCC": cell(None, ""),
    }),
    110: ("younger sister", {
        "HIN": cell("bʌhʌn"),
        "BNM": cell("bʌhʌn"),
        "RNS_Sisaikhara": cell("bʌhʌn / lʌlo", "1 / 2"),
        "DGC": cell("bʌhʌnija / babu", "1 / 3", column="left / right"),
        "DkR": cell("vʌhʌnija"),
        "SkP": cell("bʌhni"),
        "RKM": cell("bʌjinʌja"),
        "KkP": cell("bahini / babu", "1 / 3", column="left / right"),
        "RNK": cell("lʌlo", "2"),
        "TkN": cell("lʌlo", "2"),
        "RNS_Sisana": cell("lʌlo", "2", column="right"),
        "RKB": cell("lʌlːo", "2", column="right"),
        "BNT": cell("ʌlːo", "2", column="right"),
        "DKS": cell("babu", "3", column="right"),
        "DDK": cell("babu", "3", column="right"),
        "CCC": cell(None, "", column="left / right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(106, 111):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 76
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 4
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 84
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 77
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
