#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 116-120."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_116_120_hand_keyed.tsv")
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


def cell(form, labels="1", page="52", printed="47", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600-dpi rendered-page crops before
# any comparison with the legacy CSV.
ITEMS = {
    116: ("girl", {
        "HIN": cell("lʌɽʌki", page="52", printed="47", column="left"),
        "BNM": cell(
            "lʌɽʌki / ʌlːo", "1 / 4", page="52", printed="47",
            column="left / right", qualifier="second response: (110)",
        ),
        "RNK": cell("lɔɽija", "2", page="52", printed="47", column="left", qualifier="(112)"),
        "RNS_Sisaikhara": cell("lɔɽija", "2", qualifier="(112)"),
        "DkR": cell("lɔɽija", "2", qualifier="(112)"),
        "RKB": cell("lɔɽija", "2", qualifier="(112)"),
        "TkN": cell("lɔɽija", "2", qualifier="(112)"),
        "DDK": cell("lɔɽija / tʃʰaji", "2 / 3", qualifier="second response: (112)"),
        "DGC": cell("lɔɽi", "2"),
        "RKM": cell("lɔɽija", "2", qualifier="(112)"),
        "RNS_Sisana": cell("lɔɽija", "2", qualifier="(112)"),
        "KkP": cell("loɳɖia", "2"),
        "DKS": cell("loɳɖi / tʃʰaji", "2 / 3", qualifier="second response: (112)"),
        "SkP": cell("tʃais", "3", qualifier="(112)"),
        "BNT": cell("ʌlːo", "4", qualifier="(110)"),
        "CCC": cell(None, "", page="52", printed="47", column="left / right"),
    }),
    117: ("day", {
        "HIN": cell("din"),
        "RNK": cell("din"),
        "RNS_Sisaikhara": cell("din"),
        "BNM": cell("din"),
        "DkR": cell("din"),
        "SkP": cell("din"),
        "RKB": cell("din"),
        "TkN": cell("din"),
        "DKS": cell("din"),
        "BNT": cell("din"),
        "RKM": cell("din"),
        "RNS_Sisana": cell("din"),
        "DDK": cell("din"),
        "KkP": cell("din"),
        "DGC": cell("ɖin"),
        "CCC": cell("din"),
    }),
    118: ("night", {
        "HIN": cell("ɾat"),
        "RNK": cell("ɾat"),
        "RNS_Sisaikhara": cell("ɾat"),
        "BNM": cell("ɾat"),
        "DkR": cell("ɾat"),
        "SkP": cell("ɾat"),
        "RKB": cell("ɾat"),
        "TkN": cell("ɾat"),
        "DKS": cell("ɾat"),
        "BNT": cell("ɾat"),
        "RKM": cell("ɾat"),
        "RNS_Sisana": cell("ɾat"),
        "DDK": cell("ɾat"),
        "KkP": cell("ɾat"),
        "DGC": cell("ɾaʈ"),
        "CCC": cell("ɾati"),
    }),
    119: ("morning", {
        "HIN": cell("sʊbʌh / sʌweɾa", "1 / 5", page="52 / 53", printed="47 / 48", column="right / left"),
        "RNK": cell("bʰoɾ", "2", page="53", printed="48", column="left"),
        "RNS_Sisaikhara": cell("bʰoɾ", "2", page="53", printed="48", column="left"),
        "RKB": cell("bʰoɾ", "2", page="53", printed="48", column="left"),
        "RKM": cell("bʰoɾ", "2", page="53", printed="48", column="left"),
        "RNS_Sisana": cell("bʰoɾ", "2", page="53", printed="48", column="left"),
        "BNM": cell("tʌɖʌke", "3", page="53", printed="48", column="left"),
        "BNT": cell("tʌɾʌke", "3", page="53", printed="48", column="left"),
        "DGC": cell("sʌkaɾe / sʌkaɾe", "4 / 5", page="53", printed="48", column="left"),
        "DKS": cell("sʌkaɾe / sʌkaɾe", "4 / 5", page="53", printed="48", column="left"),
        "DkR": cell("sʌkaɾ / vihan", "4 / 6", page="53", printed="48", column="left"),
        "DDK": cell("sʌkoɾ", "4", page="53", printed="48", column="left"),
        "TkN": cell("sʌweɾo", "5", page="53", printed="48", column="left"),
        "SkP": cell("bihan", "6", page="53", printed="48", column="left"),
        "CCC": cell("bihan", "6", page="53", printed="48", column="left"),
        "KkP": cell("bʰen", "6", page="53", printed="48", column="left"),
    }),
    120: ("noon", {
        "HIN": cell("dopʌhʌɾ", page="53", printed="48", column="left"),
        "DGC": cell("dopʌhʌɾ", page="53", printed="48", column="left"),
        "RNK": cell("dopʌhʌɾi", page="53", printed="48", column="left"),
        "RNS_Sisaikhara": cell("dopʌhʌɾi", page="53", printed="48", column="left"),
        "BNM": cell("dupɔɾija", page="53", printed="48", column="left"),
        "SkP": cell("dupʌhʌɾ", page="53", printed="48", column="left"),
        "RKB": cell("dupʌhʌɾ", page="53", printed="48", column="left"),
        "DKS": cell("dupʌhʌɾ", page="53", printed="48", column="left"),
        "DDK": cell("dupʌhʌɾ", page="53", printed="48", column="left"),
        "KkP": cell("dupʌhʌɾ", page="53", printed="48", column="left"),
        "TkN": cell("dʊpʌhʌɾi", page="53", printed="48", column="left"),
        "BNT": cell("dʊpʌhʌɾija", page="53", printed="48", column="left"),
        "RKM": cell("dʊpʌhʌɾ", page="53", printed="48", column="left"),
        "RNS_Sisana": cell("dʊpahʌɾi", page="53", printed="48", column="left"),
        "DkR": cell("mintʃʰidʒun", "2", page="53", printed="48", column="left"),
        "CCC": cell(None, "", page="53", printed="48", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(116, 121):
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
