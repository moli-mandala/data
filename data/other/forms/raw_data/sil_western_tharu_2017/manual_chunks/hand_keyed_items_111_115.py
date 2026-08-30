#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 111-115."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_111_115_hand_keyed.tsv")
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


def cell(form, labels="1", page="51", printed="46", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    111: ("son", {
        "HIN": cell("beʈa / putra", "1 / 4"),
        "BNM": cell("beʈa"),
        "BNT": cell("beʈa"),
        "RNK": cell("lɔɽa", "2"),
        "RNS_Sisaikhara": cell("lɔɽa", "2"),
        "TkN": cell("lɔɽa", "2"),
        "RNS_Sisana": cell("lɔɽa", "2"),
        "DGC": cell("lɔɽa / pʊtʌva", "2 / 4"),
        "RKB": cell("lɔɽa", "2"),
        "RKM": cell("lɔɽã", "2"),
        "KkP": cell("loɳɖa", "2"),
        "DkR": cell("tʃʰawa", "5"),
        "DKS": cell("tʃʰawa", "5"),
        "DDK": cell("tʃʰawa", "5"),
        "SkP": cell("tʃʰawʌs", "5"),
        "CCC": cell(None, ""),
    }),
    112: ("daughter", {
        "HIN": cell("beʈi / putri", "1 / 4"),
        "BNM": cell("beʈi"),
        "BNT": cell("beʈi"),
        "DGC": cell("beʈi / lɔɽi", "1 / 2"),
        "RNK": cell("lɔɽija", "2"),
        "RNS_Sisaikhara": cell("lɔɽija", "2"),
        "TkN": cell("lɔɽija", "2"),
        "RKB": cell("lɔɽija", "2"),
        "RKM": cell("lɔɽija", "2"),
        "RNS_Sisana": cell("lɔɽija", "2"),
        "KkP": cell("loɳɖia", "2"),
        "DkR": cell("tʃʰaji", "3"),
        "DKS": cell("tʃʰaji", "3"),
        "DDK": cell("tʃʰaji", "3"),
        "SkP": cell("tʃʰais", "3"),
        "CCC": cell(None, ""),
    }),
    113: ("husband", {
        "HIN": cell("pʌti"),
        "RNK": cell("loga", "2"),
        "RNS_Sisaikhara": cell("loga", "2"),
        "RKB": cell("loga", "2"),
        "TkN": cell("loga", "2"),
        "RKM": cell("loga", "2"),
        "RNS_Sisana": cell("loga", "2"),
        "BNT": cell("log", "2", page="52", printed="47", column="left"),
        "KkP": cell(
            "log / dulʌha / misaɾwa", "2 / 3 / 5", page="52", printed="47",
            column="left", qualifier="first response: (102)",
        ),
        "BNM": cell("gʰʌɾwala", "3", page="52", printed="47", column="left"),
        "DGC": cell("tʰʌɾua", "3", page="52", printed="47", column="left"),
        "DkR": cell("tʰaɾu", "3", page="52", printed="47", column="left"),
        "DKS": cell("dʰaɾu", "3", page="52", printed="47", column="left"),
        "DDK": cell("tʰʌɾuwa", "3", page="52", printed="47", column="left"),
        "SkP": cell("bʰʌtaɾʌs", "4", page="52", printed="47", column="left"),
        "CCC": cell(None, "", page="51 / 52", printed="46 / 47", column="right / left"),
    }),
    114: ("wife", {
        "HIN": cell("pʌtni", page="52", printed="47", column="left"),
        "RNK": cell(
            "bʌzdʒʌɾ / bʌjːʌɾ", "2 / 3", page="52", printed="47",
            column="left", qualifier="second response: (103)",
        ),
        "RNS_Sisaikhara": cell("bʌjːʌɾ", "3", page="52", printed="47", column="left", qualifier="(103)"),
        "RNS_Sisana": cell("bʌjːʌɾ", "3", page="52", printed="47", column="left", qualifier="(103)"),
        "TkN": cell("bʌjːʌɾ", "3", page="52", printed="47", column="left", qualifier="(103)"),
        "RKM": cell("bʌjːʌɾ", "3", page="52", printed="47", column="left", qualifier="(103)"),
        "BNM": cell("bʌjaɾ / gʰʌɾʌwali", "3 / 4", page="52", printed="47", column="left"),
        "RKB": cell("bæjʌɾ", "3", page="52", printed="47", column="left", qualifier="(103)"),
        "BNT": cell("bæjʌɾ", "3", page="52", printed="47", column="left", qualifier="(103)"),
        "DGC": cell("dʒʌnʌni", "5", page="52", printed="47", column="left", qualifier="(103)"),
        "DkR": cell("dʒʌnni", "5", page="52", printed="47", column="left", qualifier="(103)"),
        "DKS": cell("dʒʌnni", "5", page="52", printed="47", column="left", qualifier="(103)"),
        "DDK": cell("dʒʌnewa", "5", page="52", printed="47", column="left"),
        "SkP": cell("dʒwis", "6", page="52", printed="47", column="left"),
        "KkP": cell("meharua", "7", page="52", printed="47", column="left", qualifier="(103)"),
        "CCC": cell(None, "", page="52", printed="47", column="left"),
    }),
    115: ("boy", {
        "HIN": cell("lʌɽʌka", page="52", printed="47", column="left"),
        "BNM": cell("lʌɽʌka", page="52", printed="47", column="left"),
        "BNT": cell("balʌk / loɳɖe", "1 / 5", page="52", printed="47", column="left"),
        "DGC": cell("lɔɽa", "2", page="52", printed="47", column="left"),
        "DkR": cell("lɔɽa", "2", page="52", printed="47", column="left", qualifier="(104)"),
        "DDK": cell(
            "lɔɽa / tʃʰawa", "2 / 3", page="52", printed="47", column="left",
            qualifier="first response: (104); second response: (111)",
        ),
        "RKB": cell("lɔɽa", "2", page="52", printed="47", column="left", qualifier="(111)"),
        "TkN": cell("lɔɽa", "2", page="52", printed="47", column="left"),
        "RKM": cell("lɔɽa", "2", page="52", printed="47", column="left"),
        "RNS_Sisaikhara": cell("lɔɽa", "2", page="52", printed="47", column="left"),
        "RNK": cell("lɔɽa", "2", page="52", printed="47", column="left"),
        "RNS_Sisana": cell("lɔɽa", "2", page="52", printed="47", column="left"),
        "KkP": cell("loɳɖa", "2", page="52", printed="47", column="left", qualifier="(111)"),
        "DKS": cell(
            "loɳɖa / tʃʰawa", "2 / 3", page="52", printed="47", column="left",
            qualifier="second response: (111)",
        ),
        "SkP": cell("tʃʰawʌs", "3", page="52", printed="47", column="left", qualifier="(111)"),
        "CCC": cell(None, "", page="52", printed="47", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(111, 116):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 75
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 5
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 86
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
