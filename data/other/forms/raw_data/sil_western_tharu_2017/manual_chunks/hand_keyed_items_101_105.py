#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 101-105."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_101_105_hand_keyed.tsv")
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


def cell(form, labels="1", page="49", printed="44", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    101: ("name", {
        "HIN": cell("nam"),
        "BNM": cell("nam"),
        "BNT": cell("nam"),
        "RNK": cell("nãũ", "2"),
        "RNS_Sisaikhara": cell("nãũ", "2"),
        "DGC": cell("nãũ", "2"),
        "DkR": cell("nãũ", "2"),
        "SkP": cell("nãũ", "2"),
        "RKB": cell("naõ", "2"),
        "DKS": cell("naõ", "2"),
        "TkN": cell("não", "2"),
        "RKM": cell("não", "2"),
        "RNS_Sisana": cell("não", "2"),
        "CCC": cell("nʌːu", "2"),
        "DDK": cell("naũ", "2"),
        "KkP": cell("naũ", "2"),
    }),
    102: ("man", {
        "HIN": cell("adʌmi / puruʃ", "1 / 5"),
        "RNK": cell("adʌmi"),
        "RNS_Sisaikhara": cell("adʌmi"),
        "BNM": cell("adʌmi / amʌdi", "1 / 3"),
        "SkP": cell("adʌmi"),
        "RKB": cell("adʌmi"),
        "TkN": cell("adʌmi"),
        "RKM": cell("adʌmi"),
        "RNS_Sisana": cell("adʌmi"),
        "DGC": cell("mʌnʌi", "2"),
        "DkR": cell("mʌnʌi", "2"),
        "DKS": cell("mʌnæja", "2"),
        "DDK": cell("mʌnʌj", "2"),
        "KkP": cell("mʌnæ / log", "2 / 6"),
        "BNT": cell("amʌdi", "3"),
        "CCC": cell("mardana", "4"),
    }),
    103: ("woman", {
        "HIN": cell(
            "ɔɾʌt / stri", "1 / 6", page="49 / 50", printed="44 / 45",
            column="right / left",
        ),
        "BNM": cell("ɔɾʌt"),
        "BNT": cell(
            "ɔɾʌt / bæjʌɾ", "1 / 2", page="49 / 50", printed="44 / 45",
            column="right / left",
        ),
        "RNK": cell(
            "bʌtʃːʌɾ / bʌjːʌɾ", "2 / 2", page="49 / 50",
            printed="44 / 45", column="right / left",
        ),
        "RNS_Sisaikhara": cell("bʌjːʌɾ", "2"),
        "RKM": cell("bʌjːʌɾ", "2", page="50", printed="45", column="left"),
        "RNS_Sisana": cell("bʌjːʌɾ", "2", page="50", printed="45", column="left"),
        "TkN": cell("bʌjːʌɾ", "2", page="50", printed="45", column="left"),
        "RKB": cell("bæjʌɾ", "2", page="50", printed="45", column="left"),
        "DGC": cell("dʒʌnʌwi", "4", page="50", printed="45", column="left"),
        "DkR": cell("dʒʌnːi", "4", page="50", printed="45", column="left"),
        "DKS": cell("dʒʌnːi", "4", page="50", printed="45", column="left"),
        "DDK": cell("dʒʌnːi", "4", page="50", printed="45", column="left"),
        "CCC": cell("dʒʌni", "4", page="50", printed="45", column="left"),
        "SkP": cell("lʌdija", "5", page="50", printed="45", column="left"),
        "KkP": cell("meharu", "7", page="50", printed="45", column="left"),
    }),
    104: ("child", {
        "HIN": cell("bʌtʃːa", page="50", printed="45", column="left"),
        "RNS_Sisaikhara": cell(
            "balʌk / balʌk", "2 / 3", page="50", printed="45", column="left",
        ),
        "RNS_Sisana": cell("balal", "2", page="50", printed="45", column="left"),
        "BNM": cell("balʌk / balʌk", "2 / 3", page="50", printed="45", column="left"),
        "RNK": cell("balʌk / balʌk", "2 / 3", page="50", printed="45", column="left"),
        "TkN": cell("balʌk / balʌk", "2 / 3", page="50", printed="45", column="left"),
        "BNT": cell(
            "balʌk / balʌk / æwa", "2 / 3 / 6", page="50", printed="45",
            column="left",
        ),
        "RKM": cell("balʌk / balʌk", "2 / 3", page="50", printed="45", column="left"),
        "RKB": cell("walika", "3", page="50", printed="45", column="left"),
        "DGC": cell("lʌika", "4", page="50", printed="45", column="left"),
        "DKS": cell("lʌɖʌka / lʌɖʌka", "4 / 5", page="50", printed="45", column="left"),
        "SkP": cell("lʌɽʌka / lʌɽʌka", "4 / 5", page="50", printed="45", column="left"),
        "KkP": cell("lʌɽʌka / lʌɽʌka", "4 / 5", page="50", printed="45", column="left"),
        "DkR": cell("loɽa", "5", page="50", printed="45", column="left"),
        "DDK": cell("loɽa", "5", page="50", printed="45", column="left"),
        "CCC": cell(None, "", page="50", printed="45", column="left"),
    }),
    105: ("father", {
        "HIN": cell(
            "pita / bap", "1 / 3", page="50", printed="45",
            column="left / right",
        ),
        "RNK": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "DkR": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "TkN": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "DKS": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "RKM": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "DDK": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "KkP": cell("baba / baba / baba", "2 / 3 / 4", page="50", printed="45", column="left / right"),
        "RNS_Sisaikhara": cell("ʌbːa", "2", page="50", printed="45", column="left"),
        "RNS_Sisana": cell("ʌbːa", "2", page="50", printed="45", column="right"),
        "BNM": cell("ʌbːa / bap", "2 / 3", page="50", printed="45", column="left / right"),
        "BNT": cell("ʌbːa", "2", page="50", printed="45", column="left"),
        "DGC": cell("bʌpːa / bʌpːa", "2 / 3", page="50", printed="45", column="right"),
        "SkP": cell("bau", "4", page="50", printed="45", column="right"),
        "RKB": cell("dæwa", "5", page="50", printed="45", column="right"),
        "CCC": cell(None, "", page="50", printed="45", column="left / right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(101, 106):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 111
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 103
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
