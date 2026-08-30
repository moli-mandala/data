#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 126-130."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_126_130_hand_keyed.tsv")
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


def cell(form, labels="1", page="54", printed="49", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600-dpi rendered-page crops before
# any comparison with the legacy CSV.
ITEMS = {
    126: ("month", {
        "HIN": cell("mʌhina"),
        "DGC": cell("mʌhina"),
        "DKS": cell("mʌhina"),
        "KkP": cell("mʌhina"),
        "RNK": cell("mʌhʌna"),
        "RNS_Sisaikhara": cell("mʌhʌna"),
        "RKM": cell("mʌhʌna"),
        "DDK": cell("mʌhʌna"),
        "BNM": cell("mʌhin"),
        "DkR": cell("mʌhina"),
        "SkP": cell("mʌhina"),
        "TkN": cell("mʌhɪna"),
        "RNS_Sisana": cell("mʌhina"),
        "RKB": cell("mahina"),
        "CCC": cell("mahina"),
        "BNT": cell("mahine"),
    }),
    127: ("year", {
        "HIN": cell("sal / vʌɾʃ", "1 / 2", column="left / right"),
        "RNK": cell("sal"),
        "RNS_Sisaikhara": cell("sal"),
        "BNM": cell("sal"),
        "DGC": cell("sal"),
        "SkP": cell("sal"),
        "RKB": cell("sal"),
        "BNT": cell("sal"),
        "RKM": cell("sal"),
        "RNS_Sisana": cell("sal"),
        "DDK": cell("sal"),
        "KkP": cell("sal / bʌɾʌs", "1 / 2", column="left / right"),
        "DKS": cell("sal / bʌɾʌs", "1 / 2", column="left / right"),
        "DkR": cell("bʌɾesdin", "2"),
        "TkN": cell("bʌɾʌs", "2"),
        "CCC": cell("bʌɾʌs", "2", column="right"),
    }),
    128: ("old", {
        "HIN": cell("pʊɾana", column="right"),
        "DGC": cell("pʊɾana", column="right"),
        "BNT": cell("pʊɾana", column="right"),
        "BNM": cell("pʊɾana / bʌhutsalka", "1 / 2", column="right"),
        "KkP": cell("pʊɾana", column="right"),
        "RNK": cell("pʊɾanɔ", column="right"),
        "RNS_Sisaikhara": cell("pʊɾanɔ", column="right"),
        "DkR": cell("pʊɾan", column="right"),
        "SkP": cell("pʊɾan", column="right"),
        "DDK": cell("pʊɾan", column="right"),
        "RKB": cell("purana", column="right"),
        "TkN": cell("purano", column="right"),
        "DKS": cell("puɾaɳa", column="right"),
        "RKM": cell("pʊɾano", column="right"),
        "RNS_Sisana": cell("pʊɾano", column="right"),
        "CCC": cell("puɾan", column="right"),
    }),
    129: ("new", {
        "DkR": cell("lʌbːa", column="right"),
        "SkP": cell("lawa", column="right"),
        "DDK": cell("lawa", column="right"),
        "DKS": cell("lɔa", column="right"),
        "RNK": cell("nãvi", "2", column="right"),
        "RNS_Sisaikhara": cell("nãvi", "2", column="right"),
        "RKM": cell("nʌ̃u", "2", column="right"),
        "RNS_Sisana": cell("nʌ̃u", "2", column="right"),
        "TkN": cell("nʌjã", "2", column="right"),
        "DGC": cell("nʌjã", "2", column="right"),
        "HIN": cell("nʌja", "2", column="right"),
        "BNM": cell("nʌja", "2", column="right"),
        "RKB": cell("nʌja", "2", column="right"),
        "BNT": cell("nʌja", "2", column="right"),
        "KkP": cell("nʌmːa", "2", column="right"),
        "CCC": cell("lʌota", "3", column="right"),
    }),
    130: ("good", {
        "HIN": cell("ʌtʃʰːa / bʌɾija", "1 / 3", page="54 / 55", printed="49 / 50", column="right / left"),
        "BNM": cell("ʌtʃʰːa", column="right"),
        "BNT": cell("ʌtʃʰːa", column="right"),
        "RNK": cell("ʌtʃʰːo", column="right"),
        "RNS_Sisaikhara": cell("ʌtʃʰːo", column="right"),
        "TkN": cell("ʌtʃʰːo", column="right"),
        "RKB": cell("atʃʰːa", column="right"),
        "RKM": cell("ʌtʃʰo", column="right"),
        "RNS_Sisana": cell("ʌtʃʰo", column="right"),
        "DKS": cell("mʌdʒa / bʌɾija", "2 / 3", page="54 / 55", printed="49 / 50", column="right / left"),
        "DDK": cell("mʌdʒa", "2", column="right"),
        "DGC": cell("bʌɾija", "3", page="55", printed="50", column="left"),
        "DkR": cell("sʊgʰːʌɾ", "4", page="55", printed="50", column="left"),
        "KkP": cell("sugʰʌɾ", "4", page="55", printed="50", column="left"),
        "SkP": cell("næŋg", "5", page="55", printed="50", column="left"),
        "CCC": cell("ɖol", "6", page="55", printed="50", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(126, 131):
        gloss, cells = ITEMS[item]
        assert set(cells) == set(SITES)
        for site in SITES:
            form, labels, page, printed, column, qualifier = cells[site]
            uncertainty = ""
            site_confidence = "high"
            if site.startswith("RNS_"):
                site_confidence = "medium"
                uncertainty = (
                    "duplicate source code RNS; within-group occurrence order assigned to "
                    "metadata row order; unmatched extra group occurrence assigned to "
                    "metadata row 1 (Sisaikhara)"
                )
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": SOURCE_CODES[site],
                "Source_Code_Occurrence": OCCURRENCE[site],
                "Scope": "control" if site == "HIN" else "target",
                "PDF_Page": page, "Printed_Page": printed, "Column": column,
                "Source_Group_Labels": labels, "Manual_Transcription": form,
                "Manual_Form_Count": str(len(form.split(" / "))),
                "Source_Qualifier": qualifier, "Review_Status": "attested",
                "Confidence": "high", "Site_Assignment_Confidence": site_confidence,
                "Uncertainty": uncertainty, "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 80
    assert all(row["Review_Status"] == "attested" for row in rows)
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
