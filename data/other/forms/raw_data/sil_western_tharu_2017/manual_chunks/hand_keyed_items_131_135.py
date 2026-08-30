#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 131-135."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_131_135_hand_keyed.tsv")
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


def cell(form, labels="1", page="55", printed="50", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600-dpi rendered-page crops before
# any comparison with the legacy CSV.
ITEMS = {
    131: ("bad", {
        "HIN": cell("gʌnda / bekʌɾ / kʰʌɾab / bura", "1 / 4 / 6 / 9"),
        "BNM": cell("gʌnda"),
        "BNT": cell("gʌnda"),
        "TkN": cell("gʌndo"),
        "RNK": cell("tʃʰija", "2"),
        "RNS_Sisaikhara": cell("tʃʰija", "2"),
        "RKB": cell("tʃʰija", "2"),
        "RNS_Sisana": cell("tʃʰija", "2"),
        "SkP": cell("tʃʰɪʈɔn", "3"),
        "DGC": cell("bekʌɾ", "4"),
        "DkR": cell("gʌndhʌjʌna", "5"),
        "RKM": cell("kʰʌɾab", "6"),
        "CCC": cell("kʰʌɾab / badʌmas", "6 / 8"),
        "DDK": cell("nʌhimʌdʒa", "7"),
        "DKS": cell("nʌhimʌdʒa", "7"),
        "KkP": cell(
            "bʰuhʌɾ / mælʌha", "9 / 9",
            qualifier="first response: (person); second response: (object)",
        ),
    }),
    132: ("wet", {
        "HIN": cell("bʰiga / gila", "1 / 2"),
        "BNM": cell("bʰiga / bʰidʒ", "1 / 1"),
        "RKB": cell("bʰidʒa"),
        "TkN": cell("bʰidʒo"),
        "RNK": cell("bʰidʒ"),
        "RNS_Sisaikhara": cell("bʰidʒ"),
        "DGC": cell("bʰidʒ"),
        "DKS": cell("bʰidʒ"),
        "BNT": cell("bʰidʒ"),
        "DDK": cell("bʰidʒ"),
        "SkP": cell("bʰidʒʌl"),
        "CCC": cell("bʰidʒʌl"),
        "KkP": cell("bʰidʒʌl"),
        "DkR": cell("bʰidʒʌgil"),
        "RKM": cell("bʰidʒʌna"),
        "RNS_Sisana": cell("bʰidʒʌna"),
    }),
    133: ("dry", {
        "HIN": cell("sukʰa"),
        "BNM": cell("sukʰa"),
        "DGC": cell("sukʰa", column="right"),
        "BNT": cell("sukʰa", column="right"),
        "RNK": cell("sukʰo", column="right"),
        "RNS_Sisaikhara": cell("sukʰo", column="right"),
        "TkN": cell("sukʰo", column="right"),
        "RNS_Sisana": cell("sukʰo", column="right"),
        "SkP": cell("sukʰʌl", column="right"),
        "KkP": cell("sukʰʌl", column="right"),
        "DKS": cell("sukʰʌl", column="right"),
        "RKB": cell("sukʰo", column="right"),
        "RKM": cell("sukhʌna", column="right"),
        "DkR": cell("sogʌgil", column="right"),
        "DDK": cell("sukʰago", column="right"),
        "CCC": cell("sukʰaili", column="right"),
    }),
    134: ("long", {
        "HIN": cell("lʌmba", column="right"),
        "BNM": cell("lʌmba", column="right"),
        "DGC": cell("lʌmba", column="right"),
        "DkR": cell("lʌmba", column="right"),
        "SkP": cell("lʌmba", column="right"),
        "KkP": cell("lʌmba", column="right"),
        "RNK": cell("lʌmbõ", column="right"),
        "RNS_Sisaikhara": cell("lʌmbõ", column="right"),
        "RKB": cell("lʌmbo", column="right"),
        "TkN": cell("lʌmbo", column="right"),
        "BNT": cell("lʌmbo", column="right"),
        "RKM": cell("lʌmbo", column="right"),
        "RNS_Sisana": cell("lʌmbo", column="right"),
        "DKS": cell("nʌmba", column="right"),
        "DDK": cell("lamma / dʰẽɖ", "1 / 3", column="right"),
        "CCC": cell("nʌmʌhaɾa", "2", column="right"),
    }),
    135: ("short", {
        "HIN": cell("tʃʰoʈa / tʃʰoʈa", "1 / 3", column="right"),
        "BNM": cell("tʃʰoʈa / tʃʰoʈa", "1 / 3", column="right"),
        "RNK": cell("tʃʰoʈo / tʃʰoʈo", "1 / 3", page="55 / 56", printed="50 / 51", column="right / left"),
        "RNS_Sisaikhara": cell("tʃʰoʈo / tʃʰoʈo", "1 / 3", page="55 / 56", printed="50 / 51", column="right / left"),
        "RKB": cell("tʃʰoʈo / tʃʰoʈo", "1 / 3", page="55 / 56", printed="50 / 51", column="right / left"),
        "TkN": cell("tʃʰoʈo / tʃʰoʈo", "1 / 3", page="55 / 56", printed="50 / 51", column="right / left"),
        "RKM": cell("tʃʰoʈo / tʃʰoʈo", "1 / 3", page="55 / 56", printed="50 / 51", column="right / left"),
        "RNS_Sisana": cell("tʃʰoʈo / tʃʰoʈo", "1 / 3", page="55 / 56", printed="50 / 51", column="right / left"),
        "CCC": cell("tʃʰoʈe", column="right"),
        "DDK": cell("tʃʰoʈ", column="right"),
        "KkP": cell("tʃʰoʈ / tʃoʈimoti", "1 / 1", column="right"),
        "DKS": cell("tʃʰoʈ", column="right"),
        "DGC": cell("tʃʰoʈʌmoʈ", column="right"),
        "DkR": cell("tʃoʈimoti", column="right"),
        "SkP": cell("tʃʌwaʈ", "2", column="right"),
        "BNT": cell("gʌʈa", "3", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(131, 136):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 97
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 87
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
