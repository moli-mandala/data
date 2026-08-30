#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 41-45."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_041_045_hand_keyed.tsv")
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


def cell(form, labels, page=39, printed=34, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 900/1200-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    41: ("sun", {
        "HIN": cell("suɾʌdʒ", "1", page=38, printed=33, column="right"),
        "BNM": cell("suɾʌdʒ", "1", page=38, printed=33, column="right"),
        "BNT": cell("suɾʌdʒ / din", "1 / 2", page="38 / 39", printed="33 / 34", column="right / left"),
        "RNK": cell("din", "2", page=38, printed=33, column="right"),
        "RNS_Sisaikhara": cell("din", "2"),
        "DGC": cell("din", "2"),
        "DkR": cell("din", "2"),
        "SkP": cell("din", "2"),
        "RKB": cell("din", "2"),
        "TkN": cell("din", "2"),
        "DKS": cell("din", "2"),
        "RKM": cell("din", "2"),
        "RNS_Sisana": cell("din", "2"),
        "DDK": cell("din", "2"),
        "KkP": cell("din", "2"),
        "CCC": cell("ɡʰam / beɾia", "3 / 4"),
    }),
    42: ("moon", {
        "HIN": cell("tʃãnd / tʃʌnduma", "1 / 2"),
        "RNK": cell("dʒoni", "3"),
        "BNM": cell("dʒoni", "3"),
        "TkN": cell("dʒoni", "3"),
        "BNT": cell("dʒoni", "3"),
        "RKM": cell("dʒoni", "3"),
        "RNS_Sisaikhara": cell("dʒoni", "3"),
        "RNS_Sisana": cell("dʒõni", "3"),
        "RKB": cell("dʒõnih", "3"),
        "DkR": cell("dʒonihija", "3"),
        "DKS": cell("dʒonija", "3"),
        "DDK": cell("dʒoniha", "3"),
        "KkP": cell("dʒʰonha", "3"),
        "DGC": cell("ʌdʒeɾija", "4"),
        "SkP": cell("ɾat", "5"),
        "CCC": cell("dʰoɾ", "6"),
    }),
    43: ("sky", {
        "HIN": cell("akaʃ", "1"),
        "TkN": cell("akaʃ", "1"),
        "BNT": cell("akaʃ / badʌl", "1 / 2"),
        "BNM": cell("akas / badʌl", "1 / 2"),
        "KkP": cell("akas", "1"),
        "CCC": cell("ʌkas", "1"),
        "RNK": cell("badʌɾ", "2"),
        "RNS_Sisaikhara": cell("badʌɾ", "2"),
        "RKB": cell("badʌɾ / badʌl", "2 / 2"),
        "RKM": cell("badʌɾ", "2"),
        "RNS_Sisana": cell("badʌɾ", "2"),
        "DGC": cell("bʌdʌɾi", "2"),
        "DKS": cell("bʌdʌɾi", "2"),
        "DDK": cell("bʌdʌɾi", "2"),
        "SkP": cell("badɾi", "2"),
        "DkR": cell("uppʌɾ", "3"),
    }),
    44: ("star", {
        "HIN": cell("taɾe / sitaɾa", "1 / 3", column="right"),
        "BNM": cell("taɾe", "1", column="right"),
        "BNT": cell("taɾe", "1", column="right"),
        "RNS_Sisaikhara": cell("taɾe", "1", column="right"),
        "RNK": cell("taɾa", "1", column="right"),
        "RNS_Sisana": cell("taɾa", "1", column="right"),
        "RKB": cell("taɾa", "1", column="right"),
        "TkN": cell("taɾa", "1", column="right"),
        "RKM": cell("taɾa", "1", column="right"),
        "KkP": cell("taɾa", "1", column="right"),
        "DKS": cell("toɾæ", "1", column="right"),
        "DkR": cell("tõɾĩja", "1", column="right"),
        "SkP": cell("tʌɾʌi + ja", "1", column="right"),
        "DDK": cell("toɾʌjʌ", "1", column="right"),
        "DGC": cell("taɾʌɡʌn", "2", column="right"),
        "CCC": cell("taɾaɡun", "2", column="right"),
    }),
    45: ("rain", {
        "HIN": cell("baɾiʃ / wʌɾʃa", "1 / 8", column="right"),
        "SkP": cell("bʌɾiʃ", "1", column="right"),
        "RNK": cell("mẽh", "2", column="right"),
        "RNS_Sisaikhara": cell("mẽh", "2", column="right"),
        "RKM": cell("mẽh", "2", column="right"),
        "RNS_Sisana": cell("mẽh", "2", column="right"),
        "RKB": cell("mẽ", "2", column="right"),
        "BNM": cell("bʌɾkʰa", "3", column="right"),
        "DKS": cell("bʌɾkʰa", "3", column="right"),
        "BNT": cell("bʌɾkʰa", "3", column="right"),
        "DGC": cell("bʌɾkʰa", "3", column="right"),
        "CCC": cell("baɾkʰa / tʃʰaɾi", "3 / 7", column="right"),
        "DDK": cell("baɾkʰa", "3", column="right"),
        "DkR": cell("pani", "5", column="right", qualifier="(46)"),
        "TkN": cell("bʌʃʌti", "6", column="right"),
        "KkP": cell("bʌɾʃʌt / bʌɾsʌt", "6 / 8", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(41, 46):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 90
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 82
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
