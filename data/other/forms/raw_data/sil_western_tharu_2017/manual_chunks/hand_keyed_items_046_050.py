#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 46-50."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_046_050_hand_keyed.tsv")
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


def cell(form, labels, page=40, printed=35, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 900/1200-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    46: ("water", {
        "HIN": cell("pani / dʒʌl", "1 / 2", page="39 / 40", printed="34 / 35", column="right / left"),
        "RNK": cell("pani", "1", page=39, printed=34, column="right"),
        "RNS_Sisaikhara": cell("pani", "1", page=39, printed=34, column="right"),
        "BNM": cell("pani", "1", page=39, printed=34, column="right"),
        "DGC": cell("pani", "1", page=39, printed=34, column="right"),
        "DkR": cell("pani", "1", page=39, printed=34, column="right"),
        "SkP": cell("pani", "1", page=39, printed=34, column="right"),
        "RKB": cell("pani", "1", page=39, printed=34, column="right"),
        "TkN": cell("pani", "1", page=39, printed=34, column="right"),
        "DKS": cell("pani", "1", page=39, printed=34, column="right"),
        "BNT": cell("pani", "1", page=39, printed=34, column="right"),
        "RKM": cell("pani", "1", page=39, printed=34, column="right"),
        "RNS_Sisana": cell("pani", "1", page=39, printed=34, column="right"),
        "CCC": cell("pani", "1"),
        "DDK": cell("pani", "1"),
        "KkP": cell("pani", "1"),
    }),
    47: ("river", {
        "HIN": cell("nʌdi", "1"),
        "BNT": cell("nʌdi", "1"),
        "RNK": cell("nʌdija", "1"),
        "RNS_Sisaikhara": cell("nʌdija", "1"),
        "DGC": cell("nʌdija / dundʌɾa", "1 / 2"),
        "RKB": cell("nʌdija", "1"),
        "TkN": cell("nʌdija", "1"),
        "RNS_Sisana": cell("nʌdija", "1 1", qualifier="extra printed 1 after group label"),
        "BNM": cell("nʌ̃ndi", "1"),
        "DkR": cell("lʌdija", "1"),
        "DKS": cell("lʌdija", "1"),
        "KkP": cell("lʌdija / nandi", "1 / 1"),
        "SkP": cell("lʌʈi", "1"),
        "RKM": cell("nadija", "1"),
        "DDK": cell("lʌɖijʌ", "1"),
        "CCC": cell("lʌdi", "1"),
    }),
    48: ("cloud", {
        "HIN": cell("badʌl", "1"),
        "BNM": cell("badʌl", "1"),
        "BNT": cell("badʌl", "1", qualifier="(43)"),
        "RNK": cell("badʌɾi", "1"),
        "RNS_Sisaikhara": cell("badʌɾi", "1"),
        "DkR": cell("badʌɾi", "1"),
        "CCC": cell("badʌɾi", "1"),
        "DGC": cell("badʌɾi", "1", qualifier="(43)"),
        "DKS": cell("badʌɾi", "1", qualifier="(43)"),
        "RKB": cell("badɾi", "1"),
        "SkP": cell("badɾi", "1", qualifier="(43)"),
        "TkN": cell("badʌɾ", "1"),
        "RKM": cell("badʌɾ", "1", qualifier="(43)"),
        "RNS_Sisana": cell("bʌdʌɾija", "1"),
        "DDK": cell("bʌdɾi", "1", qualifier="(43)"),
        "KkP": cell("badɾi", "1"),
    }),
    49: ("lightning", {
        "HIN": cell("bidʒʌli", "1"),
        "RNK": cell("bidʒʌli", "1"),
        "RNS_Sisaikhara": cell("bidʒʌli", "1"),
        "BNM": cell("bidʒʌli", "1"),
        "DkR": cell("bidʒʌli", "1"),
        "SkP": cell("bidʒʌli", "1"),
        "TkN": cell("bidʒʌli", "1"),
        "DKS": cell("bidʒʌli", "1"),
        "BNT": cell("bidʒʌli", "1"),
        "RKM": cell("bidʒʌli", "1"),
        "RNS_Sisana": cell("bidʒʌli", "1", column="right"),
        "KkP": cell("bidʒʌli", "1", column="right"),
        "RKB": cell("biŋʌli", "1", column="right"),
        "CCC": cell("bidʒuli / tʃilʌkai", "1 / 3", column="right"),
        "DGC": cell("bʌdʌɾitʃʌmʌkʌtʰæ", "2", column="right"),
        "DDK": cell("tʃʌmʌkʌʈa", "2", column="right"),
    }),
    50: ("rainbow", {
        "HIN": cell("indɾʌdʰʌnuʃ", "1", column="right"),
        "RNK": cell("dʰʌnʌʃban", "2", column="right"),
        "RNS_Sisaikhara": cell("dʰʌnuʃ", "2", column="right"),
        "BNM": cell("dʰʌnʌkʌman", "2", column="right"),
        "BNT": cell("dʰʌnʌkʌman", "2", column="right"),
        "RKB": cell("dʰʌnʌkman", "2", column="right"),
        "TkN": cell("dʰʌnʌkman", "2", column="right"),
        "SkP": cell("dʰʌnuʃman", "2", column="right"),
        "RNS_Sisana": cell("dʰʌnuʃman", "2", column="right"),
        "RKM": cell("dʰanuʃman", "2", column="right"),
        "KkP": cell("dʰanʌkban", "2", column="right"),
        "DDK": cell("ɖʰʌni", "3", column="right"),
        "DGC": cell("indɾʌdʰʌnuʃ / dʰʌnuhi", "1 / 3", column="right"),
        "DKS": cell("dʰʌnikʰʌni", "3", column="right"),
        "DkR": cell("ɾamʌtʃʌɾʌnketʃʰani", "4", column="right"),
        "CCC": cell(None, "", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(46, 51):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 84
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 78
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
