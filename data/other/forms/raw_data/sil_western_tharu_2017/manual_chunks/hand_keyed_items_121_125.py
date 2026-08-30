#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 121-125."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_121_125_hand_keyed.tsv")
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


def cell(form, labels="1", page="53", printed="48", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600-dpi rendered-page crops before
# any comparison with the legacy CSV.
ITEMS = {
    121: ("evening", {
        "HIN": cell("ʃam / saĩ", "1 / 6", column="left / right"),
        "RNK": cell("sʌndʒʰa", "2", column="left"),
        "RNS_Sisaikhara": cell("sʌndʒʰa", "2", column="left"),
        "DGC": cell("sʌndʒʰa", "2", column="left"),
        "DkR": cell("sʌndʒʰa", "2", column="left"),
        "RNS_Sisana": cell("sʌndʒʰa", "2", column="left"),
        "DDK": cell("sʌndʒʰa", "2", column="left"),
        "BNM": cell("sãntʃ / etʃ", "2 / 4", column="left / right"),
        "SkP": cell("sãndʒʰ", "2", column="left"),
        "KkP": cell("sãndʒʰ", "2", column="left"),
        "RKB": cell("sʌ̃dʒʰ", "2", column="left"),
        "TkN": cell("sʌndʒa", "2", column="left"),
        "RKM": cell("sʌndʒa", "2", column="left"),
        "DKS": cell("sʌndʒa / sahidʒʊn", "2 / 3", column="left"),
        "CCC": cell("saːdʒʰ", "2", column="left"),
        "BNT": cell("ɾat", "5", qualifier="(118)"),
    }),
    122: ("yesterday", {
        "HIN": cell("kʌl"),
        "RNK": cell("kʌl"),
        "RNS_Sisaikhara": cell("kʌl"),
        "BNM": cell("kʌl"),
        "TkN": cell("kʌl"),
        "DKS": cell("kʌl"),
        "BNT": cell("kʌl"),
        "RKM": cell("kʌl"),
        "RNS_Sisana": cell("kʌl"),
        "KkP": cell("kʌl"),
        "DGC": cell("kal"),
        "DkR": cell("kal"),
        "SkP": cell("kal"),
        "RKB": cell("kal"),
        "DDK": cell("kal"),
        "CCC": cell("kalu"),
    }),
    123: ("today", {
        "HIN": cell("adʒ"),
        "RNK": cell("adʒ"),
        "RNS_Sisaikhara": cell("adʒ"),
        "BNM": cell("adʒ"),
        "DGC": cell("adʒ"),
        "DkR": cell("adʒ"),
        "SkP": cell("adʒ"),
        "RKB": cell("adʒ"),
        "TkN": cell("adʒ"),
        "BNT": cell("adʒ"),
        "RKM": cell("adʒ"),
        "RNS_Sisana": cell("adʒ"),
        "DDK": cell("adʒ"),
        "KkP": cell("adʒ"),
        "DKS": cell("adʒʊ"),
        "CCC": cell("adʒʊ"),
    }),
    124: ("tomorrow", {
        "HIN": cell("kʌl", qualifier="(122)"),
        "RNK": cell("kʌl", qualifier="(122)"),
        "RNS_Sisaikhara": cell("kʌl", qualifier="(122)"),
        "BNM": cell("kʌl", qualifier="(122)"),
        "TkN": cell("kʌl", qualifier="(122)"),
        "DKS": cell("kʌl", qualifier="(122)"),
        "BNT": cell("kʌl", qualifier="(122)"),
        "RKM": cell("kʌl", qualifier="(122)"),
        "RNS_Sisana": cell("kʌl", qualifier="(122)"),
        "KkP": cell("kʌl", qualifier="(122)"),
        "RKB": cell("kʌl"),
        "DDK": cell("kal"),
        "DGC": cell("kal", qualifier="(122)"),
        "DkR": cell("kal", page="54", printed="49", column="left", qualifier="(122)"),
        "SkP": cell("kal", page="54", printed="49", column="left", qualifier="(122)"),
        "CCC": cell("andini", "2", page="54", printed="49", column="left"),
    }),
    125: ("week", {
        "HIN": cell("hʌftah", page="54", printed="49", column="left"),
        "RNK": cell("hʌftah", page="54", printed="49", column="left"),
        "RNS_Sisaikhara": cell("hʌftah", page="54", printed="49", column="left"),
        "RNS_Sisana": cell(None, "", page="54", printed="49", column="left"),
        "BNM": cell("hʌftah", page="54", printed="49", column="left"),
        "DGC": cell("hʌpta", page="54", printed="49", column="left"),
        "SkP": cell("hʌpta", page="54", printed="49", column="left"),
        "DkR": cell("hʌpʈa", page="54", printed="49", column="left"),
        "TkN": cell("hʌpʰta", page="54", printed="49", column="left"),
        "RKM": cell("hʌpʰta", page="54", printed="49", column="left"),
        "DKS": cell("hʌptʌh", page="54", printed="49", column="left"),
        "RKB": cell(
            "hʌptah / aʈʰʌdin", "1 / 2", page="54", printed="49",
            column="left", qualifier="second response: (used most)",
        ),
        "KkP": cell("hʌptah", page="54", printed="49", column="left"),
        "CCC": cell("hʌpta", page="54", printed="49", column="left"),
        "BNT": cell(None, "", page="54", printed="49", column="left"),
        "DDK": cell(None, "", page="54", printed="49", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(121, 126):
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
                if site == "RNS_Sisana":
                    uncertainty = (
                        "duplicate source code RNS; only one RNS response is printed; "
                        "metadata row 2 has no independently assignable response"
                    )
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 75
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
