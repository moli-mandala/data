#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 166-170."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_166_170_hand_keyed.tsv")
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


def cell(form, labels="1", page="61", printed="56", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/2400-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    166: ("what", {
        **{
            site: cell("ka", page="60", printed="55", column="right")
            for site in SITES
        },
        "HIN": cell("kja", page="60", printed="55", column="right"),
        "SkP": cell("kja", page="60", printed="55", column="right"),
        "KkP": cell("ka"),
        "CCC": cell("kʌtʰi", labels="2"),
    }),
    167: ("where", {
        **{site: cell("kʌhã") for site in SITES},
        "SkP": cell("kʌha"),
        "RKM": cell("kʌha"),
        "DDK": cell("kʌha"),
        "RNK": cell("kʌhãko"),
        "RNS_Sisana": cell("kʌhãko"),
        "CCC": cell(None),
    }),
    168: ("when", {
        **{site: cell("kʌb") for site in SITES},
        "CCC": cell("dʒʌb", labels="2"),
        "DGC": cell(
            "kʌb / kʌhija", labels="1 / 3",
            qualifier="second response: (future)",
        ),
    }),
    169: ("how many", {
        "HIN": cell("kitne"),
        "BNM": cell("kitne"),
        "BNT": cell("kitne"),
        "DDK": cell("kʌtʌna"),
        "DGC": cell("kʌtʌna / kæitʰo", labels="1 / 3", column="left / right"),
        "DkR": cell("kʌtʌɾa"),
        "KkP": cell("kʌtʌɾa"),
        "DKS": cell("kʌtʌɾa"),
        "SkP": cell("kʌtːa / kʌtːa", labels="3 / 4", column="left / right"),
        "RKM": cell("kitːo", labels="3"),
        "RKB": cell("kitːo", labels="3"),
        "RNK": cell("kitːe", labels="3"),
        "RNS_Sisaikhara": cell("kitːe", labels="3"),
        "RNS_Sisana": cell("kitːe", labels="3", column="right"),
        "TkN": cell("kitːe", labels="3"),
        "CCC": cell("katek", labels="4", column="right"),
    }),
    170: ("what kind", {
        "HIN": cell("kispɾʌkaɾ / kæsa", labels="1 / 6", column="right"),
        "BNM": cell("kispɾʌkaɾ", column="right"),
        "BNT": cell("kispɾʌkaɾ / konkon", labels="1 / 8", column="right"),
        "RNK": cell(
            "kɔnsitʌɾʌhʌko / konsitʌɾʌhʌko", labels="2 / 3", column="right"
        ),
        "RNS_Sisaikhara": cell(
            "kɔnsitʌɾʌhʌko / konso / konsitʌɾʌhʌko",
            labels="2 / 2 / 3", column="right",
        ),
        "RNS_Sisana": cell("kɔnsi", labels="2", column="right"),
        "RKM": cell("kitʌnekisʌm", labels="9", column="right"),
        "RKB": cell("kæso", labels="6", column="right"),
        "TkN": cell("kontʌɾikako", labels="3", column="right"),
        "KkP": cell("kæse / kæsʌn", labels="6 / 6", column="right"),
        "SkP": cell("kʌtːaekaɾ", labels="5", column="right"),
        "DKS": cell("kæisin / kɔnmeɾ", labels="6 / 7", column="right"),
        "DDK": cell("kæisin / kɔnmeɾ", labels="6 / 7", column="right"),
        "DGC": cell("kɔnseɾ / kɔʊnmeɾ", labels="2 / 7", column="right"),
        "DkR": cell("kʌtʌɾaɾʌ̃ŋke", labels="4", column="right"),
        "CCC": cell(None, labels="", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(166, 171):
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
                    "metadata row order"
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 90
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 84
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
