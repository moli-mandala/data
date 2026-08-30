#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 206-210."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_206_210_hand_keyed.tsv")
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


def cell(form, labels="1", page="68", printed="63", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1800-dpi rendered-page crops before any
# comparison with the legacy TSV. Item 206 prints only a heading and no rows.
ITEMS = {
    206: ("she", {site: cell(None) for site in SITES}),
    207: ("we (inc.)", {
        "HIN": cell("hʌm"),
        "RNK": cell("hʌm"),
        "RNS_Sisaikhara": cell("hʌm / hʌmsʌb", labels="1 / 2"),
        "BNM": cell("hʌm"),
        "RKB": cell("hʌm"),
        "BNT": cell("hʌm"),
        "RKM": cell("hʌm"),
        "RNS_Sisana": cell("hʌm"),
        "DGC": cell("hʌmre"),
        "DkR": cell("hʌmʌre"),
        "KkP": cell("hʌmʌre"),
        "DKS": cell("hʌmʌreh"),
        "CCC": cell("hamara"),
        "DDK": cell("hʌmʌrẽ / hʌmʌrẽsʌb", labels="1 / 2"),
        "SkP": cell("hʌmsʌb", labels="2"),
        "TkN": cell("apʌnsʌb", labels="3"),
    }),
    208: ("we (exc.)", {
        "HIN": cell("hʌm", qualifier="response followed by (207)"),
        "RNK": cell("hʌm", qualifier="response followed by (207)"),
        "RNS_Sisaikhara": cell("hʌm", qualifier="response followed by (207)"),
        "BNM": cell("hʌm", qualifier="response followed by (207)"),
        "RKB": cell("hʌm", qualifier="response followed by (207)"),
        "BNT": cell("hʌm", qualifier="response followed by (207)"),
        "RKM": cell("hʌm", qualifier="response followed by (207)"),
        "DGC": cell("hʌmre", qualifier="response followed by (207)"),
        "SkP": cell("hʌm"),
        "RNS_Sisana": cell("hʌm"),
        "DKS": cell("hʌmʌre", qualifier="response followed by (207)"),
        "KkP": cell("hʌmʌre", qualifier="response followed by (207)"),
        "DDK": cell("hʌmʌrẽ"),
        "DkR": cell("mæj", labels="2", qualifier="response followed by (202)"),
        "TkN": cell("mæ̃", labels="2", qualifier="response followed by (202)"),
        "CCC": cell(None),
    }),
    209: ("you (2nd pl.)", {
        "HIN": cell("aplog", column="right"),
        "RNK": cell("tum", labels="2", column="right"),
        "KkP": cell("tum / tumʌreh", labels="2 / 4", column="right"),
        "SkP": cell("tʊm", labels="2", column="right"),
        "RNS_Sisaikhara": cell("tʊm / tumlog", labels="2 / 3", column="right"),
        "BNT": cell(
            "tʊm", labels="2", column="right", qualifier="response followed by (203)"
        ),
        "RNS_Sisana": cell("tumlog", labels="3", column="right"),
        "BNM": cell("tumlog", labels="3", column="right"),
        "RKB": cell("tumlog", labels="3", column="right"),
        "DGC": cell("tohʌre / tɔhi", labels="4 / 8", column="right"),
        "DDK": cell("tohʌre / tu", labels="4 / 7", column="right"),
        "DKS": cell("ʈureh", labels="4", column="right"),
        "DkR": cell(
            "tæ̃", labels="5", column="right", qualifier="response followed by (203)"
        ),
        "TkN": cell(
            "tæ̃", labels="5", column="right", qualifier="response followed by (203)"
        ),
        "RKM": cell(
            "hʌm", labels="6", column="right", qualifier="response followed by (206)"
        ),
        "CCC": cell(None, column="right"),
    }),
    210: ("they", {
        "HIN": cell("ve", column="right"),
        "RNK": cell("ve", column="right"),
        "RNS_Sisaikhara": cell("ve / vou", labels="1 / 2", column="right"),
        "RKB": cell("ve", column="right", qualifier="response followed by (174)"),
        "BNT": cell("ve", column="right", qualifier="response followed by (174)"),
        "BNM": cell("be", column="right"),
        "TkN": cell("be", column="right", qualifier="response followed by (174)"),
        "RKM": cell("voʊ", labels="2", column="right"),
        "RNS_Sisana": cell(
            "vo", labels="2", column="right", qualifier="response followed by (205)"
        ),
        "DGC": cell("vʌjʌne", labels="3", column="right"),
        "DKS": cell("vohine", labels="3", column="right"),
        "DkR": cell("u", labels="4", column="right", qualifier="response followed by (205)"),
        "SkP": cell("u", labels="4", column="right", qualifier="response followed by (205)"),
        "DDK": cell("u", labels="4", column="right", qualifier="response followed by (205)"),
        "KkP": cell("u", labels="4", column="right", qualifier="response followed by (205)"),
        "CCC": cell("hunkasabʰ", labels="5", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(206, 211):
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
                    "duplicate source code RNS; within-item and within-group occurrence "
                    "order provisionally assigned to metadata row order"
                )
            if blank:
                uncertainty = "site code absent from the complete printed item block"
                if site.startswith("RNS_"):
                    uncertainty = (
                        "duplicate source code RNS; no RNS response is printed in the "
                        "complete item block"
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 62
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 18
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 69
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 65
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
