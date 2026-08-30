#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 171-175."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_171_175_hand_keyed.tsv")
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


def cell(form, labels="1", page="62", printed="57", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/2400-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    171: ("this", {
        "HIN": cell("jʌh", page="61", printed="56", column="right"),
        "BNM": cell("jʌh", page="61", printed="56", column="right"),
        "BNT": cell("jʌh", page="61", printed="56", column="right"),
        "RNS_Sisaikhara": cell("dʒʌw", labels="2", page="61", printed="56", column="right"),
        "RNK": cell("dʒʌw", labels="2", page="61", printed="56", column="right"),
        "RKB": cell("dʒa", labels="2", page="61", printed="56", column="right"),
        "TkN": cell("dʒa", labels="2", page="61", printed="56", column="right"),
        "RKM": cell("dʒɔ", labels="2", page="61", printed="56", column="right"),
        "RNS_Sisana": cell("dʒʌ", labels="2", page="61", printed="56", column="right"),
        "DkR": cell("i", labels="3", page="61", printed="56", column="right"),
        "SkP": cell("i", labels="3", page="61", printed="56", column="right"),
        "DKS": cell("i", labels="3", page="61", printed="56", column="right"),
        "CCC": cell("i", labels="3", page="61", printed="56", column="right"),
        "DDK": cell("i", labels="3", page="61", printed="56", column="right"),
        "DGC": cell("i", labels="3", page="61", printed="56", column="right"),
        "KkP": cell("i", labels="3", page="61", printed="56", column="right"),
    }),
    172: ("that", {
        "HIN": cell("vʌh", page="61", printed="56", column="right"),
        "BNT": cell("vʌh", page="61", printed="56", column="right"),
        "RKM": cell("vʌh", page="61", printed="56", column="right"),
        "BNM": cell("ve", page="61", printed="56", column="right"),
        "RNK": cell("dʒʌb", labels="2"),
        "RNS_Sisaikhara": cell("bo / hʊn", labels="3 / 6"),
        "RKB": cell("boh", labels="3"),
        "DGC": cell("vohe / u", labels="3 / 4"),
        "DkR": cell("u", labels="4"),
        "SkP": cell("u", labels="4"),
        "DKS": cell("u", labels="4"),
        "CCC": cell("u", labels="4"),
        "KkP": cell("u", labels="4"),
        "DDK": cell("ʊ", labels="4"),
        "TkN": cell("dʒa", labels="5", qualifier="(171)"),
        "RNS_Sisana": cell(
            None, labels="", page="61 / 62", printed="56 / 57", column="right / left"
        ),
    }),
    173: ("these", {
        "HIN": cell("je / in", labels="1 / 8"),
        "DKS": cell("je"),
        "BNT": cell("jẽ"),
        "BNM": cell("jẽ / ɪtna", labels="1 / 3"),
        "RKB": cell("be"),
        "RNK": cell("dʒɔ", labels="2"),
        "RNS_Sisaikhara": cell("dʒɔ", labels="2"),
        "RKM": cell("dʒo", labels="2"),
        "RNS_Sisana": cell("dʒʌw", labels="2"),
        "TkN": cell("dʒe", labels="2"),
        "DkR": cell("i", labels="4", qualifier="(171)"),
        "SkP": cell("i", labels="4", qualifier="(171)"),
        "DGC": cell("i / ajne", labels="4 / 7", qualifier="first response: (171)"),
        "KkP": cell("i", labels="4", qualifier="(171)"),
        "DDK": cell("i / tæ", labels="4 / 6", qualifier="first response: (171)"),
        "CCC": cell("ia", labels="5"),
    }),
    174: ("those", {
        "HIN": cell("ve"),
        "BNM": cell("ve"),
        "BNT": cell("ve"),
        "TkN": cell("be"),
        "RNK": cell("vo", labels="2"),
        "RKM": cell("vo", labels="2"),
        "RNS_Sisaikhara": cell("bo", labels="2"),
        "RKB": cell("bo", labels="2"),
        "RNS_Sisana": cell("bo", labels="2"),
        "DkR": cell("u", labels="3", qualifier="(171)"),
        "SkP": cell("u", labels="3", qualifier="(171)"),
        "DGC": cell("u", labels="3", qualifier="(171)"),
        "KkP": cell("u", labels="3", qualifier="(171)"),
        "DDK": cell("ʊ", labels="3", qualifier="(171)"),
        "CCC": cell("ua", labels="4"),
        "DKS": cell("de", labels="5"),
    }),
    175: ("same", {
        "HIN": cell("eksʌman / ekse / sʌman", labels="1 / 3 / b", column="right"),
        "TkN": cell("eksʌman", column="right"),
        "RNK": cell("ekdʒæse", labels="2", column="right"),
        "RNS_Sisaikhara": cell("ekdʒæse", labels="2", column="right"),
        "RNS_Sisana": cell("ekdʒæse", labels="2", column="right"),
        "BNM": cell("ekæ", labels="3", column="right"),
        "DkR": cell("ekʌnas", labels="5", column="right"),
        "DKS": cell("ekʌnas", labels="5", column="right"),
        "SkP": cell("ʌkekʌs", labels="5", column="right"),
        "DDK": cell("akːenas", labels="5", column="right"),
        "DGC": cell("ekːægʰʌs / ekːæmeɾ", labels="5 / 6", column="right"),
        "KkP": cell("eketaɾ", labels="6", column="right"),
        "RKB": cell("ekhani", labels="7", column="right"),
        "BNT": cell("bʌɾʌbʌɾ", labels="8", column="right"),
        "RKM": cell("ekʌjtʌɾʌh", labels="9", column="right"),
        "CCC": cell("ɾitto", labels="a", column="right", qualifier="(alike)"),
    }),
}


def main() -> None:
    rows = []
    for item in range(171, 176):
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
                if site.startswith("RNS_"):
                    uncertainty = (
                        "duplicate source code RNS; only one RNS response per printed group; "
                        "both group responses assigned to metadata row 1, leaving row 2 blank"
                    )
                else:
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 88
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 80
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
