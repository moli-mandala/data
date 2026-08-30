#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 141-145."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_141_145_hand_keyed.tsv")
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


def cell(form, labels="1", page="57", printed="52", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600/2400-dpi rendered-page crops
# before any comparison with the legacy CSV.
ITEMS = {
    141: ("far", {
        "HIN": cell("duɾ", page="56", printed="51", column="right"),
        "RNK": cell("duɾ", page="56", printed="51", column="right"),
        "RNS_Sisaikhara": cell("duɾ"),
        "BNM": cell("duɾ"),
        "DGC": cell("duɾ"),
        "DkR": cell("duɾ"),
        "SkP": cell("duɾ"),
        "RKB": cell("duɾ"),
        "BNT": cell("duɾ"),
        "RKM": cell("duɾ"),
        "RNS_Sisana": cell("duɾ"),
        "DDK": cell("duɾ"),
        "KkP": cell("duɾ"),
        "TkN": cell("duɾ"),
        "DKS": cell("dʊɾ"),
        "CCC": cell("tʌnau", "2"),
    }),
    142: ("big", {
        "HIN": cell("bʌɽa"),
        "BNM": cell("bʌɽa"),
        "SkP": cell("bʌɽa"),
        "RKM": cell("bʌɽa"),
        "RNS_Sisaikhara": cell("bʌɽo"),
        "RNK": cell("bʌɽo"),
        "TkN": cell("bʌɽo"),
        "RKB": cell("bʌdo"),
        "DGC": cell("bʰaɽi"),
        "DkR": cell("bʰaɾi"),
        "DKS": cell("bʰaɾi"),
        "RNS_Sisana": cell("bʰaɾi"),
        "DDK": cell("bʰaɾi"),
        "KkP": cell("bʰaɾi"),
        "BNT": cell("baɖa"),
        "CCC": cell("dʒabʌɖe"),
    }),
    143: ("small", {
        "HIN": cell("tʃʰoʈa", qualifier="(135)"),
        "BNM": cell("tʃʰoʈa", qualifier="(135)"),
        "DKS": cell("tʃʰoʈa"),
        "BNT": cell("tʃʰoʈa"),
        "RKM": cell("tʃʰoʈa"),
        "TkN": cell("tʃʰoʈo", qualifier="(135)"),
        "RNK": cell("tʃʰoʈo", qualifier="(135)"),
        "RNS_Sisaikhara": cell("tʃʰoʈo", qualifier="(135)"),
        "RNS_Sisana": cell("tʃʰoʈo", qualifier="(135)"),
        "DGC": cell("tʃʰoʈʌ"),
        "RKB": cell("tʃʰoʈo"),
        "CCC": cell("tʃʰoʈe"),
        "DDK": cell("tʃʰuʈinʌg"),
        "KkP": cell("tʃʰoʈ", qualifier="(135)"),
        "DkR": cell("tʃʰoʈimoʈi", "2", qualifier="(135)"),
        "SkP": cell("tʃʰʌwaʈ", "3", qualifier="(135)"),
    }),
    144: ("heavy", {
        **{site: cell(None, "") for site in SITES if site != "HIN"},
        "HIN": cell("bʰaɾi"),
    }),
    145: ("light", {
        **{
            site: cell(None, "", column="right")
            for site in SITES if site != "HIN"
        },
        "HIN": cell("hʌlka", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(141, 146):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 50
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 30
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 50
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 45
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
