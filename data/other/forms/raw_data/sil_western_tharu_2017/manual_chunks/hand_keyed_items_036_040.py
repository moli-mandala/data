#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 36-40."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_036_040_hand_keyed.tsv")
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


def cell(form, labels, page=38, printed=33, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1600-dpi crops before legacy comparison.
ITEMS = {
    36: ("rope", {
        "HIN": cell("ɾʌsːi", "1"),
        "RNS_Sisaikhara": cell("ɾʌsːi", "1"),
        "BNM": cell("ɾʌsːi / ɾʌsa", "1 / 1", qualifier="second response: (thick)"),
        "DkR": cell("ɾʌsːi", "1"),
        "SkP": cell("ɾʌsːi", "1"),
        "RKB": cell("ɾʌsːi / bʌɾha", "1 / 4", qualifier="second response: (thick)"),
        "TkN": cell("ɾʌsːi", "1"),
        "RKM": cell("ɾʌsːi", "1"),
        "RNS_Sisana": cell("ɾʌsːi", "1"),
        "RNK": cell("ɾʌsti", "1"),
        "BNT": cell("ɾʌsi / bʌɾːa", "1 / 4", qualifier="second response: (thick)"),
        "KkP": cell("hʌsia", "1"),
        "DGC": cell("lʌsʌɾi / ʊbʰan", "2 / 3", qualifier="second response: (thick)"),
        "DKS": cell("lʌsʌɾi", "2"),
        "DDK": cell("lʌsʌɾi", "2"),
        "CCC": cell("dʒeoːɾʰi", "6"),
    }),
    37: ("thread", {
        "HIN": cell("ɖʰaga", "1"),
        "RNK": cell("ɖoɾa", "1"),
        "RNS_Sisaikhara": cell("ɖoɾa", "1"),
        "BNM": cell("ɖoɾa", "1"),
        "DGC": cell("ɖoɾa", "1"),
        "DkR": cell("ɖoɾa", "1"),
        "RKB": cell("ɖoɾa", "1"),
        "RKM": cell("ɖoɾa", "1"),
        "RNS_Sisana": cell("ɖoɾa", "1"),
        "TkN": cell("doɾa", "1"),
        "KkP": cell("doɾa", "1"),
        "DKS": cell("doɾa / sut", "1 / 2"),
        "BNT": cell("doɖa", "1"),
        "CCC": cell("doaɾa / sut", "1 / 2"),
        "SkP": cell("sut", "2"),
        "DDK": cell("sut", "2"),
    }),
    38: ("needle", {
        "HIN": cell("sui", "1"),
        "RNK": cell("sui", "1"),
        "RNS_Sisaikhara": cell("sui", "1"),
        "BNM": cell("sui", "1"),
        "DGC": cell("sui", "1"),
        "DkR": cell("sui", "1"),
        "SkP": cell("sui", "1"),
        "RKB": cell("sui", "1"),
        "DKS": cell("sui", "1", column="right"),
        "BNT": cell("sui", "1", column="right"),
        "CCC": cell("sui", "1", column="right"),
        "DDK": cell("sui", "1", column="right"),
        "KkP": cell("sui", "1", column="right"),
        "TkN": cell("sui", "1", column="right"),
        "RKM": cell("sʊi", "1", column="right"),
        "RNS_Sisana": cell("sʊi", "1", column="right"),
    }),
    39: ("cloth", {
        "HIN": cell("kʌpʌɽa", "1", column="right"),
        "CCC": cell("kʌpʌɖa / luɡa", "1 / 3", column="right"),
        "RNK": cell("lʌtːa", "2", column="right"),
        "RNS_Sisaikhara": cell("lʌtːa", "2", column="right"),
        "SkP": cell("lʌtːa", "2", column="right"),
        "RKB": cell("lʌtːa", "2", column="right"),
        "TkN": cell("lʌtːa", "2", column="right"),
        "BNT": cell("lʌtːa", "2", column="right"),
        "RKM": cell("lʌtːa", "2", column="right"),
        "RNS_Sisana": cell("lʌtːa", "2", column="right"),
        "BNM": cell("lʌta", "2", column="right"),
        "KkP": cell("lʌʈa", "2", column="right"),
        "DkR": cell("luɡːa", "3", column="right"),
        "DKS": cell("luɡːa", "3", column="right"),
        "DDK": cell("luɡʌɾa", "3", column="right"),
        "DGC": cell("lʊɡʌɽa", "3", column="right"),
    }),
    40: ("ring", {
        "HIN": cell("tʃʰʌkka / ãŋɡutʰi / tʃʰʌlːa", "1 / 2 / 3", column="right"),
        "SkP": cell("ãŋɡutʰi", "2", column="right"),
        "CCC": cell("jʌŋɡuti", "2", column="right"),
        "KkP": cell("ʌŋɡutʰi / mʊndʌɾi", "2 / 4", column="right", qualifier="first response: (men's); second response: (women's)"),
        "RNK": cell("tʃʰʌla", "3", column="right"),
        "RNS_Sisaikhara": cell("tʃʰʌla", "3", column="right"),
        "RNS_Sisana": cell("tʃʰʌla", "3", column="right"),
        "BNM": cell("mudʌɾija", "4", column="right", qualifier="(both)"),
        "DGC": cell("mʊndʌɾi", "4", column="right"),
        "DkR": cell("mʊndʌɾi", "4", column="right"),
        "RKB": cell("mʊndʌɾi", "4", column="right"),
        "DDK": cell("mʊndʌɾi", "4", column="right"),
        "TkN": cell("mʊndʌɾija", "4", column="right"),
        "BNT": cell("mʊndʌɾija", "4", column="right"),
        "DKS": cell("mʊndiɾi", "4", column="right"),
        "RKM": cell("mũdʌɾija", "4", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(36, 41):
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
    ) == 83
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
