#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 146-150."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_146_150_hand_keyed.tsv")
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


def cell(form, labels="1", page="57", printed="52", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1200/1600/2400-dpi rendered-page crops
# before any comparison with the legacy CSV.
ITEMS = {
    146: ("above", {
        "HIN": cell("upʌɾ"),
        "RNK": cell("upʌɾ"),
        "RNS_Sisaikhara": cell("upʌɾ"),
        "BNM": cell("upʌɾ"),
        "DkR": cell("upʌɾ"),
        "DKS": cell("upʌɾ"),
        "RKM": cell("upʌɾ"),
        "RNS_Sisana": cell("upʌɾ"),
        "KkP": cell("upʌɾ"),
        "DGC": cell("ʊpːʌɾ"),
        "SkP": cell("ʊpːʌɾ"),
        "RKB": cell("upːʌɾ"),
        "TkN": cell("upːʌɾ"),
        "DDK": cell("upːʌɾ"),
        "BNT": cell("ʊpʌɾ"),
        "CCC": cell("upːiɾi"),
    }),
    147: ("below", {
        "HIN": cell("nitʃe"),
        "BNM": cell("nitʃe"),
        "TkN": cell("nitʃe"),
        "BNT": cell("nitʃe"),
        "RNK": cell("tʌɾe", "2"),
        "RNS_Sisaikhara": cell("tʌɾe", "2"),
        "DGC": cell("tʌɾe", "2"),
        "RKB": cell("tʌɾe", "2"),
        "RKM": cell("tʌɾe", "2"),
        "RNS_Sisana": cell("tʌɾe", "2"),
        "DDK": cell("tʌɾe", "2"),
        "KkP": cell("tʌɾe", "2"),
        "DkR": cell("tæɾe", "2"),
        "SkP": cell("tʌɾʌ", "2"),
        "DKS": cell("ʈʌɾe", "2"),
        "CCC": cell("eʈːo", "3"),
    }),
    148: ("white", {
        "HIN": cell("sʌfed"),
        "SkP": cell("sʌpʰed"),
        "RNK": cell("seto", "2"),
        "RNS_Sisaikhara": cell("seto", "2"),
        "RKB": cell("seto", "2"),
        "TkN": cell("seto", "2"),
        "RKM": cell("seto", "2"),
        "RNS_Sisana": cell("seto", "2"),
        "BNM": cell(
            "seta / bʰuɾo", "2 / 4", page="57 / 58", printed="52 / 53",
            column="right / left",
        ),
        "KkP": cell("set", "2"),
        "DGC": cell("ʊdʒːʌɾ", "3"),
        "DKS": cell("ʊdʒːʌɾ", "3"),
        "DkR": cell("ʊɖːal", "3"),
        "DDK": cell("uɖːaɾ", "3"),
        "BNT": cell("bʰuɾo", "4", page="58", printed="53", column="left"),
        "CCC": cell("goɾʌhʌɾ", "5", page="58", printed="53", column="left"),
    }),
    149: ("black", {
        "HIN": cell("kala", page="58", printed="53", column="left"),
        "BNM": cell("kala", page="58", printed="53", column="left"),
        "RNK": cell("kaɾo", page="58", printed="53", column="left"),
        "TkN": cell("kaɾo", page="58", printed="53", column="left"),
        "BNT": cell("kaɾo", page="58", printed="53", column="left"),
        "RNS_Sisaikhara": cell("kaɾo", page="58", printed="53", column="left"),
        "RNS_Sisana": cell("kaɾi", page="58", printed="53", column="left"),
        "RKB": cell("kaɾa", page="58", printed="53", column="left"),
        "DGC": cell("kʌɾija", page="58", printed="53", column="left"),
        "DkR": cell("kʌɾija", page="58", printed="53", column="left"),
        "SkP": cell("kʌɾija", page="58", printed="53", column="left"),
        "DKS": cell("kʌɾija", page="58", printed="53", column="left"),
        "DDK": cell("kʌɾija", page="58", printed="53", column="left"),
        "KkP": cell("kʌɾija", page="58", printed="53", column="left"),
        "RKM": cell("kʌɾo", page="58", printed="53", column="left"),
        "CCC": cell("kʌɾiʌ", page="58", printed="53", column="left"),
    }),
    150: ("red", {
        site: cell("lal", page="58", printed="53", column="left") for site in SITES
    }),
}


def main() -> None:
    rows = []
    for item in range(146, 151):
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
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 81
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 76
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
