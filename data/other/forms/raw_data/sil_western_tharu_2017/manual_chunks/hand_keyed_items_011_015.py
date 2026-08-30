#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 11-15."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_011_015_hand_keyed.tsv")
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


def cell(form, labels, page, printed, column, qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1200/1600-dpi crops before legacy comparison.
# Repeated identical responses are retained when the source prints them under
# separate comparison groups.
ITEMS = {
    11: ("breast", {
        "HIN": cell("stʌn / tʃʰati", "1 / 5", 33, 28, "left"),
        "RNS_Sisaikhara": cell("tʃutʃi / dudʰ / tʃutʃi", "2 / 3 / 5", 33, 28, "left"),
        "RNK": cell("tʃutʃi / tʃutʃi", "2 / 5", 33, 28, "left"),
        "RNS_Sisana": cell("tʃutʃi / tʃutʃi", "2 / 5", 33, 28, "left"),
        "BNM": cell("tʃutʃ", "2", 33, 28, "left"),
        "BNT": cell("tʃutʃ", "2", 33, 28, "left"),
        "SkP": cell("dudʰ", "3", 33, 28, "left"),
        "RKB": cell("dudʰ", "3", 33, 28, "left"),
        "DkR": cell("dudʰ", "3", 33, 28, "left"),
        "TkN": cell("dudʰ", "3", 33, 28, "left"),
        "DGC": cell("dudʰ", "3", 33, 28, "left"),
        "KkP": cell("dudʰ", "3", 33, 28, "left"),
        "DKS": cell("dʊʈ", "3", 33, 28, "left"),
        "DDK": cell("ɖuɖʰ", "3", 33, 28, "left"),
        "CCC": cell("dudʰaktʃʰaʈi", "4", 33, 28, "left"),
        "RKM": cell(None, "", 33, 28, "left"),
    }),
    12: ("belly", {
        "HIN": cell("peʈ", "1", 33, 28, "left"),
        "RNS_Sisaikhara": cell("peʈ", "1", 33, 28, "left"),
        "RNK": cell("peʈ", "1", 33, 28, "left"),
        "BNM": cell("peʈ", "1", 33, 28, "left"),
        "DGC": cell("peʈ", "1", 33, 28, "left"),
        "DkR": cell("peʈ", "1", 33, 28, "left"),
        "RKB": cell("peʈ", "1", 33, 28, "left"),
        "TkN": cell("peʈ", "1", 33, 28, "left"),
        "DKS": cell("peʈ", "1", 33, 28, "left"),
        "BNT": cell("peʈ", "1", 33, 28, "left"),
        "RNS_Sisana": cell("peʈ", "1", 33, 28, "left"),
        "RKM": cell("peʈ", "1", 33, 28, "left"),
        "SkP": cell("pjaʈ", "1", 33, 28, "left"),
        "DDK": cell("pjeʈ", "1", 33, 28, "right"),
        "CCC": cell("peit", "1", 33, 28, "right"),
        "KkP": cell("peʈʰ", "1", 33, 28, "right"),
    }),
    13: ("arm", {
        "HIN": cell("bãh / haʈʰ", "1 / 2", 33, 28, "right"),
        "DkR": cell("bãh / haʈʰ", "1 / 2", 33, 28, "right"),
        "RNS_Sisaikhara": cell("bãh / haʈʰ", "1 / 2", 33, 28, "right"),
        "CCC": cell("bahiː", "1", 33, 28, "right"),
        "RNS_Sisana": cell("haʈʰ", "2", 33, 28, "right"),
        "RNK": cell("haʈʰ", "2", 33, 28, "right"),
        "DGC": cell("haʈʰ", "2", 33, 28, "right"),
        "DDK": cell("haʈʰ", "2", 33, 28, "right"),
        "DKS": cell("haʈʰ / pãtʃa", "2 / 3", 33, 28, "right"),
        "RKB": cell("haʈʰ", "2", 33, 28, "right"),
        "KkP": cell("haʈʰ", "2", 33, 28, "right"),
        "TkN": cell("hatʰ", "2", 33, 28, "right"),
        "RKM": cell("hatʰ", "2", 33, 28, "right"),
        "BNM": cell("pãtʃa", "3", 33, 28, "right"),
        "BNT": cell("pãtʃa", "3", 33, 28, "right"),
        "SkP": cell("kohʌni", "4", 33, 28, "right"),
    }),
    14: ("elbow", {
        "HIN": cell("kuhʌni / kohʌni", "1 / 1", 33, 28, "right"),
        "BNM": cell("kʊhʌni", "1", 33, 28, "right"),
        "RNS_Sisaikhara": cell("kahʌni", "1", 33, 28, "right"),
        "RNK": cell("kohʌni", "1", 33, 28, "right"),
        "RKB": cell("kohʌni", "1", 33, 28, "right"),
        "TkN": cell("kohʌni", "1", 33, 28, "right"),
        "RKM": cell("kohʌni", "1", 33, 28, "right"),
        "RNS_Sisana": cell("kohʌni", "1", 33, 28, "right"),
        "BNT": cell("koni", "1", 33, 28, "right"),
        "CCC": cell("kehuni", "1", 33, 28, "right"),
        "KkP": cell("kihoni", "1", 33, 28, "right"),
        "DkR": cell("ʈʰɪhũn", "2", 33, 28, "right"),
        "DDK": cell("ʈhehoni", "2", 33, 28, "right"),
        "DKS": cell("ʈʌjuni / ɡãʈi", "2 / 4", 33, 28, "right"),
        "SkP": cell("hʌtʰoɾi", "3", 33, 28, "right"),
        "DGC": cell("ɡãɳʈʰ", "4", 33, 28, "right"),
    }),
    15: ("palm", {
        "HIN": cell("hʌtʰeli", "1", 33, 28, "right"),
        "RNS_Sisaikhara": cell("hʌtʰeli / hʌtɔɾi", "1 / 2", "33 / 34", "28 / 29", "right / left"),
        "BNM": cell("hʌtʰeli", "1", 33, 28, "right"),
        "BNT": cell("hʌtʰeli", "1", 33, 28, "right"),
        "RNK": cell("hʌtɔɾi / hʌtɔɾi", "1 / 2", "33 / 34", "28 / 29", "right / left"),
        "TkN": cell("hʌtɔɾi / hʌtɔɾi", "1 / 2", "33 / 34", "28 / 29", "right / left"),
        "RKM": cell("hʌtɔɾi / hʌtɔɾi", "1 / 2", "33 / 34", "28 / 29", "right / left"),
        "RNS_Sisana": cell("hʌtɔɾi", "1", 34, 29, "left"),
        "SkP": cell("hʌtʰoɾi / hʌtʰoɾi", "1 / 2", 34, 29, "left"),
        "RKB": cell("hʌtʰoɾi / hʌtʰoɾi", "1 / 2", 34, 29, "left", "first response: (13)"),
        "DKS": cell("ɡʌɽɔɾi", "2", 34, 29, "left"),
        "DDK": cell("ɡʌɽɔɾi / ɡadi", "2 / 3", 34, 29, "left"),
        "KkP": cell("ɡʌɖɔɾi", "2", 34, 29, "left"),
        "DGC": cell("ɡadi", "3", 34, 29, "left"),
        "DkR": cell("hat", "4", 34, 29, "left", "(13)"),
        "CCC": cell("tʌɾʌhʌtʰi", "5", 34, 29, "left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(11, 16):
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
                    "metadata row order; sole group occurrence assigned to metadata row 1 "
                    "(Sisaikhara)"
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 97
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 89
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
