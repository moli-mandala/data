#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 21-25."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_021_025_hand_keyed.tsv")
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


def cell(form, labels, page=35, printed=30, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1200/1600-dpi crops before legacy comparison.
ITEMS = {
    21: ("heart", {
        "HIN": cell("dɪl / hudʌi", "1 / 5"),
        "RNS_Sisaikhara": cell("dɪl / kʌledʒa", "1 / 2"),
        "BNM": cell("dɪl", "1"),
        "SkP": cell("dɪl", "1"),
        "RKB": cell("dɪl", "1"),
        "BNT": cell("dɪl", "1"),
        "RNS_Sisana": cell("dɪl", "1"),
        "KkP": cell("dɪl", "1"),
        "RNK": cell("kʌledʒa", "2"),
        "DGC": cell("kʌledʒa / dʒiw", "2 / 4"),
        "TkN": cell("kʌledʒa", "2"),
        "DkR": cell("kʌɾʌdʒa", "2"),
        "RKM": cell("kʌɾedʒa", "2"),
        "CCC": cell("koːɖha", "3"),
        "DDK": cell("dʒiu / kadʒʌɾa", "4 / 6"),
        "DKS": cell(None, ""),
    }),
    22: ("blood", {
        "HIN": cell("kʰun", "1"),
        "RNK": cell("kʰun", "1"),
        "RNS_Sisaikhara": cell("kʰun", "1"),
        "BNM": cell("kʰun", "1"),
        "SkP": cell("kʰun", "1"),
        "TkN": cell("kʰun", "1"),
        "BNT": cell("kʰun", "1"),
        "RKM": cell("kʰun", "1"),
        "RNS_Sisana": cell("kʰun", "1"),
        "DGC": cell("ɾʌkʌt", "2"),
        "DkR": cell("ɾʌkʌt", "2"),
        "RKB": cell("ɾʌkʌt", "2"),
        "DDK": cell("ɾʌkʌt", "2"),
        "KkP": cell("ɾʌkʌt", "2"),
        "DKS": cell("ɾʌɡʌt", "2"),
        "CCC": cell("ɾakʌtʰ", "2"),
    }),
    23: ("urine", {
        "HIN": cell("peʃab / mut", "1 / 2"),
        "RNK": cell("mut", "2"),
        "RNS_Sisaikhara": cell("mut", "2"),
        "BNM": cell("mut", "2"),
        "DGC": cell("mut", "2"),
        "DkR": cell("mut", "2"),
        "SkP": cell("mut", "2"),
        "RKB": cell("mut", "2"),
        "TkN": cell("mut", "2", column="right"),
        "DKS": cell("mut", "2", column="right"),
        "BNT": cell("mut", "2", column="right"),
        "RKM": cell("mut", "2", column="right"),
        "RNS_Sisana": cell("mut", "2", column="right"),
        "CCC": cell("mut / pʌsiena", "2 / 3", column="right"),
        "DDK": cell("mut", "2", column="right"),
        "KkP": cell("mute", "2", column="right"),
    }),
    24: ("feces", {
        "HIN": cell("tʌʈi", "1", column="right"),
        "RNK": cell("hʌɡas", "2", column="right"),
        "RNS_Sisaikhara": cell("hʌɡas / hʌɡʌdi / ɡuh", "2 / 2 / 4", column="right"),
        "RNS_Sisana": cell("hʌɡas / ɡuhu", "2 / 4", column="right"),
        "RKB": cell("hʌɡʌn", "2", column="right"),
        "BNT": cell("haɡʌna / hʌɡija", "2 / 2", column="right"),
        "RKM": cell("hʌɡʌna", "2", column="right"),
        "KkP": cell("hʌɡa", "2", column="right"),
        "BNM": cell("tʃʰaɖe", "3", column="right"),
        "CCC": cell("dʒaɖa", "3", column="right"),
        "DkR": cell("ɡʊ", "4", column="right"),
        "SkP": cell("ɡu", "4", column="right"),
        "DKS": cell("ɡu", "4", column="right"),
        "DGC": cell("ɡu", "4", column="right"),
        "TkN": cell("ɡuh", "4", column="right"),
        "DDK": cell("ɡuh", "4", column="right"),
    }),
    25: ("village", {
        "HIN": cell("ɡãũ", "1", column="right"),
        "RNK": cell("ɡãũ", "1", column="right"),
        "RNS_Sisaikhara": cell("ɡãũ", "1", column="right"),
        "BNM": cell("ɡãũ", "1", column="right"),
        "DGC": cell("ɡãũ", "1", column="right"),
        "DkR": cell("ɡãũ", "1", column="right"),
        "SkP": cell("ɡãũ", "1", column="right"),
        "RKB": cell("ɡãũ", "1", column="right"),
        "TkN": cell("ɡão", "1", column="right"),
        "DKS": cell("ɡão", "1", column="right"),
        "BNT": cell("ɡão", "1", column="right"),
        "RNS_Sisana": cell("ɡão", "1", column="right"),
        "DDK": cell("ɡão", "1", column="right"),
        "RKM": cell("ɡãõ", "1", column="right"),
        "CCC": cell("ɡaːu", "1", column="right"),
        "KkP": cell("ɡau", "1", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(21, 26):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 89
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
