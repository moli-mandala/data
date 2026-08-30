#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 196-200."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_196_200_hand_keyed.tsv")
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


def cell(form, labels="1", page="66", printed="61", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900-dpi rendered-page crops before any
# comparison with the legacy TSV. Item 200 continues onto physical p.67.
ITEMS = {
    196: ("run!; he ran", {
        "HIN": cell("dɔro / dɔra"),
        "DGC": cell("dɔrʌna"),
        "DkR": cell("dɔr / dɔrʌnu"),
        "SkP": cell("dɔrʌtahæ / dɔrʌt"),
        "RKB": cell("dɔdo / dɔda / bʰadʒ / bʰadʒʌt", labels="1 / 1 / 2 / 2"),
        "TkN": cell("dɔrdʒa / dɔrrʌho"),
        "DKS": cell("dɔr / dɔrʌnʊ"),
        "RNS_Sisaikhara": cell(
            "dɔrre / bʰadʒ / bʰadʒo / bʰadʒgʌʊ / dʒʰʌno",
            labels="1 / 2 / 2 / 2 / 3",
        ),
        "DDK": cell("dɔrtæ / dɔrʌthæ"),
        "KkP": cell("dɔrgjʌl"),
        "RNK": cell("bʰodʒdʒa / bʰodʒgʌu", labels="2"),
        "RNS_Sisana": cell("bʰadʒ / bʰadʒo", labels="2"),
        "BNM": cell("bʰodʒ", labels="2"),
        "BNT": cell("bʰodʒ / bʰodʒo", labels="2"),
        "RKM": cell(
            "bʰadʒrʌʊ / bʰadʒ", labels="2",
            qualifier="second response followed by literal ellipsis (...)"
        ),
        "CCC": cell(None),
    }),
    197: ("go!; he went", {
        "HIN": cell("dʒao / gajo"),
        "RKB": cell("dʒao / gʌo", column="right"),
        "BNT": cell("dʒao / gʌo", column="right"),
        "DGC": cell("dʒa", column="right"),
        "SkP": cell("dʒatro / dʒatrʌhʌl", column="right"),
        "TkN": cell("dʒao / gʌʊ", column="right"),
        "DKS": cell(
            "dʒao / gʌjigɪl", column="right",
            qualifier="responses separated by literal colon (:)"
        ),
        "CCC": cell("dʒæʌb", column="right"),
        "DDK": cell("dʒa / tʃʌlgʌilʌs", column="right"),
        "KkP": cell("gʌjil", column="right", qualifier="response followed by (past)"),
        "BNM": cell(
            "tʃʌlo / tʃaja", labels="2", column="right",
            qualifier="second response followed by (195)"
        ),
        "DkR": cell("tʃʌldʒa / tʃʌlgil", labels="2", column="right"),
        "RKM": cell("tʃʌl", labels="2", column="right"),
        "RNS_Sisaikhara": cell(
            "tʃʌl / gʌʊ / bʰodʒdʒa", labels="2 / 2 / 3", column="right",
            qualifier=(
                "second group-2 response followed by (195); group-3 response "
                "followed by literal ellipsis (...) and (196)"
            ),
        ),
        "RNK": cell(
            "bʰodʒdʒa", labels="3", column="right",
            qualifier="response followed by literal ellipsis (...) and (196)"
        ),
        "RNS_Sisana": cell(
            "bʰodʒdʒa", labels="3", column="right",
            qualifier="response followed by literal ellipsis (...) and (196)"
        ),
    }),
    198: ("come!; he came", {
        "HIN": cell("ao / ajo", column="right"),
        "RNK": cell("ao / ajo", column="right"),
        "RNS_Sisaikhara": cell("ao / are", column="right"),
        "BNM": cell(
            "ʊlʌ̃gao", column="right",
            qualifier="prefix ʊlʌ̃g enclosed in literal parentheses"
        ),
        "DGC": cell("a", column="right"),
        "DkR": cell("a / ail", column="right"),
        "SkP": cell("adʒa / ahegɪl", column="right"),
        "RKB": cell("aidʒa / aigo", column="right"),
        "TkN": cell("a / aorʌhæ̃", column="right"),
        "DKS": cell("ao / agʌjɪl", column="right"),
        "BNT": cell("ao / awʌhʌ̃", column="right"),
        "RKM": cell("ao / adʒa", column="right"),
        "RNS_Sisana": cell("ajːdʒa / ajʌgʊ", column="right"),
        "CCC": cell("awʌi", column="right"),
        "DDK": cell("a / ailʌs", column="right"),
        "KkP": cell("ajil", column="right"),
    }),
    199: ("speak!; he spoke", {
        "HIN": cell("bolo / bola", column="right"),
        "BNM": cell("bolo / bola", column="right"),
        "RNK": cell("bol / bolo", column="right"),
        "RNS_Sisaikhara": cell("bol / bolo", column="right"),
        "DGC": cell("bol", column="right"),
        "DkR": cell("bol / bolʌnu", column="right"),
        "SkP": cell("bol / bolʌl", column="right"),
        "DKS": cell("bol / bolːʌs", column="right"),
        "BNT": cell("bol / kʌhi", column="right"),
        "RNS_Sisana": cell("bol / bolʌtrʌhẽ", column="right"),
        "DDK": cell("bol / bolʌl", column="right"),
        "RKB": cell("mʌsko / mʌska", labels="2", column="right"),
        "TkN": cell("bʌdʌkatrʌhæ̃", labels="3", column="right"),
        "RKM": cell("kʌh / kʌhi", labels="4", column="right"),
        "KkP": cell("mʌnko / mʌnkʌl", labels="5", column="right"),
        "CCC": cell(None, column="right"),
    }),
    200: ("he hears; he heard", {
        "HIN": cell("sunta / suna", column="right"),
        "RNK": cell("suno / suni", page="67", printed="62"),
        "RNS_Sisaikhara": cell("sun / suno", page="67", printed="62"),
        "BNM": cell("suno / suna", page="67", printed="62"),
        "DGC": cell("sʊn", page="67", printed="62"),
        "DkR": cell("sun / sunːʊ", page="67", printed="62"),
        "SkP": cell("sʊn / sʊnʌl", page="67", printed="62"),
        "RKB": cell("sunlo / sunlɔ", page="67", printed="62"),
        "TkN": cell("sʊno / sʊnohæ̃", page="67", printed="62"),
        "DKS": cell("sʊno / sʊnːʊ", page="67", printed="62"),
        "BNT": cell("sʊnʌle / sʊnʌlæ", page="67", printed="62"),
        "RKM": cell("sʊn / sʊno", page="67", printed="62"),
        "RNS_Sisana": cell("sʊnle / sʊnʌtrʌhẽ", page="67", printed="62"),
        "CCC": cell("sunʌi", page="67", printed="62"),
        "DDK": cell("sunʌt / sunʌlæ̃", page="67", printed="62"),
        "KkP": cell("sunʌl", page="67", printed="62"),
    }),
}


def main() -> None:
    rows = []
    for item in range(196, 201):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 144
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 134
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
