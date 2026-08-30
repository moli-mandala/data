#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 186-190."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_186_190_hand_keyed.tsv")
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


def cell(form, labels="1", page="64", printed="59", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1800-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    186: ("he is/was thirsty", {
        "HIN": cell("pjasa / pjasa"),
        "SkP": cell("pjasa / pjasa"),
        "RNK": cell("pjaso"),
        "RNS_Sisaikhara": cell("pjaso"),
        "RKB": cell("pjaso"),
        "BNM": cell("pjasahæ / pjasatʰa"),
        "BNT": cell("pjasahæ / pjasatʰa"),
        "DkR": cell("pjasʌl"),
        "TkN": cell("pjasohʊ̃ / pjaso"),
        "DKS": cell(
            "pjas / pjasʌn",
            qualifier=(
                "first response followed by literal ellipsis (...); second response "
                "followed by two literal periods (..)"
            ),
        ),
        "RKM": cell("pjaso / pjasoɾʌhʌgʊ"),
        "RNS_Sisana": cell("pjaso / pjasoɾʌhʊ"),
        "DDK": cell("pjasʌl / pjasʌlɾʌhʊ"),
        "DGC": cell("pjasala / pjasalrʌhe"),
        "KkP": cell("pjasʌl"),
        "CCC": cell(None),
    }),
    187: ("he sleeps; he slept", {
        "HIN": cell("soo / soja"),
        "BNM": cell("soo / soja"),
        "BNT": cell("soo / soja"),
        "RNK": cell("sodʒa / soʊɾʌhæ"),
        "RNS_Sisaikhara": cell("sodʒa / soʊɾʌhæ"),
        "RKB": cell("sojdʒa / sojjo"),
        "TkN": cell("sojdʒa / sojʌlu"),
        "RKM": cell("sojdʒa / sojgʊɾʌho"),
        "RNS_Sisana": cell("sojdʒa / sotɾʌhẽ"),
        "DGC": cell("suʈʌt", labels="2"),
        "DkR": cell("sut / sutsekʌnu", labels="2"),
        "SkP": cell("sutdʒa / sutʌnu", labels="2"),
        "DKS": cell("sutdʒa / sutʌgʌjjilʌs", labels="2"),
        "CCC": cell("sutʌi", labels="2"),
        "DDK": cell("sutdʒa / sutʌlɾʌhʊ", labels="2"),
        "KkP": cell("sutgjal", labels="2"),
    }),
    188: ("lie down!; he lay down", {
        "HIN": cell("leʈo / leʈa"),
        "BNM": cell("leʈo / leʈa"),
        "RNK": cell("leʈdʒa / leʈoɾʌhæ"),
        "RNS_Sisaikhara": cell("leʈdʒa / leʈoɾʌhæ"),
        "RNS_Sisana": cell("leʈdʒa / leʈoɾʌhæ"),
        "RKB": cell("leʈdʒa / leʈa"),
        "TkN": cell("leʈdʒa / leʈoɾʌho"),
        "BNT": cell("leʈedʒa / leʈʌgʌtʰa"),
        "RKM": cell("ledʒdʒaːleʈõɾʌhõ"),
        "KkP": cell("leʈgʌ̃jʌl", page="65", printed="60", column="left"),
        "DGC": cell("bolʌɾdʒa", labels="2", page="65", printed="60", column="left"),
        "SkP": cell(
            "ʊɽʌɾdʒa / ʊɽʌɾʌl", labels="2",
            page="65", printed="60", column="left"
        ),
        "CCC": cell(
            "ulʈʌi / pulʈʌi", labels="2 / 4", page="65", printed="60",
            column="left"
        ),
        "DKS": cell(
            "suʈo / suʈʌgʌj", labels="3", page="65", printed="60",
            column="left", qualifier="second response: (187)"
        ),
        "DDK": cell(
            "sutdʒa / sutʌl", labels="3", page="65", printed="60",
            column="left", qualifier="second response: (187)"
        ),
        "DkR": cell(
            None, page="64 / 65", printed="59 / 60", column="right / left"
        ),
    }),
    189: ("sit down; he sat do", {
        "HIN": cell("bæʈʰo / bæʈʰa", page="65", printed="60", column="left"),
        "BNM": cell("bæʈʰo / bæʈʰa", page="65", printed="60", column="left"),
        "BNT": cell("bæʈʰo / bæʈʰa", page="65", printed="60", column="left"),
        "RNK": cell("bæʈdʒa / baʈʰoɾʌʊ", page="65", printed="60", column="left"),
        "RNS_Sisaikhara": cell("bæʈdʒa / baʈʰoɾʌʊ", page="65", printed="60", column="left"),
        "DGC": cell("bæʈʰʌl", page="65", printed="60", column="left"),
        "DkR": cell("bæʈʰ / bæʈʰʌl", page="65", printed="60", column="left"),
        "SkP": cell("bæʈʰdʒa / bæʈʰʌl", page="65", printed="60", column="left"),
        "RKB": cell("bæʈʰʌgʌo / bæʈʰa", page="65", printed="60", column="left"),
        "TkN": cell("bæʈʰdʒa / bæʈʰoɾʌh", page="65", printed="60", column="left"),
        "RNS_Sisana": cell("bæʈʰdʒa / bæʈʰoɾʌh", page="65", printed="60", column="left"),
        "DKS": cell("bæʈʰai / bæʈʰʌgʌjjinʊ", page="65", printed="60", column="left"),
        "RKM": cell("bæʈʰdʒa / bæʈoɾʌho", page="65", printed="60", column="left"),
        "CCC": cell("betʌi / besʌi", labels="1 / 2", page="65", printed="60", column="left"),
        "DDK": cell("bæʈtaji / bæʈʌlɾʌhʊ", page="65", printed="60", column="left"),
        "KkP": cell("bæʈgjʌl", page="65", printed="60", column="left"),
    }),
    190: ("give!; he gave", {
        "HIN": cell("do / doja", labels="1", page="65", printed="60", column="left"),
        "BNM": cell("do / doja", labels="1", page="65", printed="60", column="left"),
        "BNT": cell("do / dija", labels="1", page="65", printed="60", column="left"),
        "RNK": cell("dæjʌdæ / dʌiɾʌhæ", labels="2", page="65", printed="60", column="left"),
        "RNS_Sisaikhara": cell("dæjʌdæ / dʌiɾʌhæ", labels="2", page="65", printed="60", column="left"),
        "DGC": cell("de", labels="2", page="65", printed="60", column="left"),
        "DkR": cell("de / dehʌl", labels="2", page="65", printed="60", column="left"),
        "SkP": cell("dedʌja / djal", labels="2", page="65", printed="60", column="left"),
        "RKB": cell("dedæ / dija", labels="2", page="65", printed="60", column="left"),
        "TkN": cell("dæjde / dejdʌi", labels="2", page="65", printed="60", column="left"),
        "DKS": cell("de / dædenʊ", labels="2", page="65", printed="60", column="left"),
        "RKM": cell("dæidæ / dæidæi", labels="2", page="65", printed="60", column="left"),
        "RNS_Sisana": cell("dejʌde / dejʌdɪ", labels="2", page="65", printed="60", column="left"),
        "CCC": cell("dei", labels="2", page="65", printed="60", column="left"),
        "DDK": cell("dæitæ / delʌs", labels="2", page="65", printed="60", column="left"),
        "KkP": cell("dedʌhʌl", labels="2", page="65", printed="60", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(186, 191):
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
                    "duplicate source code RNS; within-item occurrence order assigned to "
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 140
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 130
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
