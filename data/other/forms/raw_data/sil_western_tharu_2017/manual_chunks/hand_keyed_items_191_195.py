#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 191-195."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_191_195_hand_keyed.tsv")
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


def cell(form, labels="1", page="65", printed="60", column="right", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1800-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    191: ("it burns; it burned", {
        "HIN": cell("dʒʌlta / dʒʌli", column="left"),
        "DGC": cell(
            "dʒʌla / bʌɾʌta / bʌɾʌtʰæ̃", labels="1 / 2 / 2",
            column="left / right"
        ),
        "KkP": cell("dʒʌrgjʌl", column="left"),
        "DkR": cell("bʌɾʌta / bʌɾʌl", labels="2", column="left"),
        "SkP": cell("bʌɾʌtɾʌhe", labels="2", column="left"),
        "DKS": cell("bʌɾʌta / dʒʌɾʌgʌjji", labels="2", column="left"),
        "CCC": cell("bʌɾʌi", labels="2", column="left"),
        "DDK": cell("bʌɾʌlba / bʌɾʌtehæ", labels="2", column="left"),
        "RNK": cell("pʌdʒʌɾ / pʌdʒʌɾʌt", labels="3"),
        "RNS_Sisaikhara": cell("pʌdʒʌɾ / pʌdʒʌɾʌt", labels="3"),
        "BNM": cell("pʌdʒʌɾ / pʌdʒʌɾʌti", labels="3"),
        "BNT": cell("pʌdʒʌɾ / pʌdʒʌɾʌti", labels="3"),
        "RKB": cell("pʌdʒʌɾʌt / pʌdʒʌɾʌt", labels="3"),
        "TkN": cell(
            "pʌdʒʌɾʌtʰæ̃ / pʌ", labels="3",
            qualifier="second response followed by literal ellipsis (...)"
        ),
        "RKM": cell(
            "pʰʌdʒʌɾʌjji / pʌ", labels="3",
            qualifier="second response followed by literal ellipsis (...)"
        ),
        "RNS_Sisana": cell(
            "pʌdʒʌɾʌt / pʌdʒʌɾʌt", labels="3",
            qualifier="second response followed by literal period (.)"
        ),
    }),
    192: ("he dies; he died", {
        "HIN": cell("mʌrta / mʌrgʌja"),
        "RNK": cell("mʌrvaro / mʌrgʌu"),
        "RNS_Sisaikhara": cell("mʌrvaro / mʌrgʌu"),
        "SkP": cell("mʌrʌtɾʌhe"),
        "DkR": cell("mʌrʌtɾʌhʌ / mʌrʌgil"),
        "RKB": cell("mʌrʌnlego / mʌrɪgao"),
        "TkN": cell("mʌrɾʌhohæ̃ / mʌr"),
        "BNT": cell("mʌrʌnevas / mʌrʌgʌo"),
        "RKM": cell("mʌrʌtʰæ̃ / mʌrʌgʊ"),
        "RNS_Sisana": cell("mʌrʌnwaro / mʌrʌgʊ"),
        "CCC": cell("mʌrʌi"),
        "DDK": cell("mʌrʌtʰæ"),
        "DGC": cell("mʌrʌtʰæ̃"),
        "KkP": cell("mʌrgjʌl"),
        "DKS": cell("moj / mugil", labels="2"),
        "BNM": cell("girʌgaja", labels="3"),
    }),
    193: ("kill!; he killed", {
        "HIN": cell("maro / marta"),
        "RNK": cell("marʌt / mari"),
        "RNS_Sisaikhara": cell("mar / mari"),
        "RKM": cell("mar / mari"),
        "BNM": cell("mari / marʌna"),
        "DkR": cell("mʌrʌt / marʌl", qualifier="second response: (192)"),
        "SkP": cell("mʌr / mʌrʌnu", qualifier="second response: (192)"),
        "RKB": cell("marʌdæ / mara"),
        "TkN": cell("maro / mari"),
        "DKS": cell("mʌrʌnu"),
        "BNT": cell("marʌdo / marʌdithi"),
        "RNS_Sisana": cell("marhe / mari"),
        "DDK": cell("mar / marʌtʰæ"),
        "DGC": cell("marʌl"),
        "KkP": cell("mardarʌl"),
        "CCC": cell(None),
    }),
    194: ("it flies; it flew", {
        "HIN": cell("ʊɽti / ʊɽgʌji"),
        "RNK": cell("ʊɽʌt"),
        "RNS_Sisaikhara": cell("ʊɽʌɾʌu / ʊɽgʌji"),
        "BNM": cell("ʊɽʌtihæ / ʊɽʌgaji"),
        "DGC": cell("ʊɽʌʈ"),
        "DkR": cell("ʊɽʌta / ʊɾʌt"),
        "SkP": cell("uɖʌtʰæ / uɖʌtrʌhæ"),
        "RKB": cell("uɖʌt / uɖʌgaji"),
        "TkN": cell("ʊɽɾʌhihæ̃ʊɽʌt", page="66", printed="61", column="left"),
        "DKS": cell("ʊɾʌʈa / ʊɾʌgʌjel", page="66", printed="61", column="left"),
        "BNT": cell("uɖʌtihæ / uɖʌgʌtʰi", page="66", printed="61", column="left"),
        "RKM": cell("uɽʌtrʌh / uɽʌtrʌhe", page="66", printed="61", column="left"),
        "RNS_Sisana": cell("uɾːɪhæ̃ / uɾːɪrʌhẽ", page="66", printed="61", column="left"),
        "CCC": cell("uɖiʌi", page="66", printed="61", column="left"),
        "DDK": cell("uɽʌt / uɽʌtɾʌhæ", page="66", printed="61", column="left"),
        "KkP": cell("uɾgjal", page="66", printed="61", column="left"),
    }),
    195: ("walk!; he walked", {
        "HIN": cell("tʃʌlo / tʃʌla", page="66", printed="61", column="left"),
        "BNM": cell("tʃʌlo / tʃʌla", page="66", printed="61", column="left"),
        "BNT": cell("tʃʌlo / tʃʌla", page="66", printed="61", column="left"),
        "SkP": cell("tʃʌl / dʒatrʌhʌl", page="66", printed="61", column="left"),
        "RKB": cell(
            "tʃʌlo / tʃʌlodæ / negʌtʰæ / negʌtrʌhe",
            labels="1 / 1 / 2 / 2", page="66", printed="61", column="left"
        ),
        "TkN": cell("tʃʌl / tʃʌldʌi", page="66", printed="61", column="left"),
        "DKS": cell("tʃʌlo / gʌjʌgʌjjil", page="66", printed="61", column="left"),
        "RNS_Sisaikhara": cell("tʃʌlo / tʃʌlorʌhẽ", page="66", printed="61", column="left"),
        "DDK": cell(
            "tʃol / dʒajta / negʌtʰæ / negʌtrʌhe",
            labels="1 / 1 / 2 / 2", page="66", printed="61", column="left"
        ),
        "KkP": cell("tʃʌlgjʌl", page="66", printed="61", column="left"),
        "RNK": cell("negʌtʰæ / negʌtrʌhe", labels="2", page="66", printed="61", column="left"),
        "RNS_Sisana": cell("negʌtʰæ / negʌtrʌhe", labels="2", page="66", printed="61", column="left"),
        "DGC": cell(
            "negʌŋʌt / ɾæ̃ŋʌg", labels="2 / 5", page="66", printed="61",
            column="left"
        ),
        "RKM": cell(
            "nigʌtrʌh / negʌt", labels="2", page="66", printed="61",
            column="left", qualifier="second response followed by literal ellipsis (...)"
        ),
        "DkR": cell("næɽ / næɽʌl", labels="2", page="66", printed="61", column="left"),
        "CCC": cell("bulʌi / gʰɪmʌi", labels="3 / 4", page="66", printed="61", column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(191, 196):
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 79
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 145
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 135
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
