#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 1-5."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_001_005_hand_keyed.tsv")
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


# Literal readings keyed by eye from the 1200-dpi crops. A spaced slash separates
# multiple source responses for one list/prompt cell; source group labels remain
# evidence only and are not converted into cognacy/etymology assertions.
ITEMS = {
    1: ("body", {
        "HIN": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "RNS_Sisaikhara": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "RNK": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "BNM": cell("ʃʌɾiɾ / batʌn", "1 / 2", 31, 26, "left"),
        "SkP": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "TkN": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "RNS_Sisana": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "RKM": cell("ʃʌɾiɾ", "1", 31, 26, "left"),
        "RKB": cell("ʃʌɾiɾ / deh", "1 / 3", 31, 26, "left"),
        "BNT": cell("ʃʌɾaɾ / bʌdʌn", "1 / 2", 31, 26, "left"),
        "DGC": cell("dẽh", "3", 31, 26, "left"),
        "DKS": cell("deh", "3", 31, 26, "left"),
        "KkP": cell("deh", "3", 31, 26, "left"),
        "DkR": cell("aŋ", "4", 31, 26, "left"),
        "DDK": cell("aŋ", "4", 31, 26, "left"),
        "CCC": cell(None, "", 31, 26, "left"),
    }),
    2: ("head", {
        "HIN": cell("sɪɾ", "1", 31, 26, "left"),
        "BNM": cell("sɪɾ / muɖ", "1 / 2", 31, 26, "left"),
        "RNS_Sisaikhara": cell("mʊɖ", "2", 31, 26, "left"),
        "RNK": cell("mʊɖ", "2", 31, 26, "left"),
        "RNS_Sisana": cell("mʊɖ", "2", 31, 26, "left"),
        "RKB": cell("muɖ", "2", 31, 26, "left"),
        "TkN": cell("muɖ", "2", 31, 26, "left"),
        "RKM": cell("muɖ", "2", 31, 26, "left"),
        "DGC": cell("muɖ / kʌpaɾ", "2 / 3", 31, 26, "right"),
        "SkP": cell("mʊɖɪja", "2", 31, 26, "right"),
        "DKS": cell("mʊɖi", "2", 31, 26, "right"),
        "CCC": cell("muːɖ", "2", 31, 26, "right"),
        "KkP": cell("muɾija", "2", 31, 26, "right"),
        "DkR": cell("kʌpaɾ", "3", 31, 26, "right"),
        "DDK": cell("kʌpaɾ", "3", 31, 26, "right"),
        "BNT": cell("ɡʰopʌɖi", "3", 31, 26, "right"),
    }),
    3: ("hair", {
        "HIN": cell("bal", "1", 31, 26, "right"),
        "BNM": cell("bal", "1", 31, 26, "right"),
        "BNT": cell("bal", "1", 31, 26, "right"),
        "RNS_Sisaikhara": cell("baɾ", "1", 31, 26, "right"),
        "RNK": cell("baɾ", "1", 31, 26, "right"),
        "DGC": cell("baɾ", "1", 31, 26, "right"),
        "SkP": cell("baɾ", "1", 31, 26, "right"),
        "RKB": cell("baɾ", "1", 31, 26, "right"),
        "TkN": cell("baɾ", "1", 31, 26, "right"),
        "RNS_Sisana": cell("baɾ", "1", 31, 26, "right"),
        "RKM": cell("baɾ", "1", 31, 26, "right"),
        "KkP": cell("baɾ", "1", 31, 26, "right"),
        "DkR": cell("bhuʈʌla", "2", 31, 26, "right"),
        "DKS": cell("bhuʈʌla", "2", 31, 26, "right"),
        "DDK": cell("bʰutla", "2", 31, 26, "right"),
        "CCC": cell("keis", "3", 31, 26, "right"),
    }),
    4: ("face", {
        "HIN": cell("tʃehʌɾa / mũh", "1 / 2", "31 / 32", "26 / 27", "right / left"),
        "DGC": cell("tʃehʌɾa / muh", "1 / 2", 32, 27, "left"),
        "RNS_Sisaikhara": cell("mo", "2", 32, 27, "left"),
        "BNM": cell("mo", "2", 32, 27, "left"),
        "RKB": cell("muh", "2", 32, 27, "left", "used most"),
        "RNK": cell("muh", "2", 32, 27, "left"),
        "DKS": cell("muh", "2", 32, 27, "left"),
        "TkN": cell("muh", "2", 32, 27, "left"),
        "DkR": cell("mũh", "2", 32, 27, "left"),
        "BNT": cell("mũh", "2", 32, 27, "left"),
        "DDK": cell("mũh", "2", 32, 27, "left"),
        "SkP": cell("mʊh", "2", 32, 27, "left"),
        "RNS_Sisana": cell("mʊh", "2", 32, 27, "left"),
        "RKM": cell("moh", "2", 32, 27, "left"),
        "KkP": cell("mũ", "2", 32, 27, "left"),
        "CCC": cell(None, "", "31 / 32", "26 / 27", "right / left"),
    }),
    5: ("eye", {
        "HIN": cell("aŋkʰ", "1", 32, 27, "left"),
        "BNM": cell("aŋkʰ", "1", 32, 27, "left"),
        "DGC": cell("aŋkʰ", "1", 32, 27, "left"),
        "DKS": cell("aŋkʰ", "1", 32, 27, "left"),
        "TkN": cell("aŋkʰ", "1", 32, 27, "left"),
        "BNT": cell("aŋkʰ", "1", 32, 27, "left"),
        "DDK": cell("aŋkʰ", "1", 32, 27, "left"),
        "KkP": cell("aŋkʰ", "1", 32, 27, "left"),
        "RNS_Sisaikhara": cell("aŋkʰi", "1", 32, 27, "left"),
        "RNK": cell("aŋkʰi", "1", 32, 27, "left"),
        "DkR": cell("aŋkʰi", "1", 32, 27, "left"),
        "RNS_Sisana": cell("aŋkʰi", "1", 32, 27, "left"),
        "SkP": cell("akʰi", "1", 32, 27, "left"),
        "RKB": cell("akʰi", "1", 32, 27, "left"),
        "RKM": cell("ãŋkʰi", "1", 32, 27, "left"),
        "CCC": cell("aiːkʰ", "2", 32, 27, "left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(1, 6):
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
                    "duplicate source code RNS; occurrence 1/2 assigned to metadata row 1/2 "
                    "(Sisaikhara/Sisana)"
                )
            if blank:
                uncertainty = "site code absent from the complete printed item block"
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": SOURCE_CODES[site],
                "Source_Code_Occurrence": OCCURRENCE[site],
                "Scope": "control" if site == "HIN" else "target",
                "PDF_Page": page, "Printed_Page": printed, "Column": column,
                "Source_Group_Labels": labels,
                "Manual_Transcription": form or "",
                "Manual_Form_Count": str(len(form.split(" / "))) if form else "0",
                "Source_Qualifier": qualifier,
                "Review_Status": "source_blank" if blank else "attested",
                "Confidence": "high", "Site_Assignment_Confidence": site_confidence,
                "Uncertainty": uncertainty, "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 80
    assert sum(row["Review_Status"] == "attested" for row in rows) == 78
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 85
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
